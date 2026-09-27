"""DermaInsight AI Flask inference service.

Training, auditing, evaluation, and the saved Grad-CAM examples live in
``acne_training.ipynb``. This service only loads the exported artifact,
validates an uploaded image, performs inference, and returns a post-hoc
Grad-CAM explanation for the current image.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from matplotlib import cm
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from torchvision import models, transforms
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'
REPORT_DIR = BASE_DIR / 'reports'
RUN_STATE_PATH = MODEL_DIR / 'run_state.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ImageFile.LOAD_TRUNCATED_IMAGES = False


class SmallCNN(nn.Module):
    """Fallback architecture used if the notebook exports the baseline CNN."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == 'custom_cnn':
        return SmallCNN(num_classes)
    raise ValueError(f'Unsupported model name: {model_name}')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifacts() -> tuple[nn.Module, list[str], transforms.Compose, dict[str, Any]]:
    metadata_path = MODEL_DIR / 'model_metadata.json'
    preprocessing_path = MODEL_DIR / 'preprocessing.json'
    labels_path = MODEL_DIR / 'class_names.json'

    if not metadata_path.exists():
        raise FileNotFoundError(
            'model/model_metadata.json is missing. Run the final export cell in '
            'acne_training.ipynb first.'
        )
    if not RUN_STATE_PATH.exists():
        raise FileNotFoundError(
            'model/run_state.json is missing. Run the complete notebook before starting Flask.'
        )

    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    run_state = json.loads(RUN_STATE_PATH.read_text(encoding='utf-8'))
    if run_state.get('status') != 'completed':
        raise RuntimeError(
            'The exported model is not ready for inference: '
            f"run status is {run_state.get('status', 'unknown')!r}. Complete the notebook run first."
        )
    if metadata.get('training_run_is_smoke'):
        raise RuntimeError('Flask refuses to load a smoke-run model. Complete a full training run first.')
    if metadata.get('execution_profile') != 'full' or int(metadata.get('epochs_requested', 0)) < 8:
        raise RuntimeError('Flask requires a full model export with at least eight requested epochs.')
    preprocessing = json.loads(preprocessing_path.read_text(encoding='utf-8'))
    class_names = json.loads(labels_path.read_text(encoding='utf-8'))
    checkpoint_path = MODEL_DIR / metadata['checkpoint']

    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Exported model checkpoint is missing: {checkpoint_path}')
    actual_sha256 = sha256_file(checkpoint_path)
    metadata_sha256 = metadata.get('model_sha256')
    state_sha256 = run_state.get('model_sha256')
    if not metadata_sha256 or actual_sha256 != metadata_sha256:
        raise RuntimeError('Checkpoint checksum does not match model metadata. Re-run the notebook export.')
    if not state_sha256 or actual_sha256 != state_sha256:
        raise RuntimeError('Checkpoint checksum does not match the completed run state. Re-run the notebook export.')

    model = build_model(metadata['model_name'], len(class_names))
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    image_size = int(preprocessing['image_size'])
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=preprocessing['mean'], std=preprocessing['std']),
        ]
    )
    return model, class_names, transform, metadata


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

# These names remain module-level for simple local testing and compatibility
# with the first version of the app. The model is loaded in a background thread
# so the UI can show a real startup state instead of appearing frozen.
MODEL: nn.Module | None = None
CLASS_NAMES: list[str] = []
PREPROCESS: transforms.Compose | None = None
MODEL_METADATA: dict[str, Any] = {}
LOAD_ERROR: str | None = None

STATE_LOCK = Lock()
INFERENCE_LOCK = Lock()
EVENT_HISTORY: deque[dict[str, Any]] = deque(maxlen=100)
EVENT_SUBSCRIBERS: set[Queue[dict[str, Any]]] = set()


def _now_label() -> str:
    return datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S')


def publish_log(message: str, *, kind: str = 'runtime', level: str = 'info') -> None:
    event = {
        'id': time.time_ns(),
        'time': _now_label(),
        'kind': kind,
        'level': level,
        'message': message,
    }
    with STATE_LOCK:
        EVENT_HISTORY.append(event)
        subscribers = tuple(EVENT_SUBSCRIBERS)
    for subscriber in subscribers:
        subscriber.put(event)


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        'model_name',
        'class_names',
        'image_size',
        'execution_profile',
        'epochs_requested',
        'epochs_completed',
        'checkpoint_epoch',
        'early_stopping_patience',
        'training_run_is_smoke',
        'run_id',
        'test_rows',
        'clinical_diagnosis_supported',
        'exploratory_dataset_labels_only',
    ]
    result = {key: metadata[key] for key in keys if key in metadata}
    result['device'] = str(DEVICE)
    result['model_label'] = _display_model_name(metadata.get('model_name', 'unknown'))
    return result


def _display_model_name(model_name: str) -> str:
    return {
        'efficientnet_b0': 'EfficientNet-B0',
        'custom_cnn': 'Custom CNN',
    }.get(model_name, model_name.replace('_', ' ').title())


def runtime_snapshot() -> dict[str, Any]:
    with STATE_LOCK:
        status = 'ready' if MODEL is not None and LOAD_ERROR is None else 'error' if LOAD_ERROR else 'loading'
        return {
            'status': status,
            'error': LOAD_ERROR,
            'model': _public_metadata(MODEL_METADATA) if MODEL_METADATA else {},
            'events': list(EVENT_HISTORY),
        }


def _load_model_in_background() -> None:
    global MODEL, CLASS_NAMES, PREPROCESS, MODEL_METADATA, LOAD_ERROR

    publish_log('DermaInsight service started.', kind='system')
    publish_log('Loading model contract and checkpoint…', kind='model')
    try:
        loaded_model, loaded_classes, loaded_preprocess, loaded_metadata = load_artifacts()
        with STATE_LOCK:
            MODEL = loaded_model
            CLASS_NAMES = loaded_classes
            PREPROCESS = loaded_preprocess
            MODEL_METADATA = loaded_metadata
            LOAD_ERROR = None
        publish_log(
            f"{loaded_metadata.get('model_name', 'Model')} ready on {DEVICE}.",
            kind='model',
        )
        publish_log(
            f"Input contract: RGB {loaded_metadata.get('image_size', '?')}×{loaded_metadata.get('image_size', '?')} | "
            f"labels: {', '.join(loaded_classes)}.",
            kind='model',
        )
    except Exception as exc:  # startup must expose the failure to the UI
        with STATE_LOCK:
            LOAD_ERROR = str(exc)
        publish_log(f'Model unavailable: {exc}', kind='model', level='error')


MODEL_THREAD = Thread(target=_load_model_in_background, name='dermainsight-model-loader', daemon=True)
MODEL_THREAD.start()


def _last_conv_layer(model: nn.Module) -> nn.Module:
    layers = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
    if not layers:
        raise RuntimeError('No convolutional layer found for Grad-CAM.')
    return layers[-1]


def _gradcam_overlay(
    model: nn.Module,
    image_tensor: torch.Tensor,
    source_image: Image.Image,
    target_index: int,
) -> str:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer = _last_conv_layer(model)
    forward_handle = target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    backward_handle = target_layer.register_full_backward_hook(
        lambda _module, _grad_input, grad_output: gradients.append(grad_output[0])
    )

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(image_tensor)
            logits[0, target_index].backward()
        if not activations or not gradients:
            raise RuntimeError('Grad-CAM hooks did not receive activations and gradients.')

        activation = activations[0].detach()
        gradient = gradients[0].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * activation).sum(dim=1).squeeze(0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        display_image = source_image.copy()
        display_image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        heatmap_image = Image.fromarray(
            np.uint8(cm.magma(heatmap)[..., :3] * 255)
        ).resize(display_image.size, Image.Resampling.BILINEAR)
        overlay = Image.blend(display_image, heatmap_image.convert('RGB'), alpha=0.42)
        buffer = io.BytesIO()
        overlay.save(buffer, format='JPEG', quality=88, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        return f'data:image/jpeg;base64,{encoded}'
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


def _label_explanation(label: str) -> dict[str, str]:
    descriptions = {
        'Acne': (
            'The highest model score matches the dataset label “Acne”. '
            'It indicates what this trained classifier saw as the closest label in its training data.'
        ),
        'Comedo': (
            'The highest model score matches the dataset label “Comedo”. '
            'It indicates what this trained classifier saw as the closest label in its training data.'
        ),
        'Clear': (
            'The highest model score matches the dataset label “Clear”. '
            'This dataset label is not a verified clinical assessment of healthy skin.'
        ),
    }
    return {
        'label_context': descriptions.get(label, 'The model selected the highest-scoring dataset label.'),
        'method': 'Grad-CAM',
        'method_note': (
            'The heatmap highlights image regions that contributed most to the selected class score. '
            'This is not a YOLO-style detector: it does not mark exact acne spots or draw bounding boxes. '
            'It is an interpretability aid, not proof of a lesion boundary or a medical conclusion.'
        ),
        'score_note': (
            'The percentage is a softmax model score. It is not a calibrated probability and should not '
            'be read as diagnostic certainty.'
        ),
    }


def _error_response(message: str, status_code: int):
    publish_log(message, kind='error', level='error')
    return jsonify({'error': message}), status_code


@app.get('/')
def index():
    return render_template('index.html', status=runtime_snapshot())


@app.get('/about')
def about():
    metrics_path = REPORT_DIR / 'efficientnet_b0_test_metrics.json'
    audit_path = REPORT_DIR / 'audit_summary.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}
    audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
    return render_template(
        'about.html',
        status=runtime_snapshot(),
        metrics=metrics,
        audit=audit,
    )


@app.get('/api/status')
def api_status():
    return jsonify(runtime_snapshot())


@app.get('/events')
def events():
    subscriber: Queue[dict[str, Any]] = Queue()
    with STATE_LOCK:
        history = list(EVENT_HISTORY)
        EVENT_SUBSCRIBERS.add(subscriber)

    def generate():
        try:
            for event in history:
                yield f"data: {json.dumps(event)}\n\n"
            while True:
                try:
                    event = subscriber.get(timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except Empty:
                    yield ': keep-alive\n\n'
        finally:
            with STATE_LOCK:
                EVENT_SUBSCRIBERS.discard(subscriber)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@app.post('/predict')
def predict():
    with STATE_LOCK:
        model = MODEL
        class_names = list(CLASS_NAMES)
        preprocess = PREPROCESS
        metadata = dict(MODEL_METADATA)
        load_error = LOAD_ERROR

    if load_error or model is None or preprocess is None:
        return _error_response(load_error or 'Model is still loading. Please wait a moment.', 503)

    uploaded = request.files.get('image')
    if uploaded is None or not uploaded.filename:
        return _error_response('Please upload one image.', 400)

    safe_name = secure_filename(uploaded.filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return _error_response('Use a JPG, JPEG, PNG, or WEBP image.', 400)

    started = time.perf_counter()
    publish_log('Image received; validating the upload.', kind='input')
    try:
        opened = Image.open(uploaded.stream)
        opened.load()
        source_image = ImageOps.exif_transpose(opened).convert('RGB')
        source_format = opened.format or suffix.lstrip('.').upper()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        return _error_response(f'Uploaded file is not a valid image: {exc}', 400)

    publish_log(
        f"Image validated: {source_format} | {source_image.width}×{source_image.height} px.",
        kind='input',
    )
    tensor = preprocess(source_image).unsqueeze(0).to(DEVICE)

    with INFERENCE_LOCK:
        publish_log('Preprocessing to the model input contract.', kind='preprocess')
        publish_log('Running EfficientNet classification.', kind='inference')
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                logits = model(tensor)
                probabilities_tensor = torch.softmax(logits, dim=1)[0]
            probabilities = probabilities_tensor.detach().cpu().tolist()
            predicted_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
            predicted_label = class_names[predicted_index]
            publish_log(
                f"Prediction complete: {predicted_label} at {probabilities[predicted_index] * 100:.2f}% model score.",
                kind='inference',
            )
            publish_log('Building a Grad-CAM explanation for this image.', kind='explanation')
            explanation_image = _gradcam_overlay(model, tensor, source_image, predicted_index)
            publish_log('Grad-CAM explanation ready.', kind='explanation')
        except RuntimeError as exc:
            return _error_response(f'Model inference failed: {exc}', 500)

    result = {
        'prediction': predicted_label,
        'confidence': round(float(probabilities[predicted_index]), 6),
        'probabilities': {
            label: round(float(probability), 6)
            for label, probability in zip(class_names, probabilities)
        },
        'model': metadata.get('model_name', 'unknown'),
        'model_label': _display_model_name(metadata.get('model_name', 'unknown')),
        'device': str(DEVICE),
        'image_meta': {
            'filename': safe_name,
            'format': source_format,
            'width': source_image.width,
            'height': source_image.height,
        },
        'duration_ms': round((time.perf_counter() - started) * 1000, 1),
        'explanation': _label_explanation(predicted_label),
        'explanation_image': explanation_image,
        'notice': (
            'Exploratory dataset-label prediction only. This system is not a medical diagnosis '
            'and the current Clear label is not verified clinical clear-skin data.'
        ),
    }

    if request.args.get('format') == 'json' or request.accept_mimetypes.best == 'application/json':
        return jsonify(result)
    return render_template('index.html', status=runtime_snapshot(), result=result)


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return _error_response('Image is too large. Please upload a file under 12 MB.', 413)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
