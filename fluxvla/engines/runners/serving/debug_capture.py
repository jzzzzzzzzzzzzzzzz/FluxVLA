"""Debug capture helpers for remote VLA inference.

The capture format deliberately uses plain JSON, NPY and PNG files so a
request can be inspected without importing FluxVLA or loading a model.
"""

from __future__ import annotations

import json
import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from PIL import Image


def _safe_name(value: str) -> str:
    """Return a filesystem-safe representation of a payload key."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._') or 'value'


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _array_summary(value: np.ndarray) -> dict:
    summary = {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if value.size == 0 or not np.issubdtype(value.dtype, np.number):
        return summary
    numeric = value.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    summary['finite_fraction'] = float(finite.mean())
    if finite.any():
        finite_values = numeric[finite]
        summary.update(
            min=float(finite_values.min()),
            max=float(finite_values.max()),
            mean=float(finite_values.mean()),
            std=float(finite_values.std()),
        )
    return summary


class DebugCaptureWriter:
    """Persist selected ZMQ requests and their exact model inputs.

    Capturing is disabled unless ``root`` is provided. ``every`` and
    ``max_captures`` bound the disk overhead during a real-robot run.
    """

    def __init__(self,
                 root: str,
                 every: int = 1,
                 max_captures: int = 50,
                 image_color_order: str = 'rgb') -> None:
        if every <= 0:
            raise ValueError('debug capture interval must be positive')
        if max_captures <= 0:
            raise ValueError('debug capture maximum must be positive')
        if image_color_order not in {'rgb', 'bgr'}:
            raise ValueError('image_color_order must be rgb or bgr')
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.every = int(every)
        self.max_captures = int(max_captures)
        self.image_color_order = image_color_order
        self._captured = 0
        self._lock = threading.Lock()

    def start(self, request_index: int, obs: Mapping[str, Any],
              unnorm_key: str) -> Optional[Path]:
        """Start a capture and save the observation decoded by the server."""
        with self._lock:
            if (request_index % self.every != 0
                    or self._captured >= self.max_captures):
                return None
            self._captured += 1

        timestamp = datetime.now(timezone.utc)
        request_dir = self.root / (
            f'request_{request_index:06d}_{timestamp:%Y%m%dT%H%M%S_%fZ}')
        request_dir.mkdir(parents=True, exist_ok=False)
        self._update_metadata(
            request_dir, {
                'request_index': request_index,
                'captured_at_utc': timestamp.isoformat(),
                'unnorm_key': unnorm_key,
                'image_color_order': self.image_color_order,
                'status': 'received',
            })
        self.write_mapping(request_dir, 'raw_observation', obs, images=True)
        return request_dir

    def write_mapping(self,
                      request_dir: Path,
                      stage: str,
                      values: Mapping[str, Any],
                      images: bool = False) -> None:
        """Save tensors/arrays recursively and record scalar metadata."""
        stage_dir = request_dir / _safe_name(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        summaries = {}
        scalars = {}

        def visit(name: str, value: Any) -> None:
            array = _to_numpy(value)
            safe = _safe_name(name)
            if array is not None:
                np.save(stage_dir / f'{safe}.npy', array, allow_pickle=False)
                summaries[name] = _array_summary(array)
                if images and self._is_uint8_image(array):
                    self._save_png(stage_dir / f'{safe}.png', array)
                return
            if isinstance(value, Mapping):
                for child_key, child_value in value.items():
                    visit(f'{name}.{child_key}' if name else str(child_key),
                          child_value)
                return
            if isinstance(value, (list, tuple)):
                if not value:
                    scalars[name] = []
                    return
                for index, child_value in enumerate(value):
                    visit(f'{name}.{index}', child_value)
                return
            if value is None or isinstance(value, (str, bool, int, float)):
                if isinstance(value, float) and not math.isfinite(value):
                    value = str(value)
                scalars[name] = value
            else:
                scalars[name] = repr(value)

        for key, value in values.items():
            visit(str(key), value)
        self._update_metadata(
            request_dir, {stage: {
                'arrays': summaries,
                'values': scalars,
            }})

    def write_array(self, request_dir: Path, name: str, value: Any) -> None:
        self.write_mapping(request_dir, name, {name: value})

    def finish(self, request_dir: Path, infer_time: float) -> None:
        self._update_metadata(request_dir, {
            'status': 'complete',
            'infer_time_s': float(infer_time),
        })

    def fail(self, request_dir: Optional[Path], error: Exception) -> None:
        if request_dir is None:
            return
        self._update_metadata(
            request_dir, {
                'status': 'error',
                'error_type': type(error).__name__,
                'error': str(error),
            })

    @staticmethod
    def _is_uint8_image(value: np.ndarray) -> bool:
        return (value.dtype == np.uint8 and value.ndim == 3
                and value.shape[-1] in {3, 4})

    def _save_png(self, path: Path, value: np.ndarray) -> None:
        image = value
        if self.image_color_order == 'bgr' and image.shape[-1] >= 3:
            if image.shape[-1] == 3:
                image = image[..., ::-1]
            else:
                image = image[..., [2, 1, 0, 3]]
        Image.fromarray(image).save(path)

    @staticmethod
    def _update_metadata(request_dir: Path, update: dict) -> None:
        path = request_dir / 'metadata.json'
        metadata = {}
        if path.is_file():
            with path.open('r', encoding='utf-8') as file:
                metadata = json.load(file)
        metadata.update(update)
        temporary = path.with_suffix('.json.tmp')
        with temporary.open('w', encoding='utf-8') as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
            file.write('\n')
        temporary.replace(path)
