#!/usr/bin/env python3
"""Compare captured Franka ZMQ inputs with Task-1 training data.

This tool does not load a VLA checkpoint. It compares raw states against the
checkpoint's dataset statistics and samples the three training video streams
to detect resolution, camera-order, and likely RGB/BGR mismatches.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

CAMERAS = {
    'cam_front': 'observation.images.cam_front',
    'cam_wrist_left': 'observation.images.cam_wrist_left',
    'cam_wrist_right': 'observation.images.cam_wrist_right',
}
STATE_NAMES = [
    'left_x', 'left_y', 'left_z', 'left_qx', 'left_qy', 'left_qz', 'left_qw',
    'left_gripper', 'right_x', 'right_y', 'right_z', 'right_qx', 'right_qy',
    'right_qz', 'right_qw', 'right_gripper'
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--capture-dir',
        required=True,
        help='One request_* directory or its debug-capture parent')
    parser.add_argument(
        '--stats', required=True, help='Checkpoint dataset_statistics.json')
    parser.add_argument(
        '--dataset-root',
        required=True,
        nargs='+',
        help='Training LeRobot dataset roots')
    parser.add_argument('--statistic-name', default='private')
    parser.add_argument('--sample-frames', type=int, default=12)
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: CAPTURE_DIR/comparison)')
    return parser.parse_args()


def request_dirs(path: Path) -> list[Path]:
    if (path / 'metadata.json').is_file():
        return [path]
    requests = sorted(
        p for p in path.glob('request_*') if (p / 'metadata.json').is_file())
    if not requests:
        raise FileNotFoundError(f'No captured requests found below {path}')
    return requests


def summarize_images(images: Iterable[np.ndarray]) -> dict:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    shapes = Counter()
    count = 0
    for image in images:
        if image.ndim != 3 or image.shape[-1] < 3:
            continue
        image = image[..., :3]
        shapes[str(list(image.shape))] += 1
        resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        values = resized.reshape(-1, 3).astype(np.float64)
        channel_sum += values.sum(axis=0)
        channel_square_sum += np.square(values).sum(axis=0)
        pixel_count += values.shape[0]
        count += 1
    if not pixel_count:
        return {'count': 0, 'shapes': dict(shapes)}
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_square_sum / pixel_count - mean**2, 0.0)
    return {
        'count': count,
        'shapes': dict(shapes),
        'rgb_mean': mean.tolist(),
        'rgb_std': np.sqrt(variance).tolist(),
    }


def load_capture(requests: list[Path]) -> tuple[dict, list[str], dict]:
    states = []
    prompts = []
    images = {key: [] for key in CAMERAS}
    model_input_manifests = []
    action_manifests = []
    for request in requests:
        metadata = json.loads((request / 'metadata.json').read_text())
        qpos = request / 'raw_observation/qpos.npy'
        if qpos.is_file():
            states.append(np.load(qpos, allow_pickle=False).reshape(-1))
        prompt = metadata.get('raw_observation',
                              {}).get('values', {}).get('task_description')
        if prompt is not None:
            prompts.append(str(prompt))
        for camera in CAMERAS:
            path = request / f'raw_observation/{camera}.npy'
            if path.is_file():
                images[camera].append(np.load(path, allow_pickle=False))
        if 'model_inputs' in metadata:
            model_input_manifests.append(metadata['model_inputs'])
        for key in ('normalized_actions', 'denormalized_actions'):
            if key in metadata:
                action_manifests.append({key: metadata[key]})
    return {
        'states': states,
        'images': images,
        'request_count': len(requests),
        'model_input_manifests': model_input_manifests,
        'action_manifests': action_manifests,
    }, prompts, json.loads((requests[-1] / 'metadata.json').read_text())


def read_training_tasks(roots: list[Path]) -> list[str]:
    tasks = set()
    for root in roots:
        path = root / 'meta/tasks.jsonl'
        if not path.is_file():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get('task')
            if value:
                tasks.add(str(value))
    return sorted(tasks)


def select_evenly(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[index] for index in indices]


def decode_training_images(roots: list[Path], samples: int):
    images = {key: [] for key in CAMERAS}
    sources = {key: [] for key in CAMERAS}
    for camera, video_key in CAMERAS.items():
        paths = []
        for root in roots:
            paths.extend(
                sorted(
                    (root / 'videos').glob(f'**/{video_key}/episode_*.mp4')))
        for sample_index, path in enumerate(select_evenly(paths, samples)):
            video = cv2.VideoCapture(str(path))
            frame_count = max(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            frame_index = (sample_index * 104729) % frame_count
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = video.read()
            video.release()
            if not ok or bgr is None:
                continue
            images[camera].append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            sources[camera].append(f'{path}:{frame_index}')
    return images, sources


def state_report(states: list[np.ndarray], statistics: dict) -> dict:
    if not states:
        return {'count': 0, 'error': 'No raw_observation/qpos.npy found'}
    values = np.stack(states)
    if values.shape[1] != len(STATE_NAMES):
        return {
            'count': len(states),
            'shape': list(values.shape),
            'error':
            f'Expected 16D dual-Franka eepose, got {values.shape[1]}D',
        }
    mean = np.asarray(statistics['mean'])
    std = np.asarray(statistics['std'])
    q01 = np.asarray(statistics['q01'])
    q99 = np.asarray(statistics['q99'])
    minimum = np.asarray(statistics['min'])
    maximum = np.asarray(statistics['max'])
    captured_mean = values.mean(axis=0)
    z_score = (captured_mean - mean) / (std + 1e-8)
    outside_quantiles = np.logical_or(values < q01, values > q99)
    outside_range = np.logical_or(values < minimum, values > maximum)
    dimensions = []
    for index, name in enumerate(STATE_NAMES):
        dimensions.append({
            'index':
            index,
            'name':
            name,
            'capture_mean':
            float(captured_mean[index]),
            'training_mean':
            float(mean[index]),
            'training_std':
            float(std[index]),
            'z_score':
            float(z_score[index]),
            'outside_q01_q99_fraction':
            float(outside_quantiles[:, index].mean()),
            'outside_min_max_fraction':
            float(outside_range[:, index].mean()),
        })
    return {
        'count': len(states),
        'shape': list(values.shape),
        'outside_q01_q99_fraction': float(outside_quantiles.mean()),
        'outside_min_max_fraction': float(outside_range.mean()),
        'left_quaternion_norm': np.linalg.norm(values[:, 3:7],
                                               axis=1).tolist(),
        'right_quaternion_norm': np.linalg.norm(values[:, 11:15],
                                                axis=1).tolist(),
        'dimensions': dimensions,
    }


def image_report(capture_images: dict, training_images: dict) -> dict:
    report = {}
    for camera in CAMERAS:
        capture = summarize_images(capture_images[camera])
        training = summarize_images(training_images[camera])
        result = {'capture': capture, 'training': training}
        if capture.get('count') and training.get('count'):
            capture_mean = np.asarray(capture['rgb_mean'])
            training_mean = np.asarray(training['rgb_mean'])
            rgb_distance = np.linalg.norm(capture_mean - training_mean)
            bgr_distance = np.linalg.norm(capture_mean[::-1] - training_mean)
            result.update(
                rgb_mean_distance=float(rgb_distance),
                channel_swapped_mean_distance=float(bgr_distance),
                likely_rgb_bgr_swap=bool(bgr_distance + 5.0 < rgb_distance),
            )
        report[camera] = result
    return report


def make_montage(path: Path, capture_images: dict,
                 training_images: dict) -> None:
    tile_size = (320, 240)
    canvas = Image.new('RGB', (tile_size[0] * 3, tile_size[1] * 2 + 40),
                       'white')
    draw = ImageDraw.Draw(canvas)
    for column, camera in enumerate(CAMERAS):
        draw.text((column * tile_size[0] + 4, 2), camera, fill='black')
        pairs = [capture_images[camera], training_images[camera]]
        for row, candidates in enumerate(pairs):
            if not candidates:
                continue
            image = Image.fromarray(candidates[0][..., :3]).convert('RGB')
            image.thumbnail(tile_size)
            x = column * tile_size[0] + (tile_size[0] - image.width) // 2
            y = 20 + row * tile_size[1] + (tile_size[1] - image.height) // 2
            canvas.paste(image, (x, y))
    draw.text((4, 22),
              'captured',
              fill='white',
              stroke_width=2,
              stroke_fill='black')
    draw.text((4, 22 + tile_size[1]),
              'training',
              fill='white',
              stroke_width=2,
              stroke_fill='black')
    canvas.save(path)


def markdown_report(report: dict) -> str:
    lines = [
        '# Franka inference/training comparison',
        '',
        f"Captured requests: {report['capture']['request_count']}",
        '',
        f"Prompt exact match: {report['prompt']['all_match_training_tasks']}",
        '',
        '## State',
        '',
        f"Outside training q01-q99: "
        f"{report['state'].get('outside_q01_q99_fraction', 'n/a')}",
        f"Outside training min-max: "
        f"{report['state'].get('outside_min_max_fraction', 'n/a')}",
        '',
        '| dim | capture mean | train mean | z | outside q01-q99 |',
        '|---|---:|---:|---:|---:|',
    ]
    for item in report['state'].get('dimensions', []):
        lines.append(
            f"| {item['index']} {item['name']} | "
            f"{item['capture_mean']:.6f} | {item['training_mean']:.6f} | "
            f"{item['z_score']:.2f} | "
            f"{item['outside_q01_q99_fraction']:.1%} |")
    lines.extend([
        '', '## Images', '',
        '| camera | capture shapes | train shapes | RGB/BGR warning |',
        '|---|---|---|---|'
    ])
    for camera, item in report['images'].items():
        lines.append(f"| {camera} | {item['capture'].get('shapes', {})} | "
                     f"{item['training'].get('shapes', {})} | "
                     f"{item.get('likely_rgb_bgr_swap', 'n/a')} |")
    lines.extend([
        '', 'See `comparison.json` for tensor shapes/ranges and '
        '`camera_montage.png` for the labelled visual comparison.', ''
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    capture_root = Path(args.capture_dir).expanduser().resolve()
    requests = request_dirs(capture_root)
    roots = [Path(path).expanduser().resolve() for path in args.dataset_root]
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f'Dataset roots do not exist: {missing}')
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else capture_root / 'comparison')
    output.mkdir(parents=True, exist_ok=True)

    captured, prompts, last_metadata = load_capture(requests)
    all_stats = json.loads(Path(args.stats).read_text(encoding='utf-8'))
    namespace = all_stats[args.statistic_name]
    proprio_stats = namespace.get('proprio', namespace.get('state'))
    if proprio_stats is None:
        raise KeyError('Statistics contain neither proprio nor state')
    training_images, sources = decode_training_images(roots,
                                                      args.sample_frames)
    tasks = read_training_tasks(roots)
    report = {
        'capture': {
            'root': str(capture_root),
            'request_count': captured['request_count'],
            'last_request_status': last_metadata.get('status'),
            'has_model_inputs': bool(captured['model_input_manifests']),
            'has_actions': bool(captured['action_manifests']),
        },
        'statistics': {
            'path': str(Path(args.stats).expanduser().resolve()),
            'namespace': args.statistic_name,
        },
        'dataset_roots': [str(path) for path in roots],
        'prompt': {
            'captured':
            sorted(set(prompts)),
            'training_tasks':
            tasks,
            'all_match_training_tasks':
            bool(prompts) and all(prompt in tasks for prompt in prompts),
        },
        'state': state_report(captured['states'], proprio_stats),
        'images': image_report(captured['images'], training_images),
        'training_image_sources': sources,
        'model_input_manifests': captured['model_input_manifests'],
        'action_manifests': captured['action_manifests'],
    }
    (output / 'comparison.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (output / 'comparison.md').write_text(
        markdown_report(report), encoding='utf-8')
    make_montage(output / 'camera_montage.png', captured['images'],
                 training_images)
    print(f'Wrote comparison to {output}')


if __name__ == '__main__':
    main()
