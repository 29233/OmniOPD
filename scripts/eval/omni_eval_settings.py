"""Check evaluation preprocessing against the selected dual OPSD training run."""
import argparse
import json
import math
import wave
from pathlib import Path


def inspect_refavs_fps(data_dir, splits=('test_s', 'test_u')):
    """Audit stored frame lists against REF-AVS clip boundaries and WAV headers.

    REF-AVS clip IDs end in start/end milliseconds. Density is a nominal
    sampling rate, not recovery of missing original per-frame timestamps.
    Refuse partial lists, inconsistent densities or unsynchronized audio.
    """
    clips, row_count = {}, 0
    for split in splits:
        with (Path(data_dir) / f'refavs_{split}.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                frames = row['videos'][0]
                if not isinstance(frames, list) or not frames:
                    raise ValueError('FPS audit requires nonempty frame-path lists')
                directory = Path(frames[0]).parent.parent
                clips.setdefault(str(directory), (directory, frames, Path(row['audios'][0])))
                row_count += 1
    rates, audio_lengths = [], []
    for directory, frames, audio_path in clips.values():
        parts = directory.name.rsplit('_', 2)
        seconds = (int(parts[-1]) - int(parts[-2])) / 1000
        available = set((directory / 'frames').glob('*.jpg'))
        if seconds <= 0 or set(map(Path, frames)) != available or len(frames) != len(available):
            raise ValueError(f'Cannot infer nominal dataset FPS from a partial/duplicated frame list: {directory}')
        with wave.open(str(audio_path)) as audio:
            audio_seconds = audio.getnframes() / audio.getframerate()
        if abs(audio_seconds - seconds) > .25:
            raise ValueError(f'Clip boundary and audio duration disagree: {directory}')
        rates.append(len(frames) / seconds)
        audio_lengths.append(audio_seconds)
    if not rates or any(not math.isclose(value, rates[0]) for value in rates):
        raise ValueError('Dataset has missing or inconsistent nominal frame sampling rates')
    return {'nominal_fps': rates[0], 'unique_clips_checked': len(clips), 'rows_checked': row_count,
            'audio_duration_range_sec': [min(audio_lengths), max(audio_lengths)],
            'evidence': 'Distinct stored frames / clip duration from REF-AVS start/end millisecond IDs; WAV duration checked.',
            'limitation': 'Original per-frame timestamps are not stored; this verifies nominal sampling density.'}


def validate_training_settings(checkpoint, environment, *, torch_dtype, max_pixels,
                               max_length, max_new_tokens, attn_impl, fps_policy='training'):
    if fps_policy not in ('training', 'dataset'):
        raise ValueError('fps_policy must be training or dataset')
    checkpoint = Path(checkpoint).resolve()
    arguments_path = checkpoint / 'args.json'
    launch_path = checkpoint.parent.parent / 'dual_launch.json'
    if not arguments_path.is_file() or not launch_path.is_file():
        raise ValueError(f'Training settings required: {arguments_path} and {launch_path}')
    arguments = json.loads(arguments_path.read_text(encoding='utf-8'))
    launch = json.loads(launch_path.read_text(encoding='utf-8'))
    comparisons = {}
    training_fps_differences = {}
    for key in ('FPS', 'AURORA_VIDEO_FPS', 'AURORA_NUM_SAM_FRAMES'):
        if key not in launch or key not in environment:
            raise ValueError(f'Missing explicit training/evaluation setting: {key}')
        pair = (float(launch[key]), float(environment[key]))
        if fps_policy == 'dataset' and key in ('FPS', 'AURORA_VIDEO_FPS'):
            if pair[0] != pair[1]:
                training_fps_differences[key] = {'training': pair[0], 'evaluation': pair[1]}
        else:
            comparisons[key] = pair
    if fps_policy == 'dataset' and 'DATASET_VIDEO_FPS' not in environment:
        raise ValueError('Dataset FPS policy requires audited DATASET_VIDEO_FPS')
    if 'DATASET_VIDEO_FPS' in environment:
        for key in ('FPS', 'AURORA_VIDEO_FPS'):
            comparisons[f'dataset_fps_vs_{key}'] = (float(environment['DATASET_VIDEO_FPS']),
                                                   float(environment[key]))
    # Newer launch manifests also record limits; old manifests must not be
    # silently represented as having recorded these values.
    for key in ('FPS_MAX_FRAMES', 'VIDEO_MAX_TOKEN_NUM', 'IMAGE_MAX_TOKEN_NUM'):
        if key in launch:
            comparisons[key] = (int(launch[key]), int(environment[key]))
    for key, actual in [('torch_dtype', torch_dtype), ('max_pixels', max_pixels),
                        ('max_length', max_length), ('max_completion_length', max_new_tokens),
                        ('attn_impl', attn_impl)]:
        if key not in arguments:
            raise ValueError(f'Training args do not record {key}')
        comparisons[key] = (arguments[key], actual)
    mismatch = {key: {'training': before, 'evaluation': after}
                for key, (before, after) in comparisons.items() if before != after}
    if mismatch:
        raise ValueError('Training/evaluation setting mismatch: ' + json.dumps(mismatch))
    return {'training_args': str(arguments_path), 'training_launch': str(launch_path),
            'fps_policy': fps_policy, 'training_fps_differences': training_fps_differences,
            'matched_settings': {key: before for key, (before, _) in comparisons.items()},
            'limits_without_historical_manifest_evidence': [key for key in
                ('FPS_MAX_FRAMES', 'VIDEO_MAX_TOKEN_NUM', 'IMAGE_MAX_TOKEN_NUM') if key not in launch],
            'decoding': 'Greedy evaluation; training rollouts used sampling.',
            'audio': 'Original test audio; audio degradation is a separate robustness evaluation.'}


def main() -> None:
    """CLI entrypoint to inspect REF-AVS split FPS and emit an audit JSON."""
    parser = argparse.ArgumentParser(
        description='Inspect REF-AVS frame lists and WAV durations to emit dataset timing audit JSON.'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing refavs_<split>.jsonl manifests.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['test_s', 'test_u'],
        help='Dataset splits to inspect (default: test_s test_u).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional output path to write formatted audit JSON (creates parent dirs).',
    )
    args = parser.parse_args()
    report = inspect_refavs_fps(args.data_dir, splits=tuple(args.splits))
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload, end='', flush=True)


if __name__ == '__main__':
    main()
