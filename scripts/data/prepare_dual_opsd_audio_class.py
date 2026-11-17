#!/usr/bin/env python3
"""Prepare isolated degraded-Student / class / clean-audio OPSD metadata."""
import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.data.prepare_refavs_opsd import class_from_sample_id, has_valid_class_privilege


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--degraded-name', default='audio_opsd_random_mute50.wav')
    p.add_argument('--max-samples', type=int, default=-1)
    p.add_argument('--include-null', action='store_true', help='Smoke set includes one null row')
    p.add_argument('--selection-only', action='store_true', help='Write raw rows before audio preparation')
    args = p.parse_args()
    if Path(args.degraded_name).name != args.degraded_name:
        p.error('degraded-name must be a filename')
    rows = [json.loads(line) for line in args.source.open() if line.strip()]
    if args.max_samples > 0:
        selected = rows[:args.max_samples]
        if args.include_null and not any(class_from_sample_id(r['sample_id']) == 'null' for r in selected):
            selected[-1] = next(r for r in rows if class_from_sample_id(r['sample_id']) == 'null')
        rows = selected
    counts = {'rows': len(rows), 'class_valid': 0, 'class_invalid': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as output:
        for row in rows:
            if not args.selection_only:
                cls = row.get('class_name') or class_from_sample_id(row['sample_id'])
                original = [str(Path(x).resolve()) for x in row['audios']]
                degraded = [str(Path(x).with_name(args.degraded_name)) for x in original]
                if len(original) != 1 or original == degraded:
                    raise ValueError('Expected one original audio and a distinct degraded path')
                for path in original + degraded:
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
                row.update(class_name=cls, audios=degraded, dual_clean_audio_paths=original,
                           dual_class_valid=has_valid_class_privilege(cls))
                for key in ('teacher_prompt', 'teacher_messages', 'teacher_audio_paths'):
                    row.pop(key, None)
                counts['class_valid' if row['dual_class_valid'] else 'class_invalid'] += 1
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(args.output.resolve()), **counts}))


if __name__ == '__main__':
    main()
