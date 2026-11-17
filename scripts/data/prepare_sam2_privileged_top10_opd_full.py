#!/usr/bin/env python3
"""Build full SAM2 sparse-OPD rows after clip-level audio synthesis completes."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding='utf-8') if line.strip()]


def clip_id(row):
    return Path(row['audios'][0]).parent.name


def privilege(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Missing {field} for {row.get("sample_id")}')
    return value.strip()


def make_rows(baseline_path, visual_path, captions_path, output_path, split):
    baseline, visual, captions = map(read, (baseline_path, visual_path, captions_path))
    if len(baseline) != len(visual):
        raise ValueError(f'{split}: baseline/visual row counts differ')
    visual_by_id = {row.get('sample_id'): row for row in visual}
    if len(visual_by_id) != len(visual):
        raise ValueError(f'{split}: duplicate visual sample_id')
    caption_rows = [row for row in captions if row.get('status', 'ok') == 'ok']
    caption_by_clip = {str(row.get('clip_id')): privilege(row, 'caption') for row in caption_rows}
    if len(caption_by_clip) != len(caption_rows):
        raise ValueError(f'{split}: duplicate or missing caption clip_id')
    output = []
    for index, base in enumerate(baseline):
        sample_id = base.get('sample_id')
        visual_row = visual_by_id.get(sample_id)
        if visual_row is None:
            raise ValueError(f'{split}: visual row missing {sample_id}')
        if clip_id(base) not in caption_by_clip:
            raise ValueError(f'{split}: audio caption missing clip {clip_id(base)}')
        item = copy.deepcopy(base)
        item['case_index'] = index
        item['audio_privilege_text'] = caption_by_clip[clip_id(base)]
        item['visual_privilege_text'] = privilege(visual_row, 'privilege_text')
        item['privilege_type'] = 'audio_visual_text_dual_teacher'
        output.append(item)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in output), encoding='utf-8')
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-baseline', type=Path, required=True)
    p.add_argument('--train-visual', type=Path, required=True)
    p.add_argument('--train-captions', type=Path, required=True)
    p.add_argument('--train-output', type=Path, required=True)
    p.add_argument('--val-baseline', type=Path, required=True)
    p.add_argument('--val-visual', type=Path, required=True)
    p.add_argument('--val-captions', type=Path, required=True)
    p.add_argument('--val-output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    train = make_rows(args.train_baseline, args.train_visual, args.train_captions, args.train_output, 'train')
    val = make_rows(args.val_baseline, args.val_visual, args.val_captions, args.val_output, 'val')
    payload = {
        'schema': 1,
        'purpose': 'full SAM2 sparse top10 dual-teacher pure OPD',
        'train': {'rows': len(train), 'unique_clips': len({clip_id(row) for row in train}),
                  'output': str(args.train_output.resolve())},
        'val': {'rows': len(val), 'unique_clips': len({clip_id(row) for row in val}),
                'output': str(args.val_output.resolve())},
        'injection': {'audio': 'Privileged clip-level audio evidence: <caption>',
                      'visual': 'Hint: Visual evidence about the target: <caption>'},
        'compute': {'initializer_dtype': 'float16', 'opd_compute_dtype': 'bfloat16',
                    'dataset_fps': 1.0, 'sam_frames': 10, 'max_length': 5120,
                    'max_pixels': 200704},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
