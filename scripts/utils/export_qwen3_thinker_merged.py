#!/usr/bin/env python3
"""Export thinker weights with this project's per-expert LoRA, one shard at a time."""
import argparse
import json
import shutil
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'export_complete.json').exists():
        report = json.loads((args.output / 'export_complete.json').read_text())
        assert report['checkpoint'] == str(args.checkpoint.resolve())
        # Repair metadata from exports created before the padded-vocab fix.
        index = json.loads((args.output / 'model.safetensors.index.json').read_text())['weight_map']
        key = 'thinker.model.embed_tokens.weight'
        with safe_open(args.output / index[key], framework='pt') as handle:
            vocab_size = handle.get_slice(key).get_shape()[0]
        config_path = args.output / 'config.json'
        config = json.loads(config_path.read_text())
        if config['thinker_config']['text_config']['vocab_size'] != vocab_size:
            config['thinker_config']['text_config']['vocab_size'] = vocab_size
            config_path.write_text(json.dumps(config, indent=2)+'\n')
        print('Already exported', flush=True)
        return
    cfg = json.loads((args.checkpoint / 'adapter_config.json').read_text())
    assert not cfg.get('use_dora') and not cfg.get('use_rslora') and not cfg.get('fan_in_fan_out')
    assert not cfg.get('rank_pattern') and not cfg.get('alpha_pattern')
    scale = cfg['lora_alpha'] / cfg['r']
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model))
    added = tok.add_special_tokens({'additional_special_tokens': ['[SEG]']})
    seg_id = tok.convert_tokens_to_ids('[SEG]')
    tok.save_pretrained(str(args.output))
    for path in args.model.iterdir():
        if path.is_file() and path.suffix in {'.json', '.txt', '.jinja'} and not path.name.startswith(('tokenizer', 'model.safetensors')):
            shutil.copy2(path, args.output / path.name)
    outmap, applied, replaced, total_bytes = {}, [], [], 0
    start = time.perf_counter()
    with safe_open(args.checkpoint / 'adapter_model.safetensors', framework='pt', device='cpu') as adapter:
        akeys = set(adapter.keys())
        expected = {k for k in akeys if k.endswith('.lora_A.weight')}
        for shard in sorted(set(index.values())):
            keys = [k for k, filename in index.items() if filename == shard and k.startswith('thinker.')]
            if not keys:
                continue
            tensors = {}
            with safe_open(args.model / shard, framework='pt', device='cpu') as base:
                for key in keys:
                    adapter_key = 'base_model.model.' + key
                    lora_a = adapter_key.removesuffix('.weight') + '.lora_A.weight'
                    if adapter_key in akeys:
                        value = adapter.get_tensor(adapter_key).clone()
                        replaced.append(adapter_key)
                    elif lora_a in akeys:
                        lora_b = lora_a.replace('.lora_A.weight', '.lora_B.weight')
                        value = base.get_tensor(key)
                        a, b = adapter.get_tensor(lora_a), adapter.get_tensor(lora_b)
                        # PEFT merges in the adapter/base dtype. Preserve that rounding.
                        delta = (b @ a) * scale
                        assert value.shape == delta.shape
                        value = (value + delta.to(value.dtype)).contiguous()
                        applied.append(lora_a)
                    else:
                        value = base.get_tensor(key).clone()
                    tensors[key] = value
                    outmap[key] = shard
                    total_bytes += value.numel() * value.element_size()
            save_file(tensors, str(args.output / shard), metadata={'format': 'pt'})
            del tensors
            print('exported', shard, 'LoRA tensors', len(applied), 'elapsed', round(time.perf_counter()-start,1), flush=True)
        assert set(applied) == expected, f'Unmerged adapter keys: {expected-set(applied)}'
        assert 'base_model.model.thinker.model.embed_tokens.weight' in replaced
        assert 'base_model.model.thinker.lm_head.weight' in replaced
        assert seg_id < adapter.get_slice('base_model.model.thinker.model.embed_tokens.weight').get_shape()[0]
        vocab_size = adapter.get_slice('base_model.model.thinker.model.embed_tokens.weight').get_shape()[0]
    (args.output / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': total_bytes}, 'weight_map': outmap}, indent=2)+'\n')
    config = json.loads((args.output / 'config.json').read_text())
    # Swift pads resized embeddings (e.g. to 152064 rows); vLLM validates the
    # exact weight shape against this field, rather than the tokenizer length.
    config['thinker_config']['text_config']['vocab_size'] = vocab_size
    (args.output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    report = {'checkpoint': str(args.checkpoint.resolve()), 'model': str(args.model.resolve()),
              'merged_lora_modules': len(applied), 'replaced_modules': replaced, 'seg_token_id': seg_id,
              'thinker_only': True, 'total_bytes': total_bytes, 'export_sec': time.perf_counter()-start}
    (args.output / 'export_complete.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
