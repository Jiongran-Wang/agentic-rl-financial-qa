#!/usr/bin/env python3
"""Greedy development evaluation for a hash-selected iterative checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.retail.environment import digest
from training.retail.grpo_iter_v1 import collect_episode
from training.retail.policy_compare_v1 import sha, metrics, REVISION
from training.retail.rewards_v2 import NumericVerifierV2
from training.retail.grpo_update_v1 import MAX_PROMPT, MAX_OUTPUT


def run(a):
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    from peft import PeftModel
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if (torch.__version__.split('+')[0], transformers.__version__, peft.__version__) != ('2.6.0', '4.51.3', '0.15.2'):
        raise ValueError('Changed evaluation runtime')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA bf16 required')
    release = json.loads((a.data / 'release.json').read_text())
    for name, h in release['input_hashes'].items():
        if sha(a.data / name) != h:
            raise ValueError('Changed input: ' + name)
    manifest = json.loads(a.model_manifest.read_text())
    if manifest['repo_id'] != 'Qwen/Qwen3-4B' or manifest['revision'] != REVISION or Path(manifest['snapshot_path']).resolve() != a.model.resolve():
        raise ValueError('Wrong base model')
    for name, h in release['tokenizer_hashes'].items():
        if sha(a.model / name) != h:
            raise ValueError('Changed tokenizer')
    if a.round == 0:
        hashes = release['adapter_hashes']
    else:
        prior = json.loads((a.adapter.parent / 'summary.json').read_text())
        if prior['status'] != 'fresh_round_updates_and_reload_passed' or prior['round'] != a.round or prior['release_sha256'] != sha(a.data / 'release.json'):
            raise ValueError('Checkpoint failed its training audit')
        hashes = prior['checkpoint_files']
    for name, h in hashes.items():
        if sha(a.adapter / name) != h:
            raise ValueError('Changed evaluated checkpoint')
    directory = a.data / 'development'
    corpus = json.loads((directory / 'corpus.json').read_text())
    tasks = list(map(json.loads, (directory / 'public.tasks.jsonl').read_text().splitlines()))
    refs = {r['id']: r for r in map(json.loads, (directory / 'scorer_only/references.jsonl').read_text().splitlines())}
    if len(tasks) != 27 or len(refs) != 27 or set(refs) != {t['id'] for t in tasks}:
        raise ValueError('Development tasks differ')
    for task in tasks:
        ref = refs[task['id']]
        if set(task) != {'id', 'question', 'as_of'} or ref['task_sha256'] != digest(task) or ref['corpus_sha256'] != digest(corpus) or ref['source_check'] != 'passed':
            raise ValueError('Development reference provenance mismatch')
    verifier = NumericVerifierV2(corpus, json.loads((directory / 'scorer_only/equivalent_cells.json').read_text()))
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True).to('cuda')
    model = PeftModel.from_pretrained(base, a.adapter)
    model.eval()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if not eos:
        raise ValueError('Missing EOS')
    config = GenerationConfig(do_sample=False, max_new_tokens=MAX_OUTPUT, eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=None, repetition_penalty=1.,
        return_dict_in_generate=True, output_scores=False, use_cache=True)
    torch.manual_seed(73)
    torch.cuda.manual_seed_all(73)
    def sampler(messages, _seed):
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(prompt) > MAX_PROMPT:
            return dict(kind='prompt_token_limit', prompt_ids=prompt)
        ids = torch.tensor([prompt], device='cuda')
        with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    generation_config=config, logits_to_keep=1)
        generated = result.sequences[0, len(prompt):].tolist()
        if not generated or any(t in eos for t in generated[:-1]):
            raise ValueError('Invalid generated sequence')
        stop = generated[-1] in eos
        if not stop and len(generated) != MAX_OUTPUT:
            raise ValueError('Unaccounted stop')
        return dict(kind='stop' if stop else 'output_token_limit', prompt_ids=prompt,
                    generated_ids=generated, raw=tokenizer.decode(generated, skip_special_tokens=True),
                    usage=dict(prompt_tokens=len(prompt), completion_tokens=len(generated), total_tokens=len(prompt)+len(generated)))
    contract = dict(round=a.round, release_sha256=sha(a.data / 'release.json'),
        adapter=str(a.adapter), adapter_hashes=hashes, backend='transformers_4.51.3_sdpa',
        seed=73, temperature=0, max_prompt=MAX_PROMPT, max_output=MAX_OUTPUT,
        max_steps=12, observation_chars=18000, thinking=False, reserved_policy_calls=0)
    (a.output / 'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    rows = []
    started = time.time()
    for task in tasks:
        row = collect_episode(corpus, task, 0, sampler, verifier, refs[task['id']], 1)
        row['round'] = a.round
        for call in row['calls']:
            call['seed'] = 73
        row['status'] = 'budget_exhausted' if row['budget'] else 'complete'
        rows.append(row)
        with (a.output / 'rollouts.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps(dict(stage='development', round=a.round, completed=len(rows), score=row['grade']['score'])), flush=True)
    summary = metrics(rows, len(tasks))
    summary.pop('infrastructure_errors')
    summary.update(status='development_complete', round=a.round, tasks=27, unscorable=0,
        successes=sum(int(r['grade']['score']) for r in rows), seconds=time.time()-started,
        adapter=str(a.adapter), adapter_hashes=hashes, release_sha256=contract['release_sha256'],
        rollouts_sha256=sha(a.output/'rollouts.jsonl'), reserved_policy_calls=0)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'model-manifest', 'adapter', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/retail_grpo_iter_v1')
    p.add_argument('--round', type=int, required=True, choices=range(4))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        run(a)
    except Exception as exc:
        (a.output/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__, error=str(exc)))+'\n')
        raise
