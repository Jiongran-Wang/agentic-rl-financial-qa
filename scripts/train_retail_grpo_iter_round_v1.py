#!/usr/bin/env python3
"""One fresh round of the frozen three-round iterative GRPO experiment."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.retail.environment import digest
from training.retail.rewards_v2 import NumericVerifierV2
from training.retail.grpo_update_v1 import (TEMPERATURE, GROUP_SIZE, UPDATES,
    LEARNING_RATE, BETA, CLIP, advantages, action_row, selected_logps)
from training.retail.grpo_update_v2 import objective, correction, check_replay
from training.retail.grpo_iter_v1 import VERSION, collect_episode, verify_parent, restore_adapter
from training.retail.grpo_update_v1 import sample_action
from training.retail.policy_compare_v1 import sha, REVISION


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def run(a):
    import torch
    import transformers
    import peft
    import accelerate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from safetensors.torch import load_file

    started = time.time()
    versions = {n: m.__version__ for n, m in [('torch', torch), ('transformers', transformers),
                                              ('peft', peft), ('accelerate', accelerate)]}
    for n, v in dict(torch='2.6.0', transformers='4.51.3', peft='0.15.2', accelerate='1.6.0').items():
        if versions[n].split('+')[0] != v:
            raise ValueError('Pinned package version mismatch: ' + n)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA bf16 required')
    release = json.loads((a.data / 'release.json').read_text())
    for name, h in release['input_hashes'].items():
        if sha(a.data / name) != h:
            raise ValueError('Changed input: ' + name)
    manifest = json.loads(a.model_manifest.read_text())
    if manifest['repo_id'] != 'Qwen/Qwen3-4B' or manifest['revision'] != REVISION or a.model.resolve() != Path(manifest['snapshot_path']).resolve():
        raise ValueError('Wrong base model')
    for name, h in release['tokenizer_hashes'].items():
        if sha(a.model / name) != h:
            raise ValueError('Changed tokenizer')
    release['_sha256'] = sha(a.data / 'release.json')
    verify_parent(a.adapter, a.anchor, a.round, release, a.parent_summary)
    corpus = json.loads((a.data / 'corpus.json').read_text())
    tasks = list(map(json.loads, (a.data / 'public.tasks.jsonl').read_text().splitlines()))
    refs = {r['id']: r for r in map(json.loads, (a.data / 'trainer_only/references.jsonl').read_text().splitlines())}
    if len(tasks) != 26 or len({t['id'] for t in tasks}) != 26 or set(refs) != {t['id'] for t in tasks} or any(c['split'] != 'train' for c in corpus):
        raise ValueError('Exactly the frozen training tasks/corpus required')
    for task in tasks:
        if set(task) != {'id', 'question', 'as_of'}:
            raise ValueError('Privileged fields in public task')
        ref = refs[task['id']]
        if ref['task_sha256'] != digest(task) or ref['corpus_sha256'] != digest(corpus) or ref.get('source_check') != 'passed':
            raise ValueError('Training reference provenance mismatch')
    verifier = NumericVerifierV2(corpus, json.loads((a.data / 'trainer_only/equivalent_cells.json').read_text()))
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    def attention():
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def base():
        return AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True).to('cuda')
    def event(**x):
        print(json.dumps(x), flush=True)
        append(a.output / 'progress.jsonl', x)
    write(a.output / 'config.json', dict(version=VERSION, versions=versions,
        release_sha256=sha(a.data / 'release.json'), base_revision=REVISION,
        initial_adapter_sha256=sha(a.adapter / 'adapter_model.safetensors'),
        planned_episodes=104, updates=UPDATES, temperature=TEMPERATURE, top_p=1.0, top_k=0,
        optimizer='AdamW', learning_rate=LEARNING_RATE, weight_decay=0.0, beta=BETA, clip=CLIP,
        reduction='Mean across 104 episodes of mean across all generated action tokens in each episode',
        reference='Fixed SFT epoch-2 policy in all rounds, temperature 0.7',
        round=a.round, parent_adapter=str(a.adapter), fixed_anchor=str(a.anchor),
        fixed_anchor_sha256=sha(a.anchor / 'adapter_model.safetensors'),
        optimizer_reset_each_round=True,
        probability_check='Round-start teacher replay versus first backward: max abs <= 1e-4, mean abs <= 1e-5',
        rollout_archive_reused=False,
        probability_contract='Behavior is sampled round-start policy; old is its teacher-forced replay',
        correction='detach(min(exp(old_teacher - sampled_behavior), 2.0)); multiplies surrogate and K3',
        ppo_ratio='exp(current_teacher - old_teacher)',
        min_mixed_groups=4, max_steps=12, max_prompt=30720, max_output=2048,
        reserved_policy_calls=0, trainer_labels_never_in_policy_messages=True))
    event(stage='loading_epoch_2')
    model = PeftModel.from_pretrained(base(), a.adapter, is_trainable=True)
    model.eval()
    if any(p.requires_grad and 'lora_' not in n for n, p in model.named_parameters()):
        raise ValueError('Unexpected trainable base parameters')
    if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in model.modules()):
        raise ValueError('Dropout would change rollout/training probability contract')
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    schedule = [dict(id=t['id'], sample_index=i) for i in range(GROUP_SIZE) for t in tasks]
    write(a.output / 'schedule.json', schedule)
    by_id = {t['id']: t for t in tasks}
    episodes = []
    def sampler(messages, seed):
        with attention():
            return sample_action(model, tokenizer, messages, seed, 'cuda')
    for job in schedule:
        t0 = time.time()
        row = collect_episode(corpus, by_id[job['id']], job['sample_index'], sampler,
                              verifier, refs[job['id']], a.round)
        row['seconds'] = time.time() - t0
        episodes.append(row)
        append(a.output / 'rollouts.jsonl', row)
        event(stage='fresh_rollout', round=a.round, completed=len(episodes),
              id=row['id'], sample_index=row['sample_index'], score=row['grade']['score'],
              budget=row['budget'], seconds=row['seconds'])
    groups = []
    for task in tasks:
        group = [e for e in episodes if e['id'] == task['id']]
        rewards = [e['grade']['score'] for e in group]
        adv = advantages(rewards)
        for e, value in zip(group, adv):
            e['advantage'] = value
        groups.append(dict(id=task['id'], rewards=rewards, advantages=adv, mixed=len(set(rewards)) > 1))
    write(a.output / 'groups.json', groups)
    if sum(g['mixed'] for g in groups) < 4:
        write(a.output / 'summary.json', dict(status='insufficient_fresh_reward_variance_no_updates',
              episodes=104, mixed_groups=sum(g['mixed'] for g in groups), optimizer_updates=0))
        raise RuntimeError('Fresh trainer batch fails variance gate')

    # Cache old/reference logps before any optimizer exists. Include failed/truncated actions.
    actions = []
    max_delta = 0.0
    delta_sum = 0.0
    total_tokens = 0
    correction_min, correction_max, correction_sum, capped_tokens = float("inf"), 0.0, 0.0, 0
    original_guard_failed_actions = 0
    for episode_index, e in enumerate(episodes, 1):
        token_count = sum(len(c['response'].get('generated_ids', [])) for c in e['calls'])
        if not token_count:
            raise ValueError('Episode has no generated tokens')
        for turn, call in enumerate(e['calls']):
            r = call['response']
            if 'generated_ids' not in r:
                continue
            if tokenizer.apply_chat_template(call['messages'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False) != r['prompt_ids']:
                raise ValueError('Source prompt tokenization changed')
            if tokenizer.decode(r['generated_ids'], skip_special_tokens=True) != r['raw']:
                raise ValueError('Source output decode changed')
            row = action_row(r['prompt_ids'], r['generated_ids'])
            with torch.no_grad(), attention():
                lp = selected_logps(model, row, 'cuda').cpu()
            sampled = torch.tensor(r['sampled_logps'])
            weights, raw_weights = correction(lp, sampled)
            correction_min = min(correction_min, raw_weights.min().item())
            correction_max = max(correction_max, raw_weights.max().item())
            correction_sum += raw_weights.sum().item()
            capped_tokens += int((raw_weights > 2).sum().item())
            delta = (lp - sampled).abs()
            if not torch.isfinite(delta).all():
                raise ValueError('Nonfinite sampling probabilities')
            original_guard_failed_actions += int(delta.max().item() > .2 or delta.mean().item() > .02)
            max_delta = max(max_delta, delta.max().item())
            delta_sum += delta.sum().item()
            total_tokens += len(lp)
            item = dict(id=e['id'], sample_index=e['sample_index'], turn=turn,
                        row=row, old_logps=lp.tolist(),
                        behavior_logps=sampled.tolist(), correction=weights.tolist(),
                        advantage=e['advantage'], episode_tokens=token_count)
            actions.append(item)
        event(stage='old_reference_cached', episodes=episode_index,
              id=e['id'], sample_index=e['sample_index'], actions=len(actions))
    write(a.output / 'probability_check.json', dict(max_abs=max_delta, mean_abs=delta_sum / total_tokens,
          generated_tokens=total_tokens, actions=len(actions), teacher_replay_check='Enforced on first backward pass',
          original_behavior_agreement_gate_passed=(original_guard_failed_actions == 0),
          original_guard_failed_actions=original_guard_failed_actions,
          correction_min=correction_min, correction_max=correction_max,
          correction_mean=correction_sum/total_tokens, capped_tokens=capped_tokens,
          old_and_reference_equal_at_initial_policy=(a.round == 1)))

    # Evaluate the fixed anchor using the same base and adapter slot. No optimizer exists yet.
    anchor_state = load_file(str(a.anchor / 'adapter_model.safetensors'), device='cpu')
    restore_adapter(model, anchor_state)
    model.eval()
    for index, item in enumerate(actions):
        with torch.no_grad(), attention():
            item['ref_logps'] = selected_logps(model, item['row'], 'cuda').cpu().tolist()
        if a.round == 1:
            check_replay(item['ref_logps'], item['old_logps'])
        append(a.output / 'training_actions.jsonl', item)
        if (index + 1) % 20 == 0:
            event(stage='fixed_reference_cached', actions=index + 1, total_actions=len(actions))
    restore_adapter(model, initial)
    del anchor_state


    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    reports = []
    probe = min(actions, key=lambda x: len(x['row']['input_ids']))['row']
    for update in range(1, UPDATES + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        totals = dict(loss=0.0, kl=0.0, clip_fraction=0.0)
        t0 = time.time()
        for index, item in enumerate(actions):
            old = torch.tensor(item['old_logps'], device='cuda')
            ref = torch.tensor(item['ref_logps'], device='cuda')
            with attention():
                lp = selected_logps(model, item['row'], 'cuda')
                if update == 1:
                    check_replay(lp.detach().cpu().tolist(), item['old_logps'])
                losses, stats = objective(lp, old, ref,
                    torch.tensor(item['behavior_logps'], device='cuda'), item['advantage'])
                scale = 1 / (len(episodes) * item['episode_tokens'])
                loss = losses.sum() * scale
                loss.backward()
            totals['loss'] += loss.item()
            totals['kl'] += stats['kl'].sum().item() * scale
            totals['clip_fraction'] += stats['clipped'].sum().item() * scale
            del lp, loss, losses, stats, old, ref
            if (index + 1) % 20 == 0:
                event(stage='backward', update=update, actions=index + 1, total_actions=len(actions))
        grads = [p.grad for p in trainable if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise RuntimeError('Missing/nonfinite gradients')
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(norm) or norm.item() == 0:
            raise RuntimeError('Zero/nonfinite gradient norm')
        if totals['kl'] > 0.1:
            raise RuntimeError('Pre-update sampled KL exceeds smoke-test limit')
        optimizer.step()
        if any(not p.requires_grad and (p.grad is not None or p._version != frozen[n]) for n, p in model.named_parameters()):
            raise RuntimeError('Base weights changed')
        if not all(torch.isfinite(p).all().item() for p in trainable):
            raise RuntimeError('Nonfinite updated adapter')
        report = dict(update=update, gradient_norm=norm.item(), seconds=time.time() - t0,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), **totals)
        reports.append(report)
        event(stage='optimizer_update_passed', **report)
        del grads
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    # Audit drift over the full batch after the second update, not only a short probe.
    final_kl = 0.0
    for item in actions:
        with torch.no_grad(), attention():
            lp = selected_logps(model, item['row'], 'cuda')
            _, stats = objective(lp, torch.tensor(item['old_logps'], device='cuda'),
                                 torch.tensor(item['ref_logps'], device='cuda'),
                                 torch.tensor(item['behavior_logps'], device='cuda'), item['advantage'])
        final_kl += stats['kl'].sum().item() / (104 * item['episode_tokens'])
        del lp, stats
    if final_kl > 0.1:
        raise RuntimeError('Final sampled KL exceeds smoke-test limit')
    with torch.no_grad(), attention():
        expected = selected_logps(model, probe, 'cuda').cpu()
    state = get_peft_model_state_dict(model)
    changed = sum(not torch.equal(v.cpu(), initial[k]) for k, v in state.items())
    if not changed:
        raise RuntimeError('No adapter tensors changed')
    directory = a.output / 'adapter_final'
    model.save_pretrained(directory, safe_serialization=True)
    del state, model, optimizer, trainable
    gc.collect()
    torch.cuda.empty_cache()
    event(stage='fresh_base_reload')
    loaded = PeftModel.from_pretrained(base(), directory)
    loaded.eval()
    with torch.no_grad(), attention():
        actual = selected_logps(loaded, probe, 'cuda').cpu()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    saved = load_file(str(directory / 'adapter_model.safetensors'), device='cpu')
    loaded_state = get_peft_model_state_dict(loaded)
    if set(saved) != set(loaded_state) or any(not torch.equal(saved[k], v.cpu()) for k, v in loaded_state.items()):
        raise RuntimeError('Reloaded adapter tensors differ')
    write(a.output / 'summary.json', dict(status='fresh_round_updates_and_reload_passed',
        round=a.round, release_sha256=release['_sha256'],
        parent_adapter_sha256=sha(a.adapter / 'adapter_model.safetensors'),
        fixed_anchor_sha256=sha(a.anchor / 'adapter_model.safetensors'),
        reused_episodes=0, new_policy_calls=sum(len(e['calls']) for e in episodes),
        optimizer_updates=2, episodes=104, successes=sum(e['grade']['score'] for e in episodes),
        mixed_groups=sum(g['mixed'] for g in groups), actions=len(actions), generated_tokens=total_tokens,
        updates=reports, final_corrected_sampled_k3=final_kl, changed_adapter_tensors=changed,
        frozen_base_unchanged=True, reload_max_abs_logp_diff=(actual - expected).abs().max().item(),
        checkpoint_files={p.name: sha(p) for p in directory.iterdir() if p.is_file()},
        wall_seconds=time.time() - started, reserved_policy_calls=0,
        policy_quality='Not evaluated; requires a separate controlled evaluation'))
    event(stage='complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, default=ROOT / 'results/retail_sft_pilot_v1/job_3159735/epoch_2')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/retail_grpo_iter_v1')
    parser.add_argument('--anchor', type=Path, default=ROOT / 'results/retail_sft_pilot_v1/job_3159735/epoch_2')
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--parent-summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        write(args.output / 'failure.json', dict(status='failed', error_type=type(exc).__name__, error=str(exc)))
        raise
