"""One fresh rollout batch, two GRPO epochs; explicit action-token probabilities."""
import hashlib
import math
import statistics

VERSION = 'retail-grpo-update-v1'
TEMPERATURE = 0.7
GROUP_SIZE = 4
UPDATES = 2
LEARNING_RATE = 1e-6
BETA = 0.02
CLIP = 0.2
MAX_OUTPUT = 2048
MAX_PROMPT = 30720
MAX_CONTEXT = 32768


def seed_for(task_id, sample_index, step):
    if not 0 <= sample_index < GROUP_SIZE or not 0 <= step < 12:
        raise ValueError('Invalid scheduled request')
    return int.from_bytes(hashlib.sha256(
        f'{VERSION}|173|{task_id}|{sample_index}|{step}'.encode()).digest()[:4], 'big') % 2**31


def advantages(rewards):
    if len(rewards) != GROUP_SIZE or any(type(r) not in (int, float) or r not in (0, 1) for r in rewards):
        raise ValueError('A complete, binary-reward group is required')
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards)
    return [(r - mean) / (std + 1e-8) for r in rewards]


def action_row(prompt_ids, generated_ids):
    """Use exact sampled IDs, including sampled EOS; never re-tokenize outputs."""
    if not prompt_ids or not generated_ids:
        raise ValueError('Empty prompt or generated action')
    if len(prompt_ids) > MAX_PROMPT or len(generated_ids) > MAX_OUTPUT:
        raise ValueError('Budget exceeded')
    if any(type(x) is not int or x < 0 for x in prompt_ids + generated_ids):
        raise ValueError('Invalid token IDs')
    return dict(input_ids=prompt_ids + generated_ids,
                attention_mask=[1] * (len(prompt_ids) + len(generated_ids)),
                labels=[-100] * len(prompt_ids) + generated_ids)


def selected_logps(model, row, device, temperature=TEMPERATURE):
    import torch
    from .lora_backend import tensor_batch
    batch = tensor_batch(row, device)
    labels = batch['labels']
    if labels[0, 0].item() != -100:
        raise ValueError('First token cannot be predicted')
    positions = torch.where(labels[0, 1:] != -100)[0]
    if positions.numel() == 0 or not torch.all(batch['attention_mask'][0, positions + 1] == 1):
        raise ValueError('Missing or padded action targets')
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                   use_cache=False, logits_to_keep=positions).logits[0].float() / temperature
    targets = labels[0, positions + 1]
    return logits.gather(1, targets[:, None]).squeeze(1) - torch.logsumexp(logits, dim=-1)


def objective(logps, old_logps, ref_logps, advantage, beta=BETA, clip=CLIP):
    """Return token losses; caller normalizes by total generated tokens per episode."""
    import torch
    if logps.ndim != 1 or old_logps.shape != logps.shape or ref_logps.shape != logps.shape:
        raise ValueError('Token probability arrays differ')
    if not math.isfinite(advantage) or beta < 0 or not 0 < clip < 1:
        raise ValueError('Invalid objective parameters')
    old_logps, ref_logps = old_logps.detach(), ref_logps.detach()
    ratio = torch.exp(logps - old_logps)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    delta = ref_logps - logps
    kl = torch.expm1(delta) - delta
    losses = -surrogate + beta * kl
    if not torch.isfinite(losses).all():
        raise ValueError('Nonfinite GRPO loss')
    return losses, dict(kl=kl.detach(), ratio=ratio.detach(),
                        clipped=((ratio - 1).abs() > clip).detach())


def generation_config(eos_ids, pad_id):
    from transformers import GenerationConfig
    # Construct from scratch: no inherited top-k, suppression or forced tokens.
    return GenerationConfig(do_sample=True, temperature=TEMPERATURE, top_p=1.0,
        top_k=0, repetition_penalty=1.0, max_new_tokens=MAX_OUTPUT,
        eos_token_id=eos_ids, pad_token_id=pad_id, bos_token_id=None,
        return_dict_in_generate=True, output_scores=True, use_cache=True)


def sample_action(model, tokenizer, messages, seed, device):
    import torch
    prompt = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    if len(prompt) > MAX_PROMPT:
        return dict(kind='prompt_token_limit', prompt_ids=prompt)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    if not eos or any(not isinstance(x, int) for x in eos):
        raise ValueError('Missing EOS configuration')
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    ids = torch.tensor([prompt], device=device)
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                generation_config=generation_config(eos, pad), logits_to_keep=1)
        generated = output.sequences[0, len(prompt):].tolist()
        if len(output.scores) != len(generated):
            raise ValueError('Generation probability count mismatch')
        sampled_logps = [float(torch.log_softmax(s[0].float(), dim=-1)[t].cpu())
                         for s, t in zip(output.scores, generated)]
    if not generated or any(t in eos for t in generated[:-1]):
        raise ValueError('Invalid generated stopping sequence')
    stop = generated[-1] in eos
    if not stop and len(generated) != MAX_OUTPUT:
        raise ValueError('Unaccounted generation stop')
    return dict(kind='stop' if stop else 'output_token_limit', prompt_ids=prompt,
                generated_ids=generated, sampled_logps=sampled_logps,
                raw=tokenizer.decode(generated, skip_special_tokens=True), eos_ids=eos)
