"""One supervised action per sample: all preceding context has loss -100."""
from .environment import canonical
from .rollout import messages_for


def action_sample(env,action,identifier,provenance):
    if not isinstance(action,dict):raise ValueError('SFT target must be a parsed valid action')
    return dict(id=identifier,task_id=env.task['id'],messages=messages_for(env),
                target=canonical(action),provenance=provenance,
                loss_policy='target_action_only; system, user, tools and prior assistant messages masked')


def tokenize_sample(tokenizer,sample,max_length=32768):
    messages=sample['messages']
    if not messages or messages[-1]['role']!='user':raise ValueError('Expected an observation before the target action')
    prefix=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':sample['target']}],
                                      tokenize=True,add_generation_prompt=False,enable_thinking=False)
    if full[:len(prefix)]!=prefix:raise ValueError('Chat template is not prefix-stable; cannot assign target token boundaries')
    if len(full)>max_length:raise ValueError('SFT sample exceeds token budget; no truncation permitted')
    if len(full)<=len(prefix):raise ValueError('Empty target')
    labels=[-100]*len(prefix)+full[len(prefix):]
    return dict(input_ids=full,attention_mask=[1]*len(full),labels=labels,
                prefix_tokens=len(prefix),target_tokens=len(full)-len(prefix))


def pad_batch(rows,pad_token_id):
    """Use this collator directly; an LM collator would overwrite custom labels."""
    if not rows:raise ValueError('Empty batch')
    width=max(len(r['input_ids']) for r in rows)
    batch={k:[] for k in ('input_ids','attention_mask','labels')}
    for row in rows:
        n=width-len(row['input_ids'])
        if not (len(row['input_ids'])==len(row['attention_mask'])==len(row['labels'])):raise ValueError('Misaligned labels')
        for k,pad in [('input_ids',pad_token_id),('attention_mask',0),('labels',-100)]:
            batch[k].append(row[k]+[pad]*n)
    return batch
