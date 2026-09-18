"""Qwen3 next-action loss with a bounded vocabulary projection; microbatch one."""
import torch
import torch.nn.functional as F


def target_loss(model,batch):
    labels=batch['labels']
    if labels.ndim!=2 or labels.shape[0]!=1:
        raise ValueError('This backend requires microbatch size one')
    if labels[0,0].item()!=-100:
        raise ValueError('The first token cannot be a next-token target')
    positions=torch.where(labels[0,1:]!=-100)[0]
    if positions.numel()==0:raise ValueError('No supervised next-token targets')
    if not torch.all(batch['attention_mask'][0,positions+1]==1):
        raise ValueError('Padding must not receive supervision')
    # Qwen3 4.51.3 supports tensor sequence indices. Keep all attention context,
    # but project only states immediately preceding supervised target tokens.
    output=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                 use_cache=False,logits_to_keep=positions)
    logits=output.logits[0].float()
    targets=labels[0,positions+1]
    return F.cross_entropy(logits,targets),logits


def tensor_batch(row,device):
    return {k:torch.tensor([row[k]],dtype=torch.long,device=device)
            for k in ('input_ids','attention_mask','labels')}


def add_lora(base):
    from peft import LoraConfig,get_peft_model
    config=LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=0.0,
        bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model=get_peft_model(base,config)
    if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):
        raise ValueError('Unexpected trainable base weights')
    return model
