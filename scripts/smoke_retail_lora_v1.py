#!/usr/bin/env python3
"""Four diagnostic LoRA updates; no evaluation and no full SFT run."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.retail.sft_data import tokenize_sample
from training.retail.lora_backend import target_loss,tensor_batch,add_lora

REVISION='1cfa9a7208912126459214e8b04321603b3df60c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def run(args):
    import torch,transformers,peft,accelerate
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel,get_peft_model_state_dict
    versions={k:m.__version__ for k,m in [('torch',torch),('transformers',transformers),('peft',peft),('accelerate',accelerate)]}
    for k,v in dict(torch='2.6.0',transformers='4.51.3',peft='0.15.2',accelerate='1.6.0').items():
        if versions[k].split('+')[0]!=v:raise ValueError(f'{k} must be {v}, found {versions[k]}')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('A CUDA GPU with bf16 support is required')
    torch.manual_seed(73);torch.cuda.manual_seed_all(73)
    manifest=json.loads(args.model_manifest.read_text())
    if manifest['repo_id']!='Qwen/Qwen3-4B' or manifest['revision']!=REVISION:
        raise ValueError('Wrong base model revision')
    if args.model.resolve()!=Path(manifest['snapshot_path']).resolve():raise ValueError('Model path differs from pinned manifest')
    release=json.loads(args.release.read_text())
    if sha(args.samples)!=release['training_export_hashes']['action_samples.train.jsonl']:
        raise ValueError('Training dataset hash mismatch')
    tokenizer=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
    for name,h in release['tokenizer']['files'].items():
        if sha(args.model/name)!=h:raise ValueError('Tokenizer changed: '+name)
    samples=[json.loads(l) for l in args.samples.read_text().splitlines() if l.strip()]
    rows=[dict(id=s['id'],**tokenize_sample(tokenizer,s)) for s in samples]
    if len(rows)!=204 or max(len(r['input_ids']) for r in rows)!=31845:raise ValueError('Unexpected data/token budget')
    # Deterministic selection before any loss is observed, from training only.
    shortest=lambda xs:min(xs,key=lambda r:(len(r['input_ids']),r['id']))
    selected=[shortest(rows),shortest([r for r in rows if '/repair/' in r['id']]),
        shortest([r for r,s in zip(rows,samples) if json.loads(s['target'])['type']=='final']),
        max(rows,key=lambda r:(len(r['input_ids']),r['id']))]
    write(args.output/'config.json',dict(status='smoke_not_training_result',seed=73,versions=versions,
        model_revision=REVISION,dataset_sha256=sha(args.samples),steps=4,learning_rate=1e-4,
        microbatch=1,gradient_checkpointing=True,precision='bf16',attention='sdpa',
        selected=[dict(id=r['id'],tokens=len(r['input_ids']),targets=r['target_tokens']) for r in selected],
        purpose='Validate updates and longest retained sequence; never promote this debug adapter.',fresh_eval_used=False))
    def event(x):
        print(json.dumps(x),flush=True)
        with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    def load_base():
        return AutoModelForCausalLM.from_pretrained(str(args.model),local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda')
    event(dict(stage='loading_base'));model=add_lora(load_base())
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train()
    base_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=1e-4)
    initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    from torch.nn.attention import sdpa_kernel,SDPBackend
    def attention():return sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION])
    for step,row in enumerate(selected,1):
        start=time.time();torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
        event(dict(stage='step_started',step=step,id=row['id'],tokens=len(row['input_ids'])))
        batch=tensor_batch(row,'cuda')
        with attention():loss,logits=target_loss(model,batch);del logits;loss.backward()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        grads=[p.grad for p in trainable if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):raise RuntimeError('Missing/nonfinite gradients')
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
        if not torch.isfinite(norm) or norm.item()==0:raise RuntimeError('Zero/nonfinite gradient norm')
        optimizer.step()
        if any(p.requires_grad is False and (p.grad is not None or p._version!=base_versions[n]) for n,p in model.named_parameters()):
            raise RuntimeError('Base weights changed or received gradients')
        event(dict(stage='step_passed',step=step,loss=loss.item(),gradient_norm=norm.item(),
            seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del batch,loss,grads;gc.collect()
    model.eval();probe=tensor_batch(selected[0],'cuda')
    with torch.no_grad(),attention():_,expected=target_loss(model,probe)
    expected=expected.cpu();state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    changed=sum(not torch.equal(v,initial[k]) for k,v in state.items())
    if changed==0:raise RuntimeError('Optimizer did not change adapters')
    checkpoint=args.output/'debug_adapter';model.save_pretrained(checkpoint,safe_serialization=True)
    tokenizer.save_pretrained(checkpoint);write(checkpoint/'SMOKE_ONLY.json',dict(promote=False,steps=4))
    del optimizer,trainable,model;gc.collect();torch.cuda.empty_cache()
    event(dict(stage='reloading_adapter'));loaded=PeftModel.from_pretrained(load_base(),checkpoint);loaded.eval()
    for k,v in get_peft_model_state_dict(loaded).items():
        if not torch.equal(v.cpu(),state[k]):raise RuntimeError('Reloaded adapter tensor differs: '+k)
    with torch.no_grad(),attention():_,actual=target_loss(loaded,probe)
    difference=(actual.cpu()-expected).abs().max().item()
    torch.testing.assert_close(actual.cpu(),expected,rtol=1e-3,atol=1e-3)
    write(args.output/'summary.json',dict(status='smoke_passed',optimizer_steps=4,changed_adapter_tensors=changed,
        base_weights_frozen=True,reload_max_absolute_logit_difference=difference,
        longest_sequence_tokens=31845,heldout_model_calls=0,full_sft_completed=False,
        checkpoint_use='debug_only_do_not_promote',artifacts={str(p.relative_to(args.output)):sha(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()}))
    event(dict(stage='complete',status='smoke_passed'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--model-manifest',type=Path,required=True)
    p.add_argument('--samples',type=Path,default=ROOT/'results/retail_reward_v2_sft_v1/action_samples.train.jsonl')
    p.add_argument('--release',type=Path,default=ROOT/'results/retail_reward_v2_sft_v1/validation.json')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except Exception as exc:
        write(a.output/'failure.json',dict(status='failed',error_type=type(exc).__name__,error=str(exc)));raise
