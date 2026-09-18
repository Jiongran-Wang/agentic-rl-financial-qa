#!/usr/bin/env python3
"""Two epochs from the pinned base; save both adapters, never touch held-out data."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smoke_retail_lora_v1 import REVISION,sha,write
from training.retail.sft_data import tokenize_sample
from training.retail.lora_backend import add_lora,target_loss,tensor_batch
from training.retail.sft_plan_v1 import EPOCHS,SEED,ACCUMULATION,epoch_groups,learning_rate


def run(a):
    import torch,transformers,peft,accelerate
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel,get_peft_model_state_dict
    from torch.nn.attention import sdpa_kernel,SDPBackend
    versions={n:m.__version__ for n,m in [('torch',torch),('transformers',transformers),('peft',peft),('accelerate',accelerate)]}
    for n,v in dict(torch='2.6.0',transformers='4.51.3',peft='0.15.2',accelerate='1.6.0').items():
        if versions[n].split('+')[0]!=v:raise ValueError(f'{n} version mismatch: {versions[n]}')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA bf16 GPU required')
    manifest=json.loads(a.model_manifest.read_text())
    if manifest['repo_id']!='Qwen/Qwen3-4B' or manifest['revision']!=REVISION or a.model.resolve()!=Path(manifest['snapshot_path']).resolve():
        raise ValueError('Original pinned base model required')
    if (a.model/'adapter_config.json').exists():raise ValueError('Cannot initialize from an adapter')
    release=json.loads(a.release.read_text())
    if sha(a.samples)!=release['training_export_hashes']['action_samples.train.jsonl']:raise ValueError('Dataset hash mismatch')
    for n,h in release['tokenizer']['files'].items():
        if sha(a.model/n)!=h:raise ValueError('Tokenizer hash mismatch: '+n)
    tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    samples=[json.loads(l) for l in a.samples.read_text().splitlines() if l.strip()]
    rows=[dict(id=s['id'],**tokenize_sample(tokenizer,s)) for s in samples]
    if len(rows)!=204 or len({s['task_id'] for s in samples})!=26 or max(len(r['input_ids']) for r in rows)!=31845:
        raise ValueError('Dataset size or context budget mismatch')
    total=sum(len(epoch_groups(len(rows),e)) for e in range(1,EPOCHS+1))
    write(a.output/'config.json',dict(experiment='retail-sft-pilot-v1',versions=versions,seed=SEED,
        base_revision=REVISION,initial_adapter=None,epochs=EPOCHS,microbatch=1,accumulation=ACCUMULATION,
        optimizer_updates=total,action_samples=204,distinct_questions=26,maximum_sequence_tokens=31845,
        lora_rank=16,lora_alpha=32,precision='bf16',gradient_checkpointing=True,attention='sdpa',
        optimizer='AdamW',weight_decay=0.01,learning_rate_peak=5e-5,warmup_updates=6,final_lr=5e-6,
        loss_reduction='Mean of per-action mean token losses within each accumulation group',
        dataset_sha256=sha(a.samples),reserved_data_used=False,
        checkpoint_selection='Save both epochs; no selection from training loss or reserved data.'))
    write(a.output/'order.json',{'epochs':[[[rows[i]['id'] for i in group] for group in epoch_groups(len(rows),e)] for e in range(1,EPOCHS+1)]})
    def event(x):
        print(json.dumps(x),flush=True)
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    def attention():return sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION])
    def base():
        return AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda')
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    event(dict(stage='loading_original_base'));model=add_lora(base());model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    trainable=[p for p in model.parameters() if p.requires_grad]
    initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    optimizer=torch.optim.AdamW(trainable,lr=learning_rate(1,total),weight_decay=0.01)
    probe_row=min(rows,key=lambda r:(len(r['input_ids']),r['id']))
    checkpoints=[];expected_probes={};update=0;start_run=time.time()
    for epoch in range(1,EPOCHS+1):
        model.train();epoch_loss=0.;seen=0
        for group in epoch_groups(len(rows),epoch):
            update+=1;started=time.time();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            lr=learning_rate(update,total)
            for g in optimizer.param_groups:g['lr']=lr
            event(dict(stage='update_started',epoch=epoch,update=update,ids=[rows[i]['id'] for i in group]))
            losses=[]
            for i in group:
                row=rows[i];batch=tensor_batch(row,'cuda')
                with attention():
                    loss,logits=target_loss(model,batch);del logits
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    (loss/len(group)).backward()
                losses.append(loss.item());del batch,loss
            grads=[p.grad for p in trainable if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):raise RuntimeError('Missing/nonfinite gradients')
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
            if not torch.isfinite(norm) or norm.item()==0:raise RuntimeError('Zero/nonfinite gradient norm')
            optimizer.step()
            if any(not p.requires_grad and (p.grad is not None or p._version!=frozen[n]) for n,p in model.named_parameters()):
                raise RuntimeError('Base weights were modified')
            epoch_loss+=sum(losses);seen+=len(group)
            event(dict(stage='update_passed',epoch=epoch,update=update,learning_rate=lr,
                mean_action_loss=sum(losses)/len(losses),action_losses=losses,gradient_norm=norm.item(),
                seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            del grads;gc.collect()
        if seen!=204:raise RuntimeError('Incomplete epoch')
        model.eval();probe=tensor_batch(probe_row,'cuda')
        with torch.no_grad(),attention():probe_loss,probe_logits=target_loss(model,probe)
        expected_probes[epoch]=probe_logits.cpu();del probe,probe_loss,probe_logits
        state=get_peft_model_state_dict(model)
        changed=sum(not torch.equal(v.cpu(),initial[k]) for k,v in state.items());del state
        if not changed:raise RuntimeError('No adapter updates')
        directory=a.output/f'epoch_{epoch}';model.save_pretrained(directory,safe_serialization=True)
        report=dict(epoch=epoch,updates=update,mean_training_action_loss=epoch_loss/seen,
            changed_adapter_tensors=changed,path=str(directory),
            files={p.name:sha(p) for p in sorted(directory.iterdir()) if p.is_file()})
        checkpoints.append(report);write(a.output/'checkpoints.json',checkpoints)
        event(dict(stage='checkpoint_saved',**report))
    del optimizer,trainable,model;gc.collect();torch.cuda.empty_cache()
    for report in checkpoints:
        epoch=report['epoch'];directory=Path(report['path'])
        event(dict(stage='reload_started',epoch=epoch))
        loaded=PeftModel.from_pretrained(base(),directory);loaded.eval()
        probe=tensor_batch(probe_row,'cuda')
        with torch.no_grad(),attention():loss,logits=target_loss(loaded,probe)
        actual=logits.cpu();expected=expected_probes[epoch]
        torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-3)
        # Validate serialized tensors against the freshly loaded adapter too.
        from safetensors.torch import load_file
        saved=load_file(str(directory/'adapter_model.safetensors'),device='cpu')
        loaded_state=get_peft_model_state_dict(loaded)
        if set(saved)!=set(loaded_state) or any(not torch.equal(saved[k],v.cpu()) for k,v in loaded_state.items()):
            raise RuntimeError('Reloaded adapter tensors differ')
        report['reload_max_absolute_logit_difference']=(actual-expected).abs().max().item()
        report['reload_passed']=True;write(a.output/'checkpoints.json',checkpoints)
        del loaded,loaded_state,saved,probe,loss,logits;gc.collect();torch.cuda.empty_cache()
    write(a.output/'summary.json',dict(status='sft_training_completed_reload_verified',epochs=2,
        optimizer_updates=update,examples_presented=408,distinct_questions=26,
        base_weights_frozen=True,initial_adapter=None,reserved_policy_calls=0,
        seconds_since_model_loaded=time.time()-start_run,checkpoints=checkpoints,
        policy_quality='Not evaluated; next run compares both epochs on development tasks.'))
    event(dict(stage='complete',updates=update))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--model-manifest',type=Path,required=True)
    p.add_argument('--samples',type=Path,default=ROOT/'results/retail_reward_v2_sft_v1/action_samples.train.jsonl')
    p.add_argument('--release',type=Path,default=ROOT/'results/retail_reward_v2_sft_v1/validation.json')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except Exception as exc:
        write(a.output/'failure.json',dict(status='failed',error_type=type(exc).__name__,error=str(exc)));raise
