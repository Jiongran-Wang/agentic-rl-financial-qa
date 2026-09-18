"""CPU integration: three changing old policies, one fixed reference, exact reloads."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import PeftModel,get_peft_model_state_dict
from training.retail.lora_backend import add_lora
from training.retail.grpo_iter_v1 import restore_adapter
from training.retail.grpo_update_v1 import action_row,selected_logps,advantages
from training.retail.grpo_update_v2 import objective,check_replay


def tiny():
    torch.manual_seed(173)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        max_position_embeddings=128,attention_dropout=0.,eos_token_id=1,pad_token_id=0))


class IterativeModelTests(unittest.TestCase):
    def test_three_rounds_fixed_anchor_and_changing_old_policy(self):
        torch.set_num_threads(1)
        model=add_lora(tiny()).eval()
        snapshot=lambda:{k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        anchor=snapshot()
        frozen={k:v.detach().clone() for k,v in model.named_parameters() if not v.requires_grad}
        saw_distinct=False
        for r in range(3):
            # New action tokens stand in for fresh round-specific sampled trajectories.
            rows=[action_row([2,3,4],[7+r*4+i,1]) for i in range(4)]
            model.eval()
            with torch.no_grad(): old=[selected_logps(model,row,'cpu') for row in rows]
            parent=snapshot()
            restore_adapter(model,anchor)
            with torch.no_grad(): ref=[selected_logps(model,row,'cpu') for row in rows]
            restore_adapter(model,parent)
            with torch.no_grad():
                for row,lp in zip(rows,old): check_replay(selected_logps(model,row,'cpu').tolist(),lp.tolist())
            if r==0:
                for a,b in zip(old,ref): torch.testing.assert_close(a,b,rtol=0,atol=0)
            else:
                saw_distinct |= any(not torch.equal(a,b) for a,b in zip(old,ref))
            trainable=[p for p in model.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(trainable,lr=1e-3,weight_decay=0.)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.enable_input_require_grads()
            for update in range(2):
                model.train(); opt.zero_grad()
                for row,o,ref_lp,adv in zip(rows,old,ref,advantages([1,0,1,0])):
                    lp=selected_logps(model,row,'cpu')
                    if update==0: check_replay(lp.detach().tolist(),o.tolist())
                    behavior=o-torch.tensor([.8,1.5]).log()
                    losses,stats=objective(lp,o,ref_lp,behavior,adv)
                    if update==0: torch.testing.assert_close(stats['ratio'],torch.ones_like(o))
                    (losses.mean()/4).backward()
                self.assertGreater(torch.nn.utils.clip_grad_norm_(trainable,1.).item(),0)
                opt.step()
            self.assertTrue(any(not torch.equal(v,parent[k]) for k,v in snapshot().items()))
            self.assertTrue(all(torch.equal(p,frozen[n]) and p.grad is None for n,p in model.named_parameters() if not p.requires_grad))
            model.eval()
            with torch.no_grad(): expected=selected_logps(model,rows[0],'cpu')
            with tempfile.TemporaryDirectory() as d:
                model.save_pretrained(d)
                loaded=PeftModel.from_pretrained(tiny(),d,is_trainable=True).eval()
                with torch.no_grad(): actual=selected_logps(loaded,rows[0],'cpu')
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                model=loaded
        self.assertTrue(saw_distinct,'Later old policy must differ from the fixed anchor')


if __name__=='__main__': unittest.main()
