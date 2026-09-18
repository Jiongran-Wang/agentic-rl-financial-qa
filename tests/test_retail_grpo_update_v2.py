import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import Qwen3Config, Qwen3ForCausalLM
from training.retail.lora_backend import add_lora
from training.retail.grpo_update_v1 import action_row, selected_logps, advantages, objective as old_objective
from training.retail.grpo_update_v2 import objective, correction, check_replay
from scripts.retail_grpo_reuse_v2 import validate_native


def tiny():
    torch.manual_seed(173)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=97, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, max_position_embeddings=128,
        attention_dropout=0., eos_token_id=1, pad_token_id=0))


class CorrectedGrpoTests(unittest.TestCase):
    def test_correction_direction_cap_and_detach(self):
        old = torch.tensor([-2., -2., -2.], requires_grad=True)
        behavior = old.detach() - torch.tensor([.6, 1.5, 3.]).log()
        behavior.requires_grad_()
        weights, raw = correction(old, behavior)
        torch.testing.assert_close(raw, torch.tensor([.6, 1.5, 3.]))
        torch.testing.assert_close(weights, torch.tensor([.6, 1.5, 2.]))
        self.assertFalse(weights.requires_grad)
        lp = old.detach().clone().requires_grad_()
        loss, stats = objective(lp, old, old, behavior, 1, beta=0)
        loss.sum().backward()
        torch.testing.assert_close(lp.grad, -weights)
        torch.testing.assert_close(stats['ratio'], torch.ones(3))
        self.assertIsNone(old.grad)
        self.assertIsNone(behavior.grad)

    def test_ppo_clipping_independent_of_behavior_correction(self):
        for adv, expected in [(1., [-1.2, -.5]), (-1., [2., .8])]:
            lp = torch.tensor([2., .5]).log().requires_grad_()
            old = torch.zeros(2)
            behavior = torch.full((2,), -math.log(1.5))
            loss, _ = objective(lp, old, old, behavior, adv, beta=0)
            torch.testing.assert_close(loss, torch.tensor(expected)*1.5)
            loss.sum().backward()
            self.assertEqual(lp.grad[0 if adv > 0 else 1].item(), 0)

    def test_identity_reduces_to_v1_loss_and_gradient(self):
        lp = torch.tensor([-.3, -.4], requires_grad=True)
        old = torch.tensor([-.5, -.5])
        a, _ = objective(lp, old, old, old, -1.)
        b, _ = old_objective(lp, old, old, -1.)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(a.sum(), lp)[0], torch.autograd.grad(b.sum(), lp)[0])

    def test_zero_variance_retains_corrected_kl_and_episode_normalization(self):
        self.assertEqual(advantages([1, 1, 1, 1]), [0.]*4)
        lp = torch.tensor([-.3, -.4, -.5], requires_grad=True)
        old = torch.full((3,), -.6)
        behavior = old - torch.tensor([.6, 1., 1.5]).log()
        loss, stats = objective(lp, old, old, behavior, 0)
        torch.testing.assert_close(loss, .02 * stats['kl'])
        torch.testing.assert_close(stats['kl'], stats['uncorrected_kl'] * stats['correction'])
        first, _ = objective(lp[:1], old[:1], old[:1], behavior[:1], 0)
        last, _ = objective(lp[1:], old[1:], old[1:], behavior[1:], 0)
        torch.testing.assert_close(loss.mean()/104, (first.sum()+last.sum())/(104*3))
        loss.sum().backward()
        self.assertTrue((lp.grad > 0).all())

    def test_rejects_invalid_probabilities_and_teacher_drift(self):
        for behavior in [torch.zeros(3), torch.tensor([float('nan')]), torch.tensor([-1e30])]:
            with self.assertRaises(ValueError):
                correction(torch.zeros(1), behavior)
        with self.assertRaises(ValueError):
            check_replay([-.3], [-.301])
        with self.assertRaises(ValueError):
            check_replay([float('nan')], [-.3])
        self.assertEqual(check_replay([-.3], [-.3]), 0)

    def test_diagnostic_alignment_rejects_wrong_identity_and_length(self):
        r = dict(id='x', sample_index=0)
        response = dict(prompt_ids=[2,3], generated_ids=[7,1])
        d = dict(index=0,id='x',sample_index=0,turn=0,prompt_tokens=2,output_tokens=2,teacher_logps=[-.3,-.4])
        validate_native(d,0,r,0,response)
        for key, value in [('index',1),('id','y'),('turn',1),('teacher_logps',[-.3])]:
            bad = copy.deepcopy(d); bad[key]=value
            with self.assertRaises(ValueError):
                validate_native(bad,0,r,0,response)

    def test_two_corrected_updates_frozen_base_and_exact_adapter_reload(self):
        model = add_lora(tiny()).eval()
        rows = [action_row([2,3,4], [7+i,1]) for i in range(4)]
        adv = advantages([1,0,1,0])
        with torch.no_grad():
            old = [selected_logps(model,r,'cpu') for r in rows]
        behavior = [lp - torch.tensor([.65,1.65]).log() for lp in old]
        initial = {n:p.detach().clone() for n,p in model.named_parameters()}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=0)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        for update in range(2):
            model.train(); opt.zero_grad()
            for r,lp0,b,a in zip(rows,old,behavior,adv):
                lp = selected_logps(model,r,'cpu')
                if update == 0: check_replay(lp.detach().tolist(),lp0.tolist())
                losses,_ = objective(lp,lp0,lp0,b,a)
                (losses.mean()/4).backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            self.assertTrue(torch.isfinite(norm) and norm > 0)
            opt.step()
        changed = [n for n,p in model.named_parameters() if not torch.equal(p,initial[n])]
        self.assertTrue(changed and all('lora_' in n for n in changed))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        model.eval()
        with torch.no_grad(): expected=selected_logps(model,rows[0],'cpu')
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded=PeftModel.from_pretrained(tiny(),directory).eval()
            with torch.no_grad(): actual=selected_logps(loaded,rows[0],'cpu')
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            for k,v in get_peft_model_state_dict(model).items():
                torch.testing.assert_close(v,get_peft_model_state_dict(loaded)[k],rtol=0,atol=0)

if __name__ == '__main__': unittest.main()
