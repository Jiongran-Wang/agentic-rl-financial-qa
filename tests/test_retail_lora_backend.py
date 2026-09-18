import tempfile
import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import PeftModel
from training.retail.lora_backend import target_loss,add_lora


def tiny():
    torch.manual_seed(73)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        max_position_embeddings=128,attention_dropout=0.0))


def batch():
    ids=torch.tensor([[2,3,4,5,6,7,8,0,0]])
    return dict(input_ids=ids,attention_mask=torch.tensor([[1,1,1,1,1,1,1,0,0]]),
        labels=torch.tensor([[-100,-100,-100,-100,6,7,8,-100,-100]]))


class LoraBackendTests(unittest.TestCase):
    def test_selected_projection_matches_full_loss_and_gradients(self):
        m=add_lora(tiny());m.eval();b=batch()
        full=m(**b,use_cache=False).loss;full.backward()
        expected={n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}
        m.zero_grad();loss,_=target_loss(m,b);loss.backward()
        torch.testing.assert_close(loss,full,rtol=1e-5,atol=1e-6)
        for n,p in m.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p.grad,expected[n],rtol=1e-4,atol=1e-6)
            else:self.assertIsNone(p.grad)

    def test_optimizer_updates_only_adapters_and_reload_matches(self):
        m=add_lora(tiny());m.train();b=batch()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
        loss,_=target_loss(m,b);loss.backward();opt.step();m.eval()
        changed=[n for n,p in m.named_parameters() if not torch.equal(p,before[n])]
        self.assertTrue(changed);self.assertTrue(all('lora_' in n for n in changed))
        with torch.no_grad():_,expected=target_loss(m,b)
        with tempfile.TemporaryDirectory() as d:
            m.save_pretrained(d);loaded=PeftModel.from_pretrained(tiny(),d);loaded.eval()
            with torch.no_grad():_,actual=target_loss(loaded,b)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_empty_or_padding_targets_rejected(self):
        m=tiny();b=batch();b['labels'][:]=-100
        with self.assertRaises(ValueError):target_loss(m,b)
        b=batch();b['labels'][0,-1]=4
        with self.assertRaises(ValueError):target_loss(m,b)


if __name__=='__main__':unittest.main()
