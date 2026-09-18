"""Backend test run on FASTER before the pilot loads the 4B weights."""
import unittest
import torch
from test_retail_lora_backend import tiny,batch
from training.retail.lora_backend import add_lora,target_loss


class AccumulationTests(unittest.TestCase):
    def test_accumulated_action_mean_matches_joint_objective(self):
        m=add_lora(tiny());m.eval()
        a=batch();b=batch();b['labels'][0,4]=-100
        x,_=target_loss(m,a);y,_=target_loss(m,b);((x+y)/2).backward()
        expected={n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}
        m.zero_grad(set_to_none=True)
        for row in [a,b]:
            loss,_=target_loss(m,row);(loss/2).backward()
        for n,p in m.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p.grad,expected[n],rtol=1e-4,atol=1e-6)
            else:self.assertIsNone(p.grad)


if __name__=='__main__':unittest.main()
