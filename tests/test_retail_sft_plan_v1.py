import unittest
from training.retail.sft_plan_v1 import epoch_groups,learning_rate,PEAK_LR


class PilotPlanTests(unittest.TestCase):
    def test_every_example_once_each_epoch(self):
        first=epoch_groups(204,1);second=epoch_groups(204,2)
        self.assertEqual(len(first)+len(second),102)
        for groups in [first,second]:
            self.assertEqual(sorted(i for g in groups for i in g),list(range(204)))
            self.assertTrue(all(len(g)==4 for g in groups))
        self.assertNotEqual(first,second);self.assertEqual(first,epoch_groups(204,1))

    def test_partial_accumulation_group_kept(self):
        groups=epoch_groups(7,1)
        self.assertEqual([len(g) for g in groups],[4,3])
        self.assertEqual(sorted(i for g in groups for i in g),list(range(7)))

    def test_warmup_decay_and_nonzero_final_update(self):
        rates=[learning_rate(i,102) for i in range(1,103)]
        self.assertEqual(rates[5],PEAK_LR)
        self.assertTrue(all(a<b for a,b in zip(rates[:5],rates[1:6])))
        self.assertTrue(all(a>=b for a,b in zip(rates[5:-1],rates[6:])))
        self.assertAlmostEqual(rates[-1],PEAK_LR*0.1)
        with self.assertRaises(ValueError):learning_rate(103,102)


if __name__=='__main__':unittest.main()
