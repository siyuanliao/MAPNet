"""End-to-end smoke: both main training stages -> evaluate (synthetic TL).

Uses a synthetic dataset and water-column index.
"""
import contextlib
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train
import evaluate
from aglc_net.model import TLCodec, AGLCNet, count_parameters
from aglc_net.metrics import LOSS_POLICY
from test_adapted import STATS, synthetic_bank

CONFIG=dict(widths=(8,16,24,32,40),latent_dim=32,bathy_dim=16,hidden=64,angle_dim=8)


class FakeTLDataset(Dataset):
    def __init__(self,root,split,**kwargs):
        self.split=split
        self.base={'train':0,'val':15972,'test':17424}[split]
        self.n=2 if split=='train' else 1
    def __len__(self): return self.n
    def __getitem__(self,i):
        sid=self.base+i
        ssp=np.full((50,),1500.1,dtype=np.float32)
        ssp[42:]=np.nan
        bathy=np.full((61,61),500+(sid%121)*2.,dtype=np.float32)
        tl=np.full((9,200,250),75.+i,dtype=np.float32)
        tl[:,199,:] = 90.0  # intentionally deep/below-bottom-like region
        return torch.from_numpy(ssp),torch.from_numpy(bathy),torch.from_numpy(tl),torch.tensor(sid)


class TestPipeline(TestCase):
    def test_two_stage_train_evaluate(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as tmp:
            np.save(Path(tmp)/'bottom_depth_index_r2.npy',np.full((121,9,250),99,dtype=np.int16))
            output=Path(tmp)/'runs'
            bank_path=output/'reference_bank.npz'
            stats={**STATS,'train_samples':15972,'tl_statistics_region':'all_grid_points',
                   'loss_policy':LOSS_POLICY,'invalid_target_fill_db':None}
            ref=synthetic_bank()
            def phase(stage,args,st,path,device):
                if stage=='pretrain':
                    net=TLCodec(widths=CONFIG['widths'],latent_dim=32).to(device)
                    groups=[{'params':list(net.parameters()),'lr':args.lr}]
                else:
                    net=AGLCNet(ref,st,**CONFIG).to(device)
                    for p in net.codec.encoder_blocks.parameters():p.requires_grad_(False)
                    for p in net.codec.to_latent.parameters():p.requires_grad_(False)
                    dec=train.decoder_params(net.codec)
                    rest=[*net.bathy_encoder.parameters(),*net.mapper.parameters(),*net.corrector.parameters()]
                    groups=[{'params':dec,'lr':args.lr},{'params':rest,'lr':args.lr}]
                return net,groups,count_parameters(net)
            def fake_ref(data_dir,where,invalid_raw):
                where.parent.mkdir(exist_ok=True,parents=True)
                where.write_bytes(b'fake-synthetic-bank')
                return where
            common=['--data-dir',str(Path(tmp)/'synthetic'),'--output-dir',str(output),
                    '--epochs','1','--batch-size','1','--grad-accum','2','--angle-chunk','3',
                    '--workers','0','--amp','off','--decoder-freeze-epochs','0']
            with patch.object(train,'check_files',return_value=None), \
                 patch.object(train,'compute_train_stats',return_value=stats), \
                 patch.object(train,'TLDataset',FakeTLDataset), \
                 patch.object(train,'set_phase_model',side_effect=phase), \
                 patch.object(train,'create_reference_bank',side_effect=fake_ref):
                for stage in ('pretrain','correct'):
                    with patch.object(sys,'argv',['train.py','--stage',stage,*common]):
                        train.main()
            self.assertTrue((output/'pretrain_best.pt').exists())
            self.assertTrue((output/'best.pt').exists())
            self.assertTrue((output/'history.csv').exists())
            with (output/'history.csv').open(newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertEqual(list(rows[0]),['epoch','lr','train_mse_normalized','train_mae_db',
                                             'train_rmse_db','val_mae_db','val_rmse_db','seconds'])
            ckpt=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
            self.assertEqual(ckpt['loss_policy'], LOSS_POLICY)
            self.assertEqual(ckpt['stage'],'correct')
            self.assertEqual(ckpt['validation']['overall']['valid_points'],9*200*250)
            def build_bank(*a,**kw): return AGLCNet(ref,stats,**CONFIG)
            with patch.object(evaluate,'check_files',return_value=None), \
                 patch.object(evaluate,'TLDataset',FakeTLDataset), \
                 patch.object(evaluate,'build_model',side_effect=build_bank):
                report=evaluate.run_evaluation(split='test',sample_ids=[17424],evaluate_full_split=False,
                    data_dir=tmp,checkpoint=output/'best.pt',reference_bank=bank_path,
                    output_dir=output/'evaluation',batch_size=1,num_workers=0,amp='off',angle_chunk=3)
            self.assertEqual(report['overall']['valid_points'],9*100*250)
            file=output/'evaluation/test_sample_17424.npz'
            self.assertTrue(file.exists())
            with np.load(file,allow_pickle=False) as f:
                self.assertEqual(f['grid_scope'].item(),'water_mask')
                self.assertEqual(int(f['valid_mask'].sum()),9*100*250)
                self.assertEqual(f['pred_db'].shape,(9,200,250))
                self.assertTrue(np.isfinite(f['pred_db'][:,:100,:]).all())
                self.assertTrue(np.isnan(f['pred_db'][:,100:,:]).all())
                self.assertTrue(np.isnan(f['truth_db'][0,199,0]))
