"""Small synthetic smoke checks for gradients, checkpoints, evaluation and plotting.

All tensors are synthetic; tests do not claim to validate accuracy on real TL.
"""
from __future__ import annotations
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dd_punet.data import TLDataset
from dd_punet.losses import fullgrid_error_sums
from dd_punet.model import DDPUNet, count_parameters, MODEL_ARCHITECTURE
from train import train_one_epoch, validate, append_history, save_checkpoint
import evaluate


class ToyC1Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=nn.Parameter(torch.tensor(0.))
    def forward(self,ssp,bathy,angle_indices=None):
        size=9 if angle_indices is None else len(angle_indices)
        return self.bias.expand(ssp.shape[0],size,200,250)
    def predict_db(self,ssp,bathy,angle_indices=None):
        return self.forward(ssp,bathy,angle_indices=angle_indices)*20+80


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_train_validate_history_and_checkpoint(self):
        ssp=torch.ones((2,50))
        bathy=torch.ones((2,61,61))
        target=torch.full((2,9,200,250),90.)
        ds=TensorDataset(ssp,bathy,target,torch.arange(2))
        loader=DataLoader(ds,batch_size=1)
        model=ToyC1Net()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-2)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        stats={'tl_mean':80.,'tl_std':20.,'tl_scope':'full_grid'}
        args=SimpleNamespace(amp='off',angle_chunk=2,grad_accum=2,grad_clip=1.)
        tr=train_one_epoch(model,loader,optimizer,scaler,stats,args,torch.device('cpu'),1)
        self.assertEqual(tr['grid_points'],900000)
        self.assertAlmostEqual(tr['train_rmse_db'],10,places=3)
        self.assertAlmostEqual(tr['train_mae_db'],10,places=3)
        self.assertNotEqual(float(model.bias.detach()),0.)
        val=validate(model,loader,stats,args,torch.device('cpu'))
        self.assertEqual(val['overall']['valid_points'],900000)
        self.assertGreater(val['overall']['rmse_db'],0.)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            row={'epoch':1,'lr':0.001,'train_mse_normalized':0.1,
                 'train_mae_db':2.,'train_rmse_db':3.,'val_mae_db':4.,
                 'val_rmse_db':5.,'seconds':0.2}
            append_history(folder/'history.csv',row)
            with (folder/'history.csv').open(newline='') as f:
                reader=csv.DictReader(f)
                self.assertEqual(reader.fieldnames,list(row))
            save_checkpoint(folder/'best.pt',epoch=0,model=model,optimizer=optimizer,
                scheduler=scheduler,scaler=scaler,stats=stats,best_rmse=5.,bad_epochs=0,
                args=args,val_metrics=val)
            state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            self.assertIn('validation_metrics_db',state)
            self.assertEqual(state['model_name'],MODEL_ARCHITECTURE)
            self.assertIn('full-grid',state['loss_definition'])

    def test_no_required_bottom_file_in_dataset_and_statistics(self):
        from dd_punet import data
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            files={}
            for key,shape in [('ssp',(4,50)),('ssp_valid_count',(4,)),
                              ('ssp_depth_m',(50,)),('bathymetry',(4,61,61)),
                              ('tl',(4,9,2,3))]:
                files[key]=d/(key+'.npy')
                if key=='ssp': x=np.tile(np.linspace(1480,1520,50,dtype='float32'),(4,1))
                elif key=='ssp_valid_count':x=np.full(4,40,dtype='int16')
                elif key=='ssp_depth_m':x=np.arange(50,dtype='float32')
                elif key=='bathymetry':x=np.tile(np.linspace(50,250,61,dtype='float32'),(4,61,1))
                else:
                    x=np.full(shape,10000,dtype='int16')
                    x[0,0,0,0]=-32768
                np.save(files[key],x)
            # Real data uses 18876 samples; mock ONLY metadata for tiny fixture.
            with patch.object(data,'check_files',return_value=d),patch.object(data,'SPLIT_SLICES',
                    {'train':(0,3),'val':(3,4),'test':(3,4)}):
                stats=data.compute_train_stats(d,tl_invalid_raw=-32768,
                    invalid_target_fill_db=50.,chunk_size=2,verbose=False)
                self.assertEqual(stats['tl_scope'],'full_grid')
                self.assertEqual(stats['tl_valid_points'],3*9*2*3)
                # All cells counted, including one replaced sentinel =50dB.
                self.assertAlmostEqual(stats['tl_mean'],(161*100+50)/162)
                ds=TLDataset(d,'train',tl_invalid_raw=-32768,
                    invalid_target_fill_db=50.,check=False)
                self.assertEqual(float(ds[0][2][0,0,0]),50.)
                self.assertEqual(tuple(ds[0][2].shape),(9,2,3))

    def test_train_entry_full_epoch_and_artifacts(self):
        import json
        import train
        from dd_punet.model import count_parameters
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            np.save(folder/'bottom_depth_index_r2.npy', np.full((121,9,250),99,dtype=np.int16))
            np.save(folder/'ssp_depth_m.npy',np.linspace(0,4000,50,dtype='float32'))
            out=folder/'runs'
            out.mkdir()
            stats={'ssp_mean':1500.,'ssp_std':20.,'bathy_mean':2200.,'bathy_std':400.,
                   'tl_mean':90.,'tl_std':20.,'tl_scope':'full_grid',
                   'tl_invalid_raw':None,'invalid_target_fill_db':None,'train_samples':15972}
            (out/'stats.json').write_text(json.dumps(stats),encoding='utf8')
            class TinyDataset:
                def __init__(self,root,split,**kw):self.n=2 if split=='train' else 1
                def __len__(self):return self.n
                def __getitem__(self,j):
                    ssp=torch.full((50,),1500.);ssp[40:]=float('nan')
                    return ssp,torch.full((61,61),2300.),torch.full((9,200,250),100.),torch.tensor(j)
            args=SimpleNamespace(data_dir=folder,output_dir=out,epochs=1,
                batch_size=1,angle_chunk=3,grad_accum=2,workers=0,lr=1e-3,
                weight_decay=1e-4,amp='off',patience=3,grad_clip=1.,seed=2026,
                width=8,range_max_m=50000.,depth_max_m=4000.,
                tl_invalid_raw=None,invalid_target_fill_db=None,stats_chunk_size=16,
                stats_from=None,resume=None,deterministic=False)
            width8=DDPUNet(np.linspace(0,4000,50),ssp_mean=1500.,
                ssp_std=20.,bathy_mean=2200.,bathy_std=400.,tl_mean=90.,tl_std=20.,width=8)
            with patch.object(train,'parse_args',return_value=args),\
                 patch.object(train,'TLDataset',TinyDataset),\
                 patch.object(train,'check_files',return_value=folder),\
                 patch.object(train,'REFERENCE_PARAMETERS',count_parameters(width8)):
                train.main()
            self.assertTrue((out/'best.pt').is_file())
            self.assertTrue((out/'last.pt').is_file())
            self.assertTrue((out/'history.csv').is_file())
            self.assertTrue((out/'last_val_metrics.json').is_file())
            with (out/'history.csv').open(newline='') as file:
                line=next(csv.DictReader(file))
                self.assertIn('train_mse_normalized',line)
                self.assertIn('val_rmse_db',line)
            ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
            self.assertEqual(ckpt['validation_metrics_db']['overall']['valid_points'],450000)
            self.assertEqual(ckpt['model_parameters'],count_parameters(width8))

    def test_evaluate_real_model_export(self):
        # Real DD-PUNet architecture with tiny width, full 9-angle output.
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            np.save(folder/'bottom_depth_index_r2.npy', np.full((121,9,250),99,dtype=np.int16))
            np.save(folder/'ssp_depth_m.npy',np.linspace(0,4000,50,dtype='float32'))
            stats={'ssp_mean':1500.,'ssp_std':20.,'bathy_mean':2200.,'bathy_std':400.,
                   'tl_mean':90.,'tl_std':20.,'tl_scope':'full_grid',
                   'tl_invalid_raw':None,'invalid_target_fill_db':None}
            model=DDPUNet(np.linspace(0,4000,50),
                ssp_mean=1500.,ssp_std=20.,bathy_mean=2200.,bathy_std=400.,
                tl_mean=90.,tl_std=20.,width=8)
            ckpt=folder/'best.pt'
            torch.save({'model_name':MODEL_ARCHITECTURE,'model':model.state_dict(),
                        'model_parameters':count_parameters(model),'stats':stats,
                        'epoch':0,'args':{'width':8,'range_max_m':50000.,'depth_max_m':4000.}},ckpt)
            class FakeDataset:
                def __init__(self,*unused,**kw):pass
                def __len__(self):return 1
                def __getitem__(self,idx):
                    ssp=torch.full((50,),1500.);ssp[40:]=float('nan')
                    return (ssp,torch.full((61,61),2200.),
                            torch.full((9,200,250),100.),torch.tensor(18875))
            with patch.object(evaluate,'check_files',return_value=None),\
                 patch.object(evaluate,'TLDataset',FakeDataset),\
                 patch.object(evaluate,'SPLIT_SLICES',{'test':(18875,18876)}):
                result=evaluate.run_evaluation(split='test',sample_ids=[18875],
                    data_dir=folder,checkpoint=ckpt,output_dir=folder/'evaluation',
                    batch_size=1,num_workers=0,angle_chunk=3,amp='off')
            self.assertEqual(result['overall']['valid_points'],225000)
            path=folder/'evaluation/test_sample_18875.npz'
            with np.load(path) as d:
                self.assertEqual(tuple(d['pred_db'].shape),(9,200,250))
                self.assertEqual(str(d['grid_scope'].item()),'water_mask')
                self.assertEqual(int(d['valid_mask'].sum()),225000)
                self.assertTrue(np.isfinite(d['pred_db'][:,:100,:]).all())
                self.assertTrue(np.isnan(d['pred_db'][:,100:,:]).all())
                self.assertTrue(np.isfinite(d['truth_db'][:,:100,:]).all())
                self.assertTrue(np.isnan(d['truth_db'][:,100:,:]).all())


if __name__=='__main__':
    unittest.main()
