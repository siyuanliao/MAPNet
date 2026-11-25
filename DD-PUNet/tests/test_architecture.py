import unittest
import torch
from dd_punet.model import DDPUNet,count_parameters,REFERENCE_PARAMETERS


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def build(self):
        return DDPUNet(torch.linspace(0.,5000.,50),
            ssp_mean=1500.,ssp_std=25.,bathy_mean=2300.,bathy_std=500.,
            tl_mean=85.,tl_std=18.,width=32)

    def test_parameter_budget(self):
        m=self.build()
        self.assertEqual(count_parameters(m),19_760_708)
        self.assertLess(abs(count_parameters(m)/REFERENCE_PARAMETERS-1),0.10)

    def test_paper_mechanism(self):
        m=self.build()
        self.assertEqual(len(m.encoder.blocks),5)
        self.assertEqual(len(m.structure_decoder.up),4)
        self.assertEqual(len(m.detail_decoder.up),4)
        self.assertIsNot(m.structure_decoder,m.detail_decoder)
        self.assertEqual(m.gdam.structure[0].kernel_size,(5,5))
        self.assertEqual(m.gdam.detail[0].kernel_size,(3,3))
        self.assertEqual(m.gdam.att[1].kernel_size,(1,1))

    def test_physical_azimuth_grids(self):
        m=self.build()
        grid=m.rasterizer.ray_grid
        self.assertEqual(tuple(grid.shape),(9,250,2))
        self.assertTrue(torch.allclose(grid[:,0,:],torch.zeros_like(grid[:,0,:]),atol=1e-5))
        self.assertAlmostEqual(grid[0,-1,0].item(),0.,places=5)
        self.assertGreater(grid[0,-1,1].item(),0.8)

    def test_forward_and_backward(self):
        m=self.build().train()
        ssp=torch.full((1,50),1510.)
        ssp[:,38:]=float('nan')
        bathy=torch.full((1,61,61),2400.)
        total,s,d=m(ssp,bathy,angle_indices=[0],return_components=True)
        self.assertEqual(tuple(total.shape),(1,1,200,250))
        self.assertTrue(torch.isfinite(total).all())
        self.assertTrue(torch.allclose(total,s+d))
        # Both independently parameterized branches receive gradients.
        (s.square().mean()+d.square().mean()).backward()
        self.assertIsNotNone(m.encoder.blocks[0][0].weight.grad)
        self.assertIsNotNone(m.gdam.structure[0].weight.grad)
        self.assertIsNotNone(m.structure_decoder.out.weight.grad)
        self.assertIsNotNone(m.detail_decoder.out.weight.grad)

    def test_bad_ssp_rejected(self):
        m=self.build().eval()
        with self.assertRaises(ValueError):
            with torch.no_grad():
                m(torch.full((1,50),float('nan')),torch.ones((1,61,61)),angle_indices=[0])

if __name__=='__main__':unittest.main()
