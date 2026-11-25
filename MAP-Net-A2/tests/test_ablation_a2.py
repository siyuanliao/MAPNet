"""A2消融：确认M2被删除且极简适配正确，不破坏M1/M3/M4。"""
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from map_net_a2.model import MAPNetA2, MODEL_CONFIG, PAPER_MODULES, count_parameters


class A2AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = MAPNetA2(
            torch.linspace(0, 5000, 50), ssp_mean=1500, ssp_std=30,
            bathy_mean=2500, bathy_std=1000, tl_mean=75, tl_std=20,
        )

    def test_only_two_fc_m2_and_no_original_dpf_layers(self):
        model = self.model
        self.assertFalse(hasattr(model, "path"))
        self.assertFalse(hasattr(model, "film"))
        self.assertEqual(PAPER_MODULES["A2_M2_INTERFACE_ONLY"], ("global_context", "path_adapter"))
        self.assertEqual(count_parameters(model.global_context), 98_560)
        self.assertEqual(count_parameters(model.path_adapter), 33_280)
        self.assertEqual(count_parameters(model), 19_752_572)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.path_adapter.modules()), 1)
        self.assertTrue(all(not k.startswith("path.") and not k.startswith("film.")
                            for k in model.state_dict()))

    def test_interfaces_shapes_and_no_pathwise_terrain_encoder(self):
        model = self.model.eval()
        ssp = torch.full((2, 50), 1500.0)
        ssp[:, 40:] = float('nan')
        bathy = torch.linspace(2000, 4000, 61 * 61).reshape(1,61,61).repeat(2,1,1)
        with torch.no_grad():
            s, features, g = model._module1_encode_environment(ssp, bathy)
            self.assertEqual([x.shape for x in features],
                             [(2,48,61,61),(2,96,31,31),(2,192,16,16)])
            path, bottom, context = model._module2_interface_without_dpf(s, g, bathy)
            self.assertEqual(path.shape, (2,9,128,250))
            self.assertEqual(context.shape, (2,256))
            self.assertEqual(bottom.shape, (2,9,250))
            self.assertTrue(torch.isfinite(path).all())
            self.assertTrue(torch.isfinite(bottom).all())
            self.assertEqual(model._module3_interact_azimuths(path).shape, path.shape)

            # 固定相同的context：改变原始地形只影响M4所需bottom_ray，不能影响path。
            altered = bathy + torch.linspace(0,100,61*61).reshape(1,61,61)
            path2, bottom2 = model.path_adapter(context, altered)
            self.assertTrue(torch.equal(path, path2))
            self.assertFalse(torch.equal(bottom, bottom2))

    def test_center_ray_geometry_is_original_zero_offset(self):
        model = self.model
        bathy = torch.arange(61 * 61, dtype=torch.float32).reshape(1,61,61)
        context = torch.randn(1,256)
        with torch.no_grad():
            _, bottom = model.path_adapter(context, bathy)
        # 原DPF在9个方位/3条偏移路径中选取中心偏移0的一条。
        angles = torch.arange(9,dtype=torch.float32)*torch.pi*40/180
        ranges = torch.linspace(0.0, 50000.0, 250)
        x = ranges[None,:]*angles.sin()[:,None]
        y = ranges[None,:]*angles.cos()[:,None]
        grid = torch.stack((x/60000., y/60000.),dim=-1)[None]
        expected = F.grid_sample(bathy[:,None],grid,mode="bilinear",
                                 padding_mode="border",align_corners=True).squeeze(1)
        torch.testing.assert_close(bottom, expected, rtol=1e-6, atol=2e-3)
        torch.testing.assert_close(bottom[:,:,0], bathy[:,30,30,None].expand(1,9), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
