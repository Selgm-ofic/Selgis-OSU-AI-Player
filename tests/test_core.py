import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from osu2.model import TinyOsuPolicy, UNetTinyOsuPolicy, PolicyLoss, policy_loss, target_heatmap


class CoreTests(unittest.TestCase):
    def test_shapes_loss_backward(self):
        m = TinyOsuPolicy()
        o = m(torch.rand(3, 6, 2, 72, 96))
        self.assertEqual(tuple(o["xy"].shape), (3, 2))
        self.assertEqual(tuple(o["heatmap"].shape), (3, 432))
        self.assertIn("click", o)
        self.assertIn("hold", o)
        y = torch.tensor([[0.2, 0.7, 100.0, 1.0, 0.0, 1.0]] * 3)
        loss, _ = policy_loss(o, y, torch.ones(3))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_unet_shapes_and_uncertainty_loss(self):
        m = UNetTinyOsuPolicy()
        o = m(torch.rand(2, 6, 2, 72, 96))
        self.assertEqual(tuple(o["xy"].shape), (2, 2))
        self.assertEqual(tuple(o["heatmap"].shape), (2, 432))
        self.assertEqual(tuple(o["click"].shape), (2,))
        self.assertEqual(tuple(o["hold"].shape), (2,))
        loss_fn = PolicyLoss()
        y = torch.tensor([[0.3, 0.6, 50.0, 0.0, 1.0, 1.0]] * 2)
        loss, _ = loss_fn(o, y, torch.ones(2))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_streaming_vs_sequence_consistency(self):
        for ModelClass in [TinyOsuPolicy, lambda: UNetTinyOsuPolicy(action_fusion="gru_only")]:
            torch.manual_seed(42)
            model = ModelClass().eval()
            x_seq = torch.randn(1, 6, 2, 72, 96)
            with torch.no_grad():
                out_full = model(x_seq)
                state = None
                for i in range(6):
                    out_step = model(x_seq[:, i:i+1], state=state)
                    state = out_step["state"]
            self.assertTrue(torch.allclose(out_full["xy"], out_step["xy"], atol=1e-5))
            self.assertTrue(torch.allclose(out_full["heatmap"], out_step["heatmap"], atol=1e-5))
            self.assertTrue(torch.allclose(out_full["click"], out_step["click"], atol=1e-5))
            self.assertTrue(torch.allclose(out_full["hold"], out_step["hold"], atol=1e-5))

    def test_grid_alignment_soft_argmax(self):
        model = UNetTinyOsuPolicy()
        h, w = model.hm_h, model.hm_w
        for pt in [[0.3, 0.7], [0.15, 0.25], [0.85, 0.50], [0.50, 0.50]]:
            target_xy = torch.tensor([pt], dtype=torch.float32)
            gt_heatmap = target_heatmap(target_xy, h=h, w=w, sigma=1.0)
            logits = torch.log(gt_heatmap.clamp_min(1e-9))
            p = torch.softmax(logits, 1)
            xy = torch.stack(((p * model.grid_x).sum(1), (p * model.grid_y).sum(1)), 1)
            self.assertTrue(torch.allclose(xy, target_xy, atol=0.05))

    def test_spatial_shape_regression(self):
        for ModelClass in [TinyOsuPolicy, UNetTinyOsuPolicy]:
            m = ModelClass()
            x = torch.zeros(1, 1, 2, 72, 96)
            b, t, c, h, w = x.shape
            if isinstance(m.vision, torch.nn.ModuleDict):
                sp = m.vision["enc3"](m.vision["enc2"](m.vision["enc1"](x.reshape(b * t, c, h, w))))
            else:
                sp = m.vision(x.reshape(b * t, c, h, w))
            self.assertEqual(sp.shape[-2], 9)
            self.assertEqual(sp.shape[-1], 12)

    def test_policy_loss_precision_init(self):
        loss_fn = PolicyLoss()
        prec = loss_fn.precisions
        self.assertTrue(torch.allclose(prec, torch.tensor([1.0, 6.0, 3.0, 4.0, 0.5]), atol=1e-3))

    def test_one_euro_filter_stability_and_lag(self):
        from osu2.live import OneEuroFilter
        f = OneEuroFilter(min_cutoff=1.5, beta=0.04)
        t = 0.0
        val = 0.5
        vals = []
        for i in range(30):
            noisy = val + 0.02 * (1.0 if i % 2 == 0 else -1.0)
            filtered = f.filter(noisy, t)
            vals.append(filtered)
            t += 0.016
        jitter_filtered = abs(vals[-1] - vals[-2])
        self.assertLess(jitter_filtered, 0.01)

        f.filter(0.9, t)
        t += 0.016
        jump_next = f.filter(0.9, t)
        self.assertGreater(jump_next, 0.58)

    def test_temperature_scaling(self):
        m = TinyOsuPolicy()
        x = torch.rand(1, 1, 2, 72, 96)
        out_normal = m(x, temperature=1.0)
        out_sharp = m(x, temperature=0.5)
        self.assertEqual(tuple(out_sharp["xy"].shape), (1, 2))
        self.assertTrue(torch.all(out_sharp["xy"] >= 0.0) and torch.all(out_sharp["xy"] <= 1.0))

    def test_local_spatial_softmax_prevents_multi_modal_collapse(self):
        from osu2.model import local_spatial_softmax
        h, w = 18, 24
        ys, xs = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        gx, gy = xs.flatten(), ys.flatten()
        # Create bimodal heatmap with two distinct notes at (0.2, 0.2) and (0.8, 0.8)
        logits = torch.zeros(1, h * w)
        idx_mode1 = int(round(0.2 * (h - 1))) * w + int(round(0.2 * (w - 1)))
        idx_mode2 = int(round(0.8 * (h - 1))) * w + int(round(0.8 * (w - 1)))
        logits[0, idx_mode1] = 10.0  # Dominant mode
        logits[0, idx_mode2] = 9.0   # Secondary mode

        # Global softmax would average them towards centroid (0.5, 0.5)
        p_global = torch.softmax(logits, dim=1)
        xy_global = torch.stack(((p_global * gx).sum(1), (p_global * gy).sum(1)), 1)
        self.assertGreater(float(xy_global[0, 0]), 0.35)  # Collapsed towards center

        # Local windowed softmax isolates the dominant mode without centroid collapse
        xy_local, p_local = local_spatial_softmax(logits, gx, gy, hm_h=h, hm_w=w, window_radius=2.5)
        self.assertLess(float(xy_local[0, 0]), 0.25)
        self.assertLess(float(xy_local[0, 1]), 0.25)

    def test_action_head_separate_predictions(self):
        from osu2.model import ActionHead
        head = ActionHead(128)
        x = torch.randn(4, 128)
        out = head(x)
        self.assertEqual(tuple(out.shape), (4, 4))
        # Ensure separate submodules exist
        self.assertIsInstance(head.tth_head, torch.nn.Linear)
        self.assertIsInstance(head.click_head, torch.nn.Linear)
        self.assertIsInstance(head.hold_head, torch.nn.Linear)
        self.assertIsInstance(head.present_head, torch.nn.Linear)

    def test_focal_loss_and_weighted_tth(self):
        from osu2.model import focal_binary_loss, weighted_tth_loss
        logits = torch.tensor([5.0, -5.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0])
        fl = focal_binary_loss(logits, targets)
        self.assertTrue(torch.isfinite(fl))
        self.assertGreater(float(fl), 0.0)

        pred_tth = torch.tensor([10.0, 300.0])
        target_tth = torch.tensor([0.0, 300.0])
        wtth = weighted_tth_loss(pred_tth, target_tth)
        self.assertTrue(torch.isfinite(wtth))
        self.assertGreater(float(wtth), 0.0)

    def test_spinner_engine_orbit_and_exit(self):
        from osu2.spinner import SpinnerEngine
        engine = SpinnerEngine(rpm=420.0, radius=0.16)
        engine.reset()
        t = 0.0
        pos1 = engine.step(t, (0.5, 0.5))
        self.assertEqual(len(pos1), 4)
        self.assertEqual(pos1[2], 1.0)  # click
        self.assertEqual(pos1[3], 1.0)  # hold

        # Advance 1 full revolution at 420 RPM: period = 60 / 420 = 0.142857 s
        period = 60.0 / 420.0
        pos2 = engine.step(period, (0.5, 0.5))
        # Distance from center should match radius ~0.16
        dist = ((pos2[0] - 0.5)**2 + (pos2[1] - 0.5)**2)**0.5
        self.assertAlmostEqual(dist, 0.16, places=2)

    def test_controller_hardware_scancodes(self):
        sys.path.insert(0, r"C:\Gamma\Osu_module")
        from control.controller import Controller
        c = Controller()
        self.assertEqual(c.SCANCODES["z"], 0x2C)
        self.assertEqual(c.SCANCODES["x"], 0x2D)
        self.assertEqual(c.SCANCODES["f2"], 0x3C)

    def test_nearest_click_lookup(self):
        from osu2.data import _nearest
        stamps = np.array([0.0, 16.6, 33.3, 50.0, 66.6, 83.3])
        # A click down at 19.0 ms is closest to 16.6 ms (index 1), not ceil (index 2)
        idx = _nearest(stamps, np.array([19.0]))
        self.assertEqual(idx[0], 1)
        # searchsorted would give 2 (ceil bias +16ms)
        self.assertEqual(np.searchsorted(stamps, [19.0])[0], 2)

    def test_hit_scheduler_stream_polyphony_no_drop(self):
        sys.path.insert(0, r"C:\Gamma\Osu_module")
        from control.hit_scheduler import HitScheduler
        class MockCtrl:
            def __init__(self): self.keys = {}
            def set_key(self, k, v): self.keys[k] = v
            def release_all(self): self.keys.clear()
        ctrl = MockCtrl()
        sched = HitScheduler(ctrl, latency_ms=0, lookahead_ms=100)
        # 5-note stacked burst at 300 BPM (50ms per note)
        t = 0.0
        for _ in range(5):
            sched.update(time_to_hit_ms=0.0, hold=False, target=(0.5, 0.5), now=t, click_prob=0.95)
            t += 0.016
            sched.update(time_to_hit_ms=34.0, hold=False, target=(0.5, 0.5), now=t, click_prob=0.05)
            t += 0.034
        self.assertEqual(sched.hits, 5)

    def test_spinner_engine_uint8_obs_detection(self):
        from osu2.spinner import SpinnerEngine
        eng = SpinnerEngine()
        obs = np.zeros((2, 72, 96), dtype=np.uint8)
        # Draw concentric ring in center in uint8 [0, 255]
        for y in range(72):
            for x in range(96):
                d = np.hypot(x - 48, y - 36)
                if 8 <= d <= 18:
                    obs[0, y, x] = 230
        self.assertTrue(eng.is_active(obs=obs))

    def test_spinner_engine_multi_spinner_lifecycle(self):
        from osu2.spinner import SpinnerEngine
        eng = SpinnerEngine()
        eng.step(10.0, (0.5, 0.5))
        eng.step(12.0, (0.5, 0.5))
        # Non-spinner gap until 30s
        eng.step(30.0, (0.5, 0.5))
        self.assertEqual(eng.start_time, 30.0)

    def test_spinner_engine_beatmap_metadata_interpolation(self):
        sys.path.insert(0, r"C:\Gamma\Osu_module")
        from perception.beatmap import Beatmap, HitObject
        from osu2.spinner import SpinnerEngine
        bm = Beatmap(path='', artist='', title='', version='')
        bm.objects = [
            HitObject(x=256.0, y=192.0, time=1000.0, type=8, end_time=2000.0),
            HitObject(x=100.0, y=100.0, time=2100.0, type=1, end_time=0.0)
        ]
        eng = SpinnerEngine(radius=0.16)
        # Near end of spinner (at 1.98s, 20ms remaining <= 80ms exit duration)
        act = eng.step(1.98, (0.5, 0.5), env_or_bm=bm)
        target_norm = (100.0 / 512.0, 100.0 / 384.0) # ~ (0.195, 0.260)
        # Should have blended significantly towards target_norm compared to circular orbit
        dist_to_next = ((act[0] - target_norm[0])**2 + (act[1] - target_norm[1])**2)**0.5
        self.assertLess(dist_to_next, 0.15)


if __name__ == "__main__":
    unittest.main()
