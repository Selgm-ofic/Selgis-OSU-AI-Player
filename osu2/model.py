import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    """Separate prediction heads for TTH regression, Click classification, Hold, and Present."""
    def __init__(self, in_features):
        super().__init__()
        self.tth_head = nn.Linear(in_features, 1)
        self.click_head = nn.Linear(in_features, 1)
        self.hold_head = nn.Linear(in_features, 1)
        self.present_head = nn.Linear(in_features, 1)

    def forward(self, x):
        tth = self.tth_head(x)
        click = self.click_head(x)
        hold = self.hold_head(x)
        present = self.present_head(x)
        return torch.cat([tth, click, hold, present], dim=-1)


def spatial_softmax(logits, grid_x, grid_y, hm_h=18, hm_w=24, temperature=0.5):
    """Fully differentiable spatial softmax with temperature scaling.
    Guarantees continuous 100% gradient flow from coordinate loss across all cells.
    Temperature tau=0.5 sharpens peak probability mass without argmax deadlocks.
    """
    temp = max(1e-3, float(temperature))
    orig_dtype = logits.dtype
    logits_f = logits.float()
    scaled_logits = logits_f / temp
    scaled_logits = scaled_logits - scaled_logits.max(dim=-1, keepdim=True).values
    p = torch.softmax(scaled_logits, dim=-1)
    grid_x_f = grid_x.float()
    grid_y_f = grid_y.float()
    xy = torch.stack(((p * grid_x_f).sum(-1), (p * grid_y_f).sum(-1)), -1)
    return xy.to(orig_dtype), p.to(orig_dtype)


def local_spatial_softmax(logits, grid_x, grid_y, hm_h=18, hm_w=24, window_radius=2.5, temperature=0.5):
    temp = max(1e-3, float(temperature))
    orig_dtype = logits.dtype
    logits_f = logits.float()
    scaled = logits_f / temp
    peak_idx = torch.argmax(logits_f, dim=-1)
    py = (peak_idx // hm_w).float()
    px = (peak_idx % hm_w).float()
    ys, xs = grid_y.float().reshape(hm_h, hm_w), grid_x.float().reshape(hm_h, hm_w)
    dist = torch.sqrt((xs.unsqueeze(0) - (px / max(1, hm_w - 1)).unsqueeze(1).unsqueeze(2)) ** 2 +
                      (ys.unsqueeze(0) - (py / max(1, hm_h - 1)).unsqueeze(1).unsqueeze(2)) ** 2)
    mask = (dist.flatten(1) <= (window_radius / max(hm_h, hm_w)))
    masked_scaled = scaled.masked_fill(~mask, -1e4)
    masked_scaled = masked_scaled - masked_scaled.max(dim=-1, keepdim=True).values
    p = torch.softmax(masked_scaled, dim=-1)
    xy = torch.stack(((p * grid_x.float()).sum(-1), (p * grid_y.float()).sum(-1)), -1)
    return xy.to(orig_dtype), p.to(orig_dtype)


class TinyOsuPolicy(nn.Module):
    def __init__(self, hidden=128, spatial_channels=64, heatmap_hw=(18, 24), tth_scale_ms=1200.0):
        super().__init__()
        self.hm_h, self.hm_w = heatmap_hw
        self.hidden, self.spatial_channels = hidden, spatial_channels
        self.tth_scale_ms = float(tth_scale_ms)
        decoder_channels = spatial_channels // 2
        self.vision = nn.Sequential(
            nn.Conv2d(2, 16, 5, 2, 2), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, spatial_channels, 3, 2, 1), nn.BatchNorm2d(spatial_channels), nn.SiLU()
        )
        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(spatial_channels * 4, 128), nn.SiLU()
        )
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.context = nn.Linear(hidden, spatial_channels)
        self.heatmap = nn.Sequential(
            nn.Conv2d(spatial_channels * 2, spatial_channels, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(spatial_channels, decoder_channels, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(decoder_channels, 1, 1)
        )
        self.action = ActionHead(hidden)
        ys, xs = torch.meshgrid(torch.linspace(0, 1, self.hm_h), torch.linspace(0, 1, self.hm_w), indexing="ij")
        self.register_buffer("grid_x", xs.flatten())
        self.register_buffer("grid_y", ys.flatten())

    def forward(self, x, state=None, temperature=0.5, *args, **kwargs):
        b, t, c, h, w = x.shape
        spatial = self.vision(x.reshape(b * t, c, h, w))
        _, C, Hs, Ws = spatial.shape
        enc = self.embed(spatial).reshape(b, t, -1)
        seq, state = self.gru(enc, state)
        last = seq[:, -1]
        last_map = spatial.reshape(b, t, C, Hs, Ws)[:, -1]
        context = self.context(last)[:, :, None, None].expand(-1, -1, Hs, Ws)
        hm = self.heatmap(torch.cat((last_map, context), 1))
        hm = F.interpolate(hm, size=(self.hm_h, self.hm_w), mode="bilinear", align_corners=False)
        logits = hm.flatten(1)
        xy, _ = spatial_softmax(logits, self.grid_x, self.grid_y, hm_h=self.hm_h, hm_w=self.hm_w, temperature=temperature)
        a = self.action(last)
        return {
            "heatmap": logits, "xy": xy, "tth": torch.tanh(a[:, 0]) * self.tth_scale_ms,
            "click": torch.sigmoid(a[:, 1]), "hold": torch.sigmoid(a[:, 2]),
            "present": torch.sigmoid(a[:, 3]),
            "click_logit": a[:, 1], "hold_logit": a[:, 2], "present_logit": a[:, 3],
            "state": state, "last": last
        }


class LocalClickCNN(nn.Module):
    """Processes localized 7x7 patch centered at predicted target (x, y).
    Inputs: 32 channels from enc2 (fine edges) + 6 channels from recent input frames (absdiff motion + gray).
    Extracts approach circle diameter, contour sharpness, and contraction momentum without touching aim gradients.
    """
    def __init__(self, in_channels=38, out_features=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=0),
            nn.GroupNorm(4, 48),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, out_features),
            nn.SiLU()
        )

    def forward(self, patch):
        return self.net(patch)


class UNetTinyOsuPolicy(nn.Module):
    def __init__(self, hidden=160, spatial_channels=80, heatmap_hw=(18, 24), action_fusion="tc_roi", tth_scale_ms=500.0):
        super().__init__()
        self.hm_h, self.hm_w = heatmap_hw
        self.hidden, self.spatial_channels = hidden, spatial_channels
        self.action_fusion = action_fusion
        self.tth_scale_ms = float(tth_scale_ms)
        decoder_channels = spatial_channels // 2

        self.vision = nn.ModuleDict({
            "enc1": nn.Sequential(
                nn.Conv2d(2, 16, 5, 2, 2),
                nn.GroupNorm(4, 16),
                nn.SiLU()
            ),
            "enc2": nn.Sequential(
                nn.Conv2d(16, 32, 3, 2, 1),
                nn.GroupNorm(8, 32),
                nn.SiLU()
            ),
            "enc3": nn.Sequential(
                nn.Conv2d(32, spatial_channels, 3, 2, 1),
                nn.GroupNorm(8, spatial_channels),
                nn.SiLU()
            )
        })

        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(spatial_channels * 4, 128), nn.SiLU()
        )
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.gru_norm = nn.LayerNorm(hidden)
        self.context = nn.Linear(hidden, spatial_channels)

        self.heatmap = nn.ModuleDict({
            "up": nn.Sequential(
                nn.Conv2d(spatial_channels * 2, spatial_channels, 3, 1, 1),
                nn.SiLU(),
                nn.ConvTranspose2d(spatial_channels, decoder_channels, 4, 2, 1),
                nn.SiLU()
            ),
            "fuse": nn.Sequential(
                nn.Conv2d(decoder_channels + 32, decoder_channels, 3, 1, 1),
                nn.GroupNorm(4, decoder_channels),
                nn.SiLU(),
                nn.Conv2d(decoder_channels, 1, 1)
            )
        })

        if action_fusion == "tc_roi":
            self.roi_encoder = LocalClickCNN(in_channels=38, out_features=96)
            in_action = hidden + 96
            u = torch.linspace(-1.0, 1.0, 7)
            vy, ux = torch.meshgrid(u, u, indexing="ij")
            self.register_buffer("base_patch_grid", torch.stack([ux, vy], dim=-1))
        elif action_fusion == "gru_only":
            in_action = hidden
        else:
            in_action = hidden + spatial_channels
        self.action = ActionHead(in_action)

        ys, xs = torch.meshgrid(torch.linspace(0, 1, self.hm_h), torch.linspace(0, 1, self.hm_w), indexing="ij")
        self.register_buffer("grid_x", xs.flatten())
        self.register_buffer("grid_y", ys.flatten())

    def forward(self, x, state=None, temperature=0.5, roi_jitter=None, roi_gt=None, roi_beta=0.0, compute_action=True):
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        e1 = self.vision["enc1"](x_flat)
        e2 = self.vision["enc2"](e1)
        spatial = self.vision["enc3"](e2)

        _, C, Hs, Ws = spatial.shape
        enc = self.embed(spatial).reshape(b, t, -1)
        seq, new_state = self.gru(enc, state)
        last = self.gru_norm(seq[:, -1])

        last_map = spatial.reshape(b, t, C, Hs, Ws)[:, -1]
        context = self.context(last)[:, :, None, None].expand(-1, -1, Hs, Ws)
        up = self.heatmap["up"](torch.cat((last_map, context), 1))

        _, C2, H2, W2 = e2.shape
        skip = e2.reshape(b, t, C2, H2, W2)[:, -1]
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        hm = self.heatmap["fuse"](torch.cat((up, skip), 1))
        hm = F.interpolate(hm, size=(self.hm_h, self.hm_w), mode="bilinear", align_corners=False)
        logits = hm.flatten(1)

        xy, p = spatial_softmax(logits, self.grid_x, self.grid_y, hm_h=self.hm_h, hm_w=self.hm_w, temperature=temperature)

        if not compute_action:
            dummy_zero = torch.zeros_like(xy[:, :1])
            return {
                "heatmap": logits,
                "xy": xy,
                "last": last,
                "skip": skip,
                "state": new_state,
                "tth": dummy_zero,
                "click": dummy_zero,
                "hold": dummy_zero,
                "present": dummy_zero,
                "click_logit": dummy_zero,
                "hold_logit": dummy_zero,
                "present_logit": dummy_zero,
            }

        if self.action_fusion == "tc_roi":
            if roi_jitter is not None and roi_jitter > 1e-5 and roi_gt is not None:
                center = (1.0 - roi_beta) * roi_gt + roi_beta * xy
                c_norm = center * 2.0 - 1.0
                noise = (torch.randn_like(c_norm) * roi_jitter).clamp(-0.14, 0.14)
                c_norm = (c_norm + noise).clamp(-1.0, 1.0)
            else:
                c_norm = (xy * 2.0 - 1.0)
            delta = 0.28
            grid = (self.base_patch_grid * delta).unsqueeze(0) + c_norm[:, None, None, :]
            grid = grid.clamp(-1.0, 1.0)
            # Keep the ROI differentiable with respect to the predicted aim.  The
            # previous implementation detached both the center and the sampled
            # features, so click/hold errors could never teach the aim head where
            # the actionable object was.  This is especially damaging when two
            # visible objects are close in time.
            # Keep coordinate gradients through the sampling grid, while
            # treating the expensive feature tensors as constants.  This gives
            # the aim head a click signal without the large activation graph
            # and backward cost of the fully coupled variant.
            feat_skip = F.grid_sample(skip.detach(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            x_recent = x[:, -3:].detach()
            if x_recent.shape[1] < 3:
                first_frame = x_recent[:, :1].repeat(1, 3 - x_recent.shape[1], 1, 1, 1)
                x_recent = torch.cat([first_frame, x_recent], dim=1)
            x_recent = x_recent.reshape(b, -1, h, w)
            feat_raw = F.grid_sample(x_recent, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            patch = torch.cat([feat_skip, feat_raw], dim=1)
            local_feat = self.roi_encoder(patch)
            action_in = torch.cat([last.detach(), local_feat], dim=1)
            a = self.action(action_in)
        elif self.action_fusion == "gru_only":
            a = self.action(last)
        else:
            p_2d = p.reshape(b, 1, self.hm_h, self.hm_w)
            p_down = F.adaptive_avg_pool2d(p_2d, (Hs, Ws))
            p_weight = p_down / p_down.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
            attended_feat = (last_map * p_weight).sum(dim=(-1, -2))
            action_in = torch.cat((last, attended_feat), 1)
            a = self.action(action_in)

        return {
            "heatmap": logits, "xy": xy, "tth": torch.tanh(a[:, 0]) * self.tth_scale_ms,
            "click": torch.sigmoid(a[:, 1]), "hold": torch.sigmoid(a[:, 2]),
            "present": torch.sigmoid(a[:, 3]),
            "click_logit": a[:, 1], "hold_logit": a[:, 2], "present_logit": a[:, 3],
            "state": new_state, "last": last
        }


def target_heatmap(xy, h=18, w=24, sigma=1.25):
    ys, xs = torch.meshgrid(torch.arange(h, device=xy.device), torch.arange(w, device=xy.device), indexing="ij")
    tx = xy[:, 0, None, None] * (w - 1)
    ty = xy[:, 1, None, None] * (h - 1)
    hm = torch.exp(-((xs - tx) ** 2 + (ys - ty) ** 2) / (2 * sigma ** 2)).flatten(1)
    return hm / hm.sum(1, keepdim=True).clamp_min(1e-6)


def focal_binary_loss(logits, targets, gamma=2.0, alpha=0.50, pos_weight=2.0):
    """Calibrated focal loss for clicks preventing dominant easy negatives without inducing spam."""
    logits_f = logits.float()
    targets_f = targets.float()
    p = torch.sigmoid(logits_f)
    bce = F.binary_cross_entropy_with_logits(logits_f, targets_f, reduction="none")
    p_t = p * targets_f + (1.0 - p) * (1.0 - targets_f)
    focal_weight = (1.0 - p_t).clamp(min=0.0, max=1.0) ** gamma
    alpha_weight = alpha * targets_f + (1.0 - alpha) * (1.0 - targets_f)
    pw_weight = 1.0 + (pos_weight - 1.0) * targets_f
    return (focal_weight * alpha_weight * pw_weight * bce).mean()


def weighted_tth_loss(pred_tth, target_tth, scale=500.0):
    """Weighted L1 loss prioritizing accuracy within the critical near-hit window."""
    tth_err = torch.abs(pred_tth - target_tth.clamp(-scale, scale)) / scale
    time_weight = 1.0 + 2.0 * torch.exp(-torch.abs(target_tth) / 100.0)
    return (time_weight * tth_err).mean()


class PolicyLoss(nn.Module):
    def __init__(self, tth_scale_ms=500.0, pos_weight=2.0, weighting="kendall"):
        super().__init__()
        self.tth_scale_ms = float(tth_scale_ms)
        self.pos_weight = float(pos_weight)
        self.weighting = weighting
        self.log_vars = nn.Parameter(-torch.log(torch.tensor([1.0, 6.0, 3.0, 4.0, 0.5])))

    @property
    def precisions(self):
        return torch.exp(-self.log_vars.clamp(min=-4.0, max=4.0))

    def forward(self, o, y, valid, lambda_action=1.0):
        mask = valid > 0.5
        present = F.binary_cross_entropy_with_logits(o["present_logit"].view(-1).float(), valid.view(-1).float())
        if bool(mask.any()):
            expected = target_heatmap(y[mask, :2])
            hm = -(expected * F.log_softmax(o["heatmap"][mask].float(), 1)).sum(1).mean()
            xy = F.smooth_l1_loss(o["xy"][mask], y[mask, :2])
            scale = self.tth_scale_ms
            tth = F.smooth_l1_loss(o["tth"][mask].view(-1) / scale, y[mask, 2].clamp(-scale, scale).view(-1) / scale)
            click_pw = torch.tensor([self.pos_weight], device=y.device)
            hold_pw = torch.tensor([3.0], device=y.device)
            if y.shape[1] >= 6:
                click = focal_binary_loss(o["click_logit"][mask].view(-1), y[mask, 3].view(-1), gamma=2.0, alpha=0.75, pos_weight=self.pos_weight)
                hold = F.binary_cross_entropy_with_logits(o["hold_logit"][mask].view(-1), y[mask, 4].view(-1), pos_weight=hold_pw)
            elif y.shape[1] >= 5:
                click = focal_binary_loss(o["click_logit"][mask].view(-1), y[mask, 3].view(-1), gamma=2.0, alpha=0.75, pos_weight=self.pos_weight)
                hold = F.binary_cross_entropy_with_logits(o["hold_logit"][mask].view(-1), y[mask, 4].view(-1), pos_weight=hold_pw)
            elif y.shape[1] >= 4:
                click = focal_binary_loss(o["click_logit"][mask].view(-1), y[mask, 3].view(-1), gamma=2.0, alpha=0.75, pos_weight=self.pos_weight)
                hold = torch.tensor(0.0, device=y.device)
            else:
                click = torch.tensor(0.0, device=y.device)
                hold = torch.tensor(0.0, device=y.device)

            # Coordinate regression loss is ~1e-4 while heatmap cross-entropy is ~5.0.
            # Scaling xy by 30.0 restores balanced gradient flow alongside heatmap loss.
            xy_scaled = 30.0 * xy
            losses = torch.stack([hm, xy_scaled, tth, click, hold])
            clamped_vars = self.log_vars.clamp(min=-4.0, max=4.0)
            precision = torch.exp(-clamped_vars)
            if self.weighting == "kendall":
                aim_loss = precision[0] * hm + clamped_vars[0] + precision[1] * xy_scaled + clamped_vars[1] + present
                action_loss = precision[2] * tth + clamped_vars[2] + precision[3] * click + clamped_vars[3] + precision[4] * hold + clamped_vars[4]
                total = aim_loss + float(lambda_action) * action_loss
            elif self.weighting in ["hybrid", "decoupled"]:
                aim_loss = precision[0] * hm + clamped_vars[0] + precision[1] * xy_scaled + clamped_vars[1] + present
                action_loss = 1.0 * tth + 2.0 * click + 1.5 * hold
                total = aim_loss + float(lambda_action) * action_loss
            else:
                aim_loss = 0.3 * hm + 30.0 * xy + present
                action_loss = 15.0 * tth + 2.0 * click + 1.0 * hold
                total = aim_loss + float(lambda_action) * action_loss
        else:
            total = o["present_logit"].sum() * 0 + present
            hm = xy = tth = click = hold = torch.tensor(0.0, device=valid.device)
        return total, {"hm": hm, "xy": xy, "tth": tth, "click": click, "hold": hold, "present": present}


def policy_loss(o, y, valid, tth_scale_ms=500.0, pos_weight=2.0):
    mask = valid > 0.5
    present = F.binary_cross_entropy_with_logits(o["present_logit"].view(-1).float(), valid.view(-1).float())
    if bool(mask.any()):
        expected = target_heatmap(y[mask, :2])
        hm = -(expected * F.log_softmax(o["heatmap"][mask], 1)).sum(1).mean()
        xy = F.smooth_l1_loss(o["xy"][mask], y[mask, :2])
        scale = tth_scale_ms
        tth = F.smooth_l1_loss(o["tth"][mask].view(-1) / scale, y[mask, 2].clamp(-scale, scale).view(-1) / scale)
        click_pw = torch.tensor([pos_weight], device=y.device)
        hold_pw = torch.tensor([3.0], device=y.device)
        if y.shape[1] >= 6:
            click = F.binary_cross_entropy_with_logits(o["click_logit"][mask].view(-1), y[mask, 3].view(-1), pos_weight=click_pw)
            hold = F.binary_cross_entropy_with_logits(o["hold_logit"][mask].view(-1), y[mask, 4].view(-1), pos_weight=hold_pw)
        elif y.shape[1] >= 5:
            click = F.binary_cross_entropy_with_logits(o["click_logit"][mask].view(-1), y[mask, 3].view(-1), pos_weight=click_pw)
            hold = F.binary_cross_entropy_with_logits(o["hold_logit"][mask].view(-1), y[mask, 4].view(-1), pos_weight=hold_pw)
        else:
            click = torch.tensor(0.0, device=y.device)
            hold = torch.tensor(0.0, device=y.device)
        aim_loss = 0.3 * hm + 30.0 * xy
        action_loss = 15.0 * tth + 2.0 * click + 1.0 * hold
        total = aim_loss + action_loss + present
    else:
        total = o["present_logit"].sum() * 0 + present
        hm = xy = tth = click = hold = torch.tensor(0.0, device=valid.device)
    return total, {"hm": hm, "xy": xy, "tth": tth, "click": click, "hold": hold, "present": present}
