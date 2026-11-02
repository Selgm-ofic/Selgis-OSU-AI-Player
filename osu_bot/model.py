import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
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
    temp = max(1e-3, float(temperature))
    scaled_logits = logits / temp
    p = torch.softmax(scaled_logits, dim=-1)
    xy = torch.stack(((p * grid_x).sum(-1), (p * grid_y).sum(-1)), -1)
    return xy, p


class LocalClickCNN(nn.Module):
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

    def forward(self, x, state=None, temperature=0.5):
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
            "present": torch.sigmoid(a[:, 3]), "click_logit": a[:, 1],
            "hold_logit": a[:, 2], "present_logit": a[:, 3], "state": state, "last": last
        }


class UNetTinyOsuPolicy(nn.Module):
    def __init__(self, hidden=160, spatial_channels=80, heatmap_hw=(18, 24), action_fusion="tc_roi", tth_scale_ms=500.0):
        super().__init__()
        self.hm_h, self.hm_w = heatmap_hw
        self.hidden, self.spatial_channels = hidden, spatial_channels
        self.action_fusion = action_fusion
        self.tth_scale_ms = float(tth_scale_ms)
        decoder_channels = spatial_channels // 2

        self.vision = nn.ModuleDict({
            "enc1": nn.Sequential(nn.Conv2d(2, 16, 5, 2, 2), nn.GroupNorm(4, 16), nn.SiLU()),
            "enc2": nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.SiLU()),
            "enc3": nn.Sequential(nn.Conv2d(32, spatial_channels, 3, 2, 1), nn.GroupNorm(8, spatial_channels), nn.SiLU())
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
                nn.Conv2d(spatial_channels * 2, spatial_channels, 3, 1, 1), nn.SiLU(),
                nn.ConvTranspose2d(spatial_channels, decoder_channels, 4, 2, 1), nn.SiLU()
            ),
            "fuse": nn.Sequential(
                nn.Conv2d(decoder_channels + 32, decoder_channels, 3, 1, 1),
                nn.GroupNorm(4, decoder_channels), nn.SiLU(),
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

    def forward(self, x, state=None, temperature=0.5):
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

        if self.action_fusion == "tc_roi":
            c_norm = (xy.detach() * 2.0 - 1.0)
            delta = 0.28
            grid = (self.base_patch_grid * delta).unsqueeze(0) + c_norm[:, None, None, :]
            grid = grid.clamp(-1.0, 1.0)
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
            "present": torch.sigmoid(a[:, 3]), "click_logit": a[:, 1],
            "hold_logit": a[:, 2], "present_logit": a[:, 3], "state": new_state, "last": last
        }


def load_policy(path, device="cpu", return_ckpt=False):
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["state"] if isinstance(ck, dict) and "state" in ck else ck
    is_unet = (isinstance(ck, dict) and ck.get("arch") == "unet") or any(k.startswith("vision.enc") for k in state.keys())
    hm_hw = tuple(ck.get("heatmap", (18, 24))) if isinstance(ck, dict) else (18, 24)
    hidden = int(ck.get("hidden", 160 if is_unet else 128)) if isinstance(ck, dict) else (160 if is_unet else 128)
    spatial = int(ck.get("spatial_channels", 80 if is_unet else 64)) if isinstance(ck, dict) else (80 if is_unet else 64)

    if "action.weight" in state:
        action_in = state["action.weight"].shape[1]
    elif "action.tth_head.weight" in state:
        action_in = state["action.tth_head.weight"].shape[1]
    elif "action.click_head.weight" in state:
        action_in = state["action.click_head.weight"].shape[1]
    else:
        action_in = hidden

    default_fusion = "tc_roi" if action_in == (hidden + 96) else ("gru_attended" if action_in == (hidden + spatial) else "gru_only")
    action_fusion = ck.get("action_fusion", default_fusion) if isinstance(ck, dict) else default_fusion
    default_scale = 1200.0 if (not is_unet or action_fusion == "gru_attended") else 500.0
    tth_scale = float(ck.get("tth_scale_ms", default_scale)) if isinstance(ck, dict) else default_scale

    if is_unet:
        model = UNetTinyOsuPolicy(heatmap_hw=hm_hw, hidden=hidden, spatial_channels=spatial, action_fusion=action_fusion, tth_scale_ms=tth_scale)
    else:
        model = TinyOsuPolicy(heatmap_hw=hm_hw, hidden=hidden, spatial_channels=spatial, tth_scale_ms=tth_scale)

    state = dict(state)
    if "action.weight" in state:
        w, b = state["action.weight"], state["action.bias"]
        if w.shape[0] == 3:
            new_w = torch.zeros(4, w.shape[1], dtype=w.dtype)
            new_b = torch.zeros(4, dtype=b.dtype)
            new_w[0] = w[0]; new_w[2] = w[1]; new_w[3] = w[2]
            new_b[0] = b[0]; new_b[2] = b[1]; new_b[3] = b[2]
            w, b = new_w, new_b
        state["action.tth_head.weight"] = w[0:1]
        state["action.tth_head.bias"] = b[0:1]
        state["action.click_head.weight"] = w[1:2]
        state["action.click_head.bias"] = b[1:2]
        state["action.hold_head.weight"] = w[2:3]
        state["action.hold_head.bias"] = b[2:3]
        state["action.present_head.weight"] = w[3:4]
        state["action.present_head.bias"] = b[3:4]
        del state["action.weight"]
        del state["action.bias"]

    if "embed.2.weight" in state:
        ew = state["embed.2.weight"]
        target_in = model.embed[2].weight.shape[1]
        if ew.shape[1] * 4 == target_in:
            state["embed.2.weight"] = ew.repeat(1, 4) / 4.0

    model.load_state_dict(state)
    loaded_model = model.to(device).eval()
    if return_ckpt:
        return loaded_model, ck
    return loaded_model
