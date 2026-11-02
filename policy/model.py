"""GRU-политика: фичи треков -> курсор + tth + hold.

v2 — legacy flat 52-dim (GRU 128).  v3 — object-centric transformer over
8 tracks (w/h/conf/one-hot, 117-dim) + GRU.  Оба хранят POLICY_VERSION
в чекпоинте, loader автоматически выбирает архитектуру.

Выход (4): [cursor_x, cursor_y] in 0..1 (sigmoid),
           [tth_norm] in -1..1 (tanh -> *1200ms),
           [hold] in 0..1 (sigmoid).
"""
import torch
import torch.nn as nn
import numpy as np
import os
from policy.features import FEATURE_DIM, FEATURE_DIM_V3, MAX_TRACKS, PER_TRACK_V3

POLICY_VERSION = 4
TTH_SCALE_MS = 1200.0
CURSOR_SPEED_SCALE = 6.0
MAX_CURSOR_STEP = 0.30
AIM_HORIZON_S = 0.25
# The geometric controller already points at the selected object.  Keep the
# learned term corrective: at 0.60 it could dominate the aim vector and cause
# persistent overshoot/oscillation under noisy live detections.
VELOCITY_RESIDUAL_SCALE = 0.25


class _TrackEncoder(nn.Module):
    def __init__(self, per_track=PER_TRACK_V3, d_model=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(per_track, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4,
                                               dim_feedforward=128, dropout=0.1,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pool_score = nn.Sequential(
            nn.Linear(d_model, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, tracks):  # [B, MAX_TRACKS, per_track]
        # padding mask: tracks that are all-zero are not attended
        pad = (tracks.abs().sum(dim=-1) == 0)  # [B,8]
        B = tracks.size(0)
        nonempty = ~pad.all(dim=1)
        pooled = tracks.new_zeros((B, 66))
        all_scores = tracks.new_full((B, MAX_TRACKS), -1e4)
        if not bool(nonempty.any()):
            return pooled, all_scores
        active_tracks = tracks[nonempty]
        active_pad = pad[nonempty]
        x = self.proj(active_tracks)
        x = self.encoder(x, src_key_padding_mask=active_pad)
        scores = self.pool_score(x).squeeze(-1)
        # Urgency is an explicit prior, while the learned score can override it
        # for slider state and other context-dependent target choices.
        scores = scores + 0.35 * active_tracks[:, :, 6]
        scores = scores.masked_fill(active_pad, -1e4)
        weights = torch.softmax(scores * 2.0, dim=1).unsqueeze(-1)
        encoded = (x * weights).sum(dim=1)
        target_xy = (active_tracks[:, :, :2] * weights).sum(dim=1)
        active_pooled = torch.cat([encoded, target_xy], dim=1)
        pooled[nonempty] = active_pooled
        all_scores[nonempty] = scores
        return pooled, all_scores


class PolicyGRU(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.is_v3 = (input_dim == FEATURE_DIM_V3)
        if self.is_v3:
            # per-track transformer + cursor tail
            self.track_enc = _TrackEncoder(PER_TRACK_V3, 64)
            self.cursor_proj = nn.Sequential(nn.Linear(5, 16), nn.ReLU())
            gru_in = 66 + 16
        else:
            gru_in = input_dim
        self.gru = nn.GRU(gru_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )

    def _prepare(self, x):  # x [B,T,FEATURE_DIM*]
        if not self.is_v3:
            return x, None
        if x.shape[-1] != self.input_dim:
            # allow v2 tensors to flow through a v3-typed model during tests
            # by falling back to flat path -> still finite
            if x.shape[-1] == FEATURE_DIM:
                return x, None  # let GRU handle legacy shape via its own gru_in
            raise RuntimeError(f"feature dim {x.shape[-1]} != model dim {self.input_dim}")
        # split v3 feats into tracks + cursor tail
        # feats = [8*14 tracks | 5 cursor]
        B, T, F = x.shape
        tracks = x[:, :, :MAX_TRACKS * PER_TRACK_V3].view(B, T, MAX_TRACKS, PER_TRACK_V3)
        cursor = x[:, :, MAX_TRACKS * PER_TRACK_V3:]  # [B,T,5]
        # encode each time-step's set of tracks independently
        # -> [B*T,8,14] -> attention-selected [B*T,66] -> [B,T,66]
        bt = B * T
        tr = tracks.reshape(bt, MAX_TRACKS, PER_TRACK_V3)
        pooled, track_scores = self.track_enc(tr)
        pooled = pooled.view(B, T, 66)
        track_scores = track_scores.view(B, T, MAX_TRACKS)
        cur_emb = self.cursor_proj(cursor)  # [B,T,16]
        target_xy = pooled[:, :, 64:66]
        has_target = (tracks.abs().sum(dim=(-1, -2)) > 0).float().unsqueeze(-1)
        aim_velocity = torch.clamp(
            (target_xy - cursor[:, :, :2]) /
            (AIM_HORIZON_S * CURSOR_SPEED_SCALE), -1.0, 1.0,
        ) * has_target
        return (torch.cat([pooled, cur_emb], dim=-1), aim_velocity[:, -1],
                track_scores[:, -1], has_target[:, -1])

    def forward(self, x: torch.Tensor, h=None, return_aux=False):
        prepared = self._prepare(x)
        if len(prepared) == 2:
            x, aim_velocity = prepared
            track_scores = None
            has_target = None
        else:
            x, aim_velocity, track_scores, has_target = prepared
        out, h = self.gru(x, h)
        y = self.head(out[:, -1])
        learned_velocity = torch.tanh(y[:, :2])
        cursor_velocity = (learned_velocity if aim_velocity is None else
                           torch.clamp(aim_velocity +
                                       VELOCITY_RESIDUAL_SCALE * learned_velocity,
                                       -1.0, 1.0))
        if has_target is not None:
            cursor_velocity = cursor_velocity * has_target
        time_to_hit = torch.tanh(y[:, 2:3])
        hold = torch.sigmoid(y[:, 3:4])
        action = torch.cat([cursor_velocity, time_to_hit, hold], dim=1)
        if return_aux:
            return action, h, {"track_scores": track_scores}
        return action, h

    @torch.no_grad()
    def act(self, feat: np.ndarray, h=None):
        self.eval()
        t = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
        if next(self.parameters()).is_cuda:
            t = t.cuda()
        y, h = self.forward(t, h)
        return y.squeeze(0).cpu().numpy(), h

    @torch.no_grad()
    def act_with_target(self, feat: np.ndarray, h=None):
        """Return policy action and the learned target slot for v3 policies."""
        self.eval()
        t = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
        if next(self.parameters()).is_cuda:
            t = t.cuda()
        y, h, aux = self.forward(t, h, return_aux=True)
        scores = aux.get("track_scores")
        target_slot = None
        if scores is not None:
            target_slot = int(torch.argmax(scores[0]).item())
        return y.squeeze(0).cpu().numpy(), h, target_slot

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state": self.state_dict(), "feat_dim": self.input_dim,
                    "hidden": self.hidden, "policy_version": POLICY_VERSION,
                    "tth_scale_ms": TTH_SCALE_MS,
                    "output_kind": "cursor_velocity_v1",
                    "cursor_speed_scale": CURSOR_SPEED_SCALE}, path)

    @classmethod
    def load(cls, path: str, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        ver = int(ck.get("policy_version", 0))
        if ver != POLICY_VERSION or ck.get("output_kind") != "cursor_velocity_v1":
            raise ValueError(
                "absolute-cursor policy is unsafe/incompatible; retrain closed-loop v4"
            )
        m = cls(input_dim=ck.get("feat_dim", FEATURE_DIM),
                hidden=ck.get("hidden", 128))
        m.load_state_dict(ck["state"])
        m.eval()
        return m

    def export_onnx(self, path: str):
        self.eval()
        dummy = torch.randn(1, 1, self.input_dim)
        torch.onnx.export(self, (dummy, None), path,
                          input_names=["feat"], output_names=["action", "h"],
                          dynamic_axes={"feat": {0: "batch"}},
                          opset_version=17)
