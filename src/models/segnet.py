"""
src/models/segnet.py

D4-equivariant bi-temporal segmentation network.

Architecture
============

Input (per branch):
    Post branch: [B, 6, 64, 64]  — VH_post, VV_post, slope, sin_asp, cos_asp, LIA
    Pre  branch: [B, 6, 64, 64]  — VH_pre,  VV_pre,  slope, sin_asp, cos_asp, LIA
Change channels (extra 4ch):     — log_ratio VH/VV, xpol post/pre  [B, 4, 64, 64]

Encoder (shared weights, D4-equivariant, 5 blocks):
    Block 1 (no pool): trivial_6 → reg×n1, output [B, n1*8, 64, 64]   skip1
    Block 2 (pool×2):  reg×n1   → reg×n2, output [B, n2*8, 32, 32]   skip2
    Block 3 (pool×2):  reg×n2   → reg×n3, output [B, n3*8, 16, 16]   skip3
    Block 4 (pool×2):  reg×n3   → reg×n4, output [B, n4*8,  8,  8]   skip4
    Block 5 (pool×2):  reg×n4   → reg×n5, output [B, n5*8,  4,  4]   bottleneck

Change feature (equivariant difference → invariant via GroupPooling):
    diff_i  = GeometricTensor(feat_post_i.tensor − feat_pre_i.tensor, type_i)
    skip_i  = GroupPooling(diff_i).tensor                  [B, n_i,  spatial_i]
    bottle  = GroupPooling(diff_5).tensor                  [B, n5,   4, 4]

Extra channels: the 4 engineered features are avg-pooled to each spatial scale
and concatenated with the skip features before decoder stages.

Dropout(0.3) on bottleneck.

Decoder (standard Conv2d — invariant features):
    Stage 1: Conv2d(n5+4→128) + BN + ELU, up×2, cat skip4 + 4ch-avg → Conv2d → [B,128, 8, 8]
    Stage 2: Conv2d(128→64)   + BN + ELU, up×2, cat skip3 + 4ch-avg → Conv2d → [B, 64,16,16]
    Stage 3: Conv2d(64→32)    + BN + ELU, up×2, cat skip2 + 4ch-avg → Conv2d → [B, 32,32,32]
    Stage 4: Conv2d(32→16)    + BN + ELU, up×2, cat skip1 + 4ch-avg → Conv2d → [B, 16,64,64]
    Final:   Conv2d(16→1, k=1)                                        → [B,  1,64,64] logit

Area head (supplementary):
    soft_mask = sigmoid(logit)  [B, 1, 64, 64]
    area_m2   = soft_mask.sum(dim=[2,3]) × (10.0)²   pixel = 10m × 10m

Parameter count: ~500–600K  (4× fewer than Gatti's 2.39M SwinV2-Tiny).

Input channel split (from CHANNEL_NAMES in preprocess.py):
    Post 6ch: indices 0-5  (VH_post, VV_post, slope, sin_asp, cos_asp, LIA)
    Pre  6ch:  indices [6, 7, 2, 3, 4, 5]  (VH_pre, VV_pre, + shared terrain)
    Extra 4ch: indices 8-11 (log_ratio_VH, log_ratio_VV, xpol_post, xpol_pre)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import escnn.nn as enn
from escnn import gspaces


# ---------------------------------------------------------------------------
# Helper: equivariant block
# ---------------------------------------------------------------------------

def _eq_block(
    in_type:  enn.FieldType,
    out_type: enn.FieldType,
    kernel:   int = 3,
    pool:     bool = False,
) -> enn.SequentialModule:
    layers = [
        enn.R2Conv(in_type, out_type, kernel_size=kernel,
                   padding=kernel // 2, bias=False),
        enn.InnerBatchNorm(out_type),
        enn.ELU(out_type, inplace=True),
    ]
    if pool:
        layers.append(enn.PointwiseAvgPool2D(out_type, kernel_size=2, stride=2))
    return enn.SequentialModule(*layers)


# ---------------------------------------------------------------------------
# Decoder block (standard Conv2d)
# ---------------------------------------------------------------------------

def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ELU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class D4SegNet(nn.Module):
    """
    D4-equivariant bi-temporal segmentation network.

    Args:
        in_ch_per_branch: Channels per encoder branch (default 6).
        extra_ch:         Engineered change channels concatenated at decoder (default 4).
        n_reg:            Regular reprs per encoder block (list of 5 ints).
        dropout_p:        Dropout probability on bottleneck (default 0.3).
    """

    # Input index slices (from the 12-channel input tensor)
    POST_IDX  = [0, 1, 2, 3, 4, 5]    # VH_post, VV_post, slope, sin_asp, cos_asp, LIA
    PRE_IDX   = [6, 7, 2, 3, 4, 5]    # VH_pre, VV_pre, slope(shared), sin(shared), cos(shared), LIA(shared)
    EXTRA_IDX = [8, 9, 10, 11]         # log_ratio_VH, log_ratio_VV, xpol_post, xpol_pre

    def __init__(
        self,
        in_ch_per_branch: int = 6,
        extra_ch:         int = 4,
        n_reg:            list[int] | None = None,
        dropout_p:        float = 0.3,
    ) -> None:
        super().__init__()

        if n_reg is None:
            n_reg = [8, 16, 32, 32, 32]
        assert len(n_reg) == 5, "n_reg must have 5 entries (one per encoder block)"

        self.in_ch_per_branch = in_ch_per_branch
        self.extra_ch         = extra_ch
        self.n_reg            = n_reg
        self.dropout_p        = dropout_p

        # ── Equivariant backbone ─────────────────────────────────────────
        gspace = gspaces.flipRot2dOnR2(N=4)
        self.gspace = gspace

        trivial_in  = enn.FieldType(gspace, [gspace.trivial_repr] * in_ch_per_branch)
        reg_types   = [
            enn.FieldType(gspace, [gspace.regular_repr] * n)
            for n in n_reg
        ]

        self.enc1 = _eq_block(trivial_in,  reg_types[0], pool=False)  # 64×64
        self.enc2 = _eq_block(reg_types[0], reg_types[1], pool=True)  # 32×32
        self.enc3 = _eq_block(reg_types[1], reg_types[2], pool=True)  # 16×16
        self.enc4 = _eq_block(reg_types[2], reg_types[3], pool=True)  #  8× 8
        self.enc5 = _eq_block(reg_types[3], reg_types[4], pool=True)  #  4× 4

        self.group_pool = [
            enn.GroupPooling(reg_types[i]) for i in range(5)
        ]
        # Register as module list so parameters are tracked
        self.gp1 = enn.GroupPooling(reg_types[0])
        self.gp2 = enn.GroupPooling(reg_types[1])
        self.gp3 = enn.GroupPooling(reg_types[2])
        self.gp4 = enn.GroupPooling(reg_types[3])
        self.gp5 = enn.GroupPooling(reg_types[4])

        self.dropout = nn.Dropout2d(p=dropout_p)

        # ── Decoder ─────────────────────────────────────────────────────
        # After GroupPooling: bottleneck has n5 channels, skips have n_i channels.
        # We concatenate extra_ch at each decoder stage (avg-pooled to match spatial).
        n1, n2, n3, n4, n5 = n_reg

        # Stage 1: bottleneck [n5, 4,4] + extra [4, 4,4] → refine → upsample → cat skip4 + extra
        self.dec1_pre = _dec_block(n5 + extra_ch,  128)                  # [128, 4, 4]
        self.dec1_up  = nn.ConvTranspose2d(128, 128, 2, stride=2)         # [128, 8, 8]
        self.dec1_post= _dec_block(128 + n4 + extra_ch, 128)             # [128, 8, 8]

        # Stage 2
        self.dec2_pre = _dec_block(128 + extra_ch, 64)                   # [64, 8, 8]
        self.dec2_up  = nn.ConvTranspose2d(64, 64, 2, stride=2)          # [64, 16, 16]
        self.dec2_post= _dec_block(64 + n3 + extra_ch, 64)              # [64, 16, 16]

        # Stage 3
        self.dec3_pre = _dec_block(64 + extra_ch, 32)                    # [32, 16, 16]
        self.dec3_up  = nn.ConvTranspose2d(32, 32, 2, stride=2)          # [32, 32, 32]
        self.dec3_post= _dec_block(32 + n2 + extra_ch, 32)              # [32, 32, 32]

        # Stage 4
        self.dec4_pre = _dec_block(32 + extra_ch, 16)                    # [16, 32, 32]
        self.dec4_up  = nn.ConvTranspose2d(16, 16, 2, stride=2)          # [16, 64, 64]
        self.dec4_post= _dec_block(16 + n1 + extra_ch, 16)              # [16, 64, 64]

        # Final 1×1 → logit
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    # ------------------------------------------------------------------

    def _encode_branch(
        self, x: torch.Tensor
    ) -> tuple[enn.GeometricTensor, enn.GeometricTensor, enn.GeometricTensor, enn.GeometricTensor, enn.GeometricTensor]:
        """
        Run shared encoder on one 6-channel branch.

        Returns (feat1, feat2, feat3, feat4, feat5) geometric tensors at
        spatial dims 64, 32, 16, 8, 4 respectively.
        """
        ft = enn.GeometricTensor(x, self.enc1.in_type)
        f1 = self.enc1(ft)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return f1, f2, f3, f4, f5

    def _group_diff(
        self,
        fpost: enn.GeometricTensor,
        fpre:  enn.GeometricTensor,
        gp:    enn.GroupPooling,
    ) -> torch.Tensor:
        """
        Equivariant change: post − pre → GroupPool → invariant tensor.
        diff is still equivariant (linear group action),
        GroupPooling maps it to a trivial (invariant) repr.
        """
        diff_tensor = fpost.tensor - fpre.tensor
        diff_geo    = enn.GeometricTensor(diff_tensor, fpost.type)
        return gp(diff_geo).tensor   # [B, n_reg_i, H_i, W_i]

    def forward(
        self,
        x12: torch.Tensor,           # [B, 12, 64, 64]
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x12: [B, 12, 64, 64]  full 12-channel normalised patch.

        Returns dict with:
            logit:   [B, 1, 64, 64]  raw logit for segmentation
            area_m2: [B, 1]          predicted deposit area in m²  (soft mask)
        """
        # ── Split channels ─────────────────────────────────────────────
        x_post  = x12[:, self.POST_IDX, :, :]   # [B, 6, 64, 64]
        x_pre   = x12[:, self.PRE_IDX,  :, :]   # [B, 6, 64, 64]
        x_extra = x12[:, self.EXTRA_IDX,:, :]   # [B, 4, 64, 64]

        # ── Encode both branches (shared weights) ─────────────────────
        fp1, fp2, fp3, fp4, fp5 = self._encode_branch(x_post)
        fr1, fr2, fr3, fr4, fr5 = self._encode_branch(x_pre)

        # ── Compute invariant change features at each scale ────────────
        # Each is [B, n_reg_i, H_i, W_i]
        s1 = self._group_diff(fp1, fr1, self.gp1)   # [B, n1, 64, 64]
        s2 = self._group_diff(fp2, fr2, self.gp2)   # [B, n2, 32, 32]
        s3 = self._group_diff(fp3, fr3, self.gp3)   # [B, n3, 16, 16]
        s4 = self._group_diff(fp4, fr4, self.gp4)   # [B, n4,  8,  8]
        bt = self._group_diff(fp5, fr5, self.gp5)   # [B, n5,  4,  4]

        bt = self.dropout(bt)

        # ── Average-pool extra channels to each spatial scale ──────────
        def _extra_at(sz: int) -> torch.Tensor:
            return F.adaptive_avg_pool2d(x_extra, (sz, sz))

        e4  = _extra_at(4)
        e8  = _extra_at(8)
        e16 = _extra_at(16)
        e32 = _extra_at(32)
        e64 = _extra_at(64)

        # ── Decoder ────────────────────────────────────────────────────
        # Stage 1: bottleneck 4×4
        x = self.dec1_pre(torch.cat([bt, e4], dim=1))   # [B, 128, 4, 4]
        x = self.dec1_up(x)                               # [B, 128, 8, 8]
        x = self.dec1_post(torch.cat([x, s4, e8], dim=1)) # [B, 128, 8, 8]

        # Stage 2
        x = self.dec2_pre(torch.cat([x, e8], dim=1))    # [B, 64, 8, 8]
        x = self.dec2_up(x)                               # [B, 64, 16, 16]
        x = self.dec2_post(torch.cat([x, s3, e16], dim=1))

        # Stage 3
        x = self.dec3_pre(torch.cat([x, e16], dim=1))
        x = self.dec3_up(x)
        x = self.dec3_post(torch.cat([x, s2, e32], dim=1))

        # Stage 4
        x = self.dec4_pre(torch.cat([x, e32], dim=1))
        x = self.dec4_up(x)
        x = self.dec4_post(torch.cat([x, s1, e64], dim=1))

        logit = self.final_conv(x)   # [B, 1, 64, 64]

        # ── Area head ──────────────────────────────────────────────────
        soft_mask = torch.sigmoid(logit)
        # Each pixel is 10m × 10m = 100 m²
        area_m2 = soft_mask.sum(dim=[2, 3]) * 100.0   # [B, 1]

        return {"logit": logit, "area_m2": area_m2}


# ---------------------------------------------------------------------------
# Ablation variant: no skip connections
# ---------------------------------------------------------------------------

class D4SegNetNoSkip(D4SegNet):
    """
    Ablation condition 1–3: same encoder, decoder without skip connections.
    Bottleneck is decoded purely top-down (no skip concatenation).
    """

    def forward(self, x12: torch.Tensor) -> dict[str, torch.Tensor]:
        x_post  = x12[:, self.POST_IDX, :, :]
        x_pre   = x12[:, self.PRE_IDX,  :, :]
        x_extra = x12[:, self.EXTRA_IDX,:, :]

        fp1, fp2, fp3, fp4, fp5 = self._encode_branch(x_post)
        fr1, fr2, fr3, fr4, fr5 = self._encode_branch(x_pre)

        bt = self._group_diff(fp5, fr5, self.gp5)
        bt = self.dropout(bt)

        def _extra_at(sz: int) -> torch.Tensor:
            return F.adaptive_avg_pool2d(x_extra, (sz, sz))

        e4 = _extra_at(4); e8 = _extra_at(8); e16 = _extra_at(16); e32 = _extra_at(32); e64 = _extra_at(64)

        # Decoder without skip concatenation — pad with zeros to match channel dims
        n1, n2, n3, n4, n5 = self.n_reg
        zeros = lambda n, sz: torch.zeros(bt.shape[0], n, sz, sz, device=bt.device, dtype=bt.dtype)

        x = self.dec1_pre(torch.cat([bt, e4], dim=1))
        x = self.dec1_up(x)
        x = self.dec1_post(torch.cat([x, zeros(n4, 8),  e8],  dim=1))

        x = self.dec2_pre(torch.cat([x, e8], dim=1))
        x = self.dec2_up(x)
        x = self.dec2_post(torch.cat([x, zeros(n3, 16), e16], dim=1))

        x = self.dec3_pre(torch.cat([x, e16], dim=1))
        x = self.dec3_up(x)
        x = self.dec3_post(torch.cat([x, zeros(n2, 32), e32], dim=1))

        x = self.dec4_pre(torch.cat([x, e32], dim=1))
        x = self.dec4_up(x)
        x = self.dec4_post(torch.cat([x, zeros(n1, 64), e64], dim=1))

        logit = self.final_conv(x)
        soft_mask = torch.sigmoid(logit)
        area_m2   = soft_mask.sum(dim=[2, 3]) * 100.0
        return {"logit": logit, "area_m2": area_m2}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(use_skip: bool = True, n_reg: list[int] | None = None) -> nn.Module:
    """
    Build the segmentation model.

    Args:
        use_skip: If True, full U-Net with skip connections.
                  If False, no-skip ablation variant.
        n_reg:    List of 5 ints for equivariant channel counts.
    """
    cls = D4SegNet if use_skip else D4SegNetNoSkip
    return cls(n_reg=n_reg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    model = D4SegNet()
    model.eval()

    x = torch.randn(2, 12, 64, 64)
    with torch.no_grad():
        out = model(x)

    print(f"logit:   {tuple(out['logit'].shape)}")
    print(f"area_m2: {tuple(out['area_m2'].shape)}")
    print(f"params:  {count_parameters(model):,}")

    model_noskip = D4SegNetNoSkip()
    print(f"params (no-skip): {count_parameters(model_noskip):,}")
