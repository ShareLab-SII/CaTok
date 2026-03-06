import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp


class DinoV3PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        return x, (h, w)


class DinoV3Rope(nn.Module):
    def __init__(self, num_freqs=16):
        super().__init__()
        self.register_buffer("periods", torch.ones(num_freqs))

    def _get_cos_sin(self, h, w, device, dtype):
        periods = self.periods.to(device=device, dtype=torch.float32)
        freqs = (2.0 * math.pi) / periods
        pos_x = torch.arange(w, device=device, dtype=torch.float32)
        pos_y = torch.arange(h, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        angles_x = grid_x.reshape(-1, 1) * freqs
        angles_y = grid_y.reshape(-1, 1) * freqs
        cos_x = torch.cos(angles_x).to(dtype)
        sin_x = torch.sin(angles_x).to(dtype)
        cos_y = torch.cos(angles_y).to(dtype)
        sin_y = torch.sin(angles_y).to(dtype)
        return cos_x, sin_x, cos_y, sin_y

    def _apply_rotary(self, x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

    def apply(self, q, k, h, w, num_special):
        if h is None or w is None:
            return q, k
        num_patches = h * w
        if q.shape[2] < num_special + num_patches:
            return q, k
        cos_x, sin_x, cos_y, sin_y = self._get_cos_sin(h, w, q.device, q.dtype)
        rope_dim = cos_x.shape[-1] * 2
        rope_total = rope_dim * 2
        head_dim = q.shape[-1]
        if head_dim < rope_total:
            return q, k
        q_special = q[..., :num_special, :]
        k_special = k[..., :num_special, :]
        q_patch = q[..., num_special : num_special + num_patches, :]
        k_patch = k[..., num_special : num_special + num_patches, :]
        q_x = self._apply_rotary(q_patch[..., :rope_dim], cos_x, sin_x)
        q_y = self._apply_rotary(q_patch[..., rope_dim:rope_total], cos_y, sin_y)
        k_x = self._apply_rotary(k_patch[..., :rope_dim], cos_x, sin_x)
        k_y = self._apply_rotary(k_patch[..., rope_dim:rope_total], cos_y, sin_y)
        q_patch = torch.cat([q_x, q_y, q_patch[..., rope_total:]], dim=-1)
        k_patch = torch.cat([k_x, k_y, k_patch[..., rope_total:]], dim=-1)
        q_tail = q[..., num_special + num_patches :, :]
        k_tail = k[..., num_special + num_patches :, :]
        q = torch.cat([q_special, q_patch, q_tail], dim=2)
        k = torch.cat([k_special, k_patch, k_tail], dim=2)
        return q, k


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        return x * self.gamma


class DinoV3Attention(nn.Module):
    def __init__(self, dim, num_heads=12, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.qkv.register_buffer("bias_mask", torch.zeros(dim * 3))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rope = rope

    def forward(self, x, hw=None, num_special=0):
        b, n, c = x.shape
        if self.qkv.bias is not None:
            bias = self.qkv.bias
            if hasattr(self.qkv, "bias_mask"):
                bias = bias * self.qkv.bias_mask
            qkv = F.linear(x, self.qkv.weight, bias)
        else:
            qkv = F.linear(x, self.qkv.weight, None)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None and hw is not None:
            q, k = self.rope.apply(q, k, hw[0], hw[1], num_special)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class DinoV3Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DinoV3Attention(dim, num_heads=num_heads, rope=rope)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=0.0)
        self.ls2 = LayerScale(dim)

    def forward(self, x, hw, num_special):
        x = x + self.ls1(self.attn(self.norm1(x), hw=hw, num_special=num_special))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV3VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=4,
    ):
        super().__init__()
        self.image_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.patch_embed = DinoV3PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.rope_embed = DinoV3Rope(num_freqs=16)
        self.blocks = nn.ModuleList(
            [
                DinoV3Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope=self.rope_embed,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, pixel_values=None, x=None):
        if pixel_values is None:
            pixel_values = x
        x, hw = self.patch_embed(pixel_values)
        batch = x.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        storage = self.storage_tokens.expand(batch, -1, -1)
        x = torch.cat([cls, storage, x], dim=1)
        num_special = 1 + self.num_register_tokens
        for blk in self.blocks:
            x = blk(x, hw=hw, num_special=num_special)
        x = self.norm(x)
        return SimpleNamespace(last_hidden_state=x)


def _strip_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict):
        if any(key.startswith("module.") for key in state):
            state = {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def build_dinov3_vitb16(checkpoint_path, img_size=224):
    model = DinoV3VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=4,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    state = _strip_state_dict(state)
    model.load_state_dict(state, strict=False)
    return model
