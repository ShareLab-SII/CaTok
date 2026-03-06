import math
import os
from collections.abc import Mapping
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch.cuda.amp import autocast

from catok.tokenizer import vision_transformer
from catok.tokenizer.dinov3 import build_dinov3_vitb16
try:
    from transformers import AutoImageProcessor
except ImportError:
    from transformers import AutoFeatureExtractor as AutoImageProcessor
from transformers import AutoModel

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=32,
        num_classes=10,
        time_dist=['lognorm', -0.4, 1.0],
        time_shift=0.0,
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale=3.0,
        cfg_uncond='v',
        jvp_api='funtorch',
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes

        self.time_dist = time_dist
        self.time_shift = float(time_shift or 0.0)
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    def sample_t_r(self, batch_size, device, flow_ratio=1.0, enable_bernoulli=False, force_r_zero=False):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm' or enable_bernoulli == False:
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))

        num_selected = int(flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        samples[:, 1][indices] = samples[:, 0][indices]

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        if force_r_zero:
            r_np = np.zeros_like(t_np)

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, t, r, concept, cfg_w=1.0, drop_mask=None):
        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        e = torch.randn_like(x)

        z = (1 - t_) * x + t_ * e
        
        with torch.amp.autocast("cuda", enabled=False):
            # iMF
            v_hat = model(z, t, t, concept_cond=concept, drop_mask=drop_mask)
            model_partial = partial(model, concept_cond=concept, drop_mask=drop_mask)
            jvp_args = (
                lambda z, t, r: model_partial(z, t, r),
                (z, t, r),
                (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            )

            with torch.no_grad():
                _, dudt = self.jvp_fn(*jvp_args)
            u = model(z, t, r, concept_cond=concept, drop_mask=drop_mask)
            V = u + (t_ - r_) * dudt
            error = V - (e - x)

            loss = adaptive_l2_loss(error, gamma=0.0, c=1e-3)
            ref = (error ** 2).mean()
        return loss, ref, z - (t_ * u)
        

def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t * 1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding(labels)
        return embeddings


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, **kwargs):
        super().__init__()
        self.scale = dim**0.5
        self.eps = eps
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        self.g = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

    def forward(self, x):
        return F.normalize(x, dim=-1, eps=self.eps) * self.scale * self.g


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # flasth attn can not be used with jvp
        self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MFDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)

        self.use_cond = num_classes is not None
        self.y_embedder = LabelEmbedder(num_classes, dim) if self.use_cond else None

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, r, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        H, W = x.shape[-2:]

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)                   # (N, D)
        r = self.r_embedder(r)
        # t = torch.cat([t, r], dim=-1)
        t = t + r

        # condition
        c = t
        if self.use_cond:
            y = self.y_embedder(y)               # (N, D)
            c = c + y                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class MFDiT_with_concept(MFDiT):
    def __init__(
        self,
        *args,
        num_concept=256,
        concept_dim=32,
        use_repa=False,
        encoder_depth=8,
        projector_dim=2048,
        z_dim=768,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_concept = num_concept
        self.hidden_size = kwargs["dim"]
        self.null_cond = nn.Parameter(torch.zeros(1, num_concept, concept_dim))
        torch.nn.init.normal_(self.null_cond, std=.02)
        self.concept_embedder = nn.Linear(concept_dim, self.hidden_size)
        self.y_embedder = nn.Identity()
        self.cond_drop_prob = 0.1

        self.use_repa = use_repa
        self._repa_hook = None
        self.encoder_depth = encoder_depth
        if use_repa:
            self.projector = build_mlp(self.hidden_size, projector_dim, z_dim)

    def embed_cond(self, concept_cond, drop_mask=None):
        # concept_cond: (N, K, D)
        # drop_ids: (N)
        # self.null_cond: (1, K, D)
        batch_size = concept_cond.shape[0]
        if drop_mask is None:
            # randomly drop all conditions, for classifier-free guidance
            if self.training:
                drop_ids = (
                    torch.rand(batch_size, 1, 1, device=concept_cond.device)
                    < self.cond_drop_prob
                )
                concept_cond_drop = torch.where(drop_ids, self.null_cond, concept_cond)
            else:
                concept_cond_drop = concept_cond
        else:
            # randomly drop some conditions according to the drop_mask (N, K)
            # True means keep
            concept_cond_drop = torch.where(drop_mask[:, :, None], concept_cond, self.null_cond)
        return self.concept_embedder(concept_cond_drop)

    def forward(self, x, t, r, concept_cond, drop_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        concept_cond: (N, K, D) tensor of autoencoder conditions (slots)
        """

        H, W = x.shape[-2:]

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)  # (N, D)
        r = self.r_embedder(r)  # (N, D)

        # timestep condition
        c = t + r

        concept = self.embed_cond(concept_cond, drop_mask)

        num_tokens = x.shape[1]
        x = torch.cat((x, concept), dim=1)

        for i, block in enumerate(self.blocks):
            x = block(x, c)  # (N, T, D)
            if (i + 1) == self.encoder_depth and self.use_repa:
                projected = self.projector(x)
                self._repa_hook = projected[:, :num_tokens]

        x = x[:, :num_tokens]
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x
    
class NestedSampler(nn.Module):
    def __init__(
        self,
        num_slots,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.register_buffer("arange", torch.arange(num_slots))

    def uniform_sample(self, num):
        return torch.randint(1, self.num_slots + 1, (num,))

    def sample(self, num):
        samples = self.uniform_sample(num)
        return samples

    def forward(self, batch_size, device, inference_with_n_slots=-1, inference_begin=0, bind_t=False, t=None, r=None):
        if inference_with_n_slots != -1:
            effective_slots = min(inference_with_n_slots, self.num_slots)
            begin_offset = inference_begin
        else:
            effective_slots = self.num_slots
            begin_offset = 0

        if self.training:
            if bind_t:
                delta_t = t - r
                delta_t = torch.ceil(delta_t * effective_slots)
                begin = torch.ceil(r * effective_slots)
                begin = torch.where(delta_t==0.0, 0, begin)
                delta_t = torch.where(delta_t==0.0, effective_slots-1, delta_t)
                end = begin + delta_t
                end = torch.clamp(end, max=effective_slots-1)
                begin = begin + begin_offset
                end = end + begin_offset
                slot_mask = (self.arange[None, :] >= begin[:, None]) & (self.arange[None, :]<= end[:, None])
                return slot_mask
            else:
                if inference_with_n_slots != -1:
                    b = torch.randint(1, effective_slots + 1, (batch_size,), device=device)
                else:
                    b = self.sample(batch_size).to(device)
        else:
            if inference_with_n_slots != -1:
                begin = torch.full((batch_size,), begin_offset, device=device)
                end = torch.full((batch_size,), begin_offset + effective_slots, device=device)
                end = torch.clamp(end, max=self.num_slots)
                slot_mask = (self.arange[None, :] >= begin[:, None]) & (self.arange[None, :] < end[:, None])
                return slot_mask
            b = torch.full((batch_size,), self.num_slots, device=device)

        b = torch.clamp(b, max=effective_slots)
        begin = torch.full((batch_size,), begin_offset, device=device)
        end = begin + b
        end = torch.clamp(end, max=begin_offset + effective_slots)
        slot_mask = (self.arange[None, :] >= begin[:, None]) & (self.arange[None, :] < end[:, None])
        return slot_mask


_REPA_ENCODER_PRESETS = {
    "dinov2": "facebook/dinov2-base",
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "siglip2": "google/siglip2-base-patch16-256",
}

def _resolve_repa_config(config):
    if hasattr(config, "vision_config"):
        return config.vision_config
    return config

def _resolve_repa_image_size(processor, config):
    size = None
    if processor is not None:
        size = getattr(processor, "size", None)
        if isinstance(size, dict):
            size = size.get("height") or size.get("shortest_edge") or size.get("width")
        elif isinstance(size, (list, tuple)):
            size = size[0]
    if size is None and config is not None:
        size = getattr(config, "image_size", None)
        if isinstance(size, (list, tuple)):
            size = size[0]
    return int(size) if size is not None else None

def _resolve_repa_patch_size(config):
    if config is None:
        return None
    patch_size = getattr(config, "patch_size", None)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    return patch_size

def _resolve_repa_hidden_size(config):
    if config is None:
        return None
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "embed_dim", None)
    return hidden_size

def _resolve_repa_num_register_tokens(config):
    if config is None:
        return 0
    num_register_tokens = getattr(config, "num_register_tokens", 0)
    return int(num_register_tokens or 0)

def _is_square(n):
    if n <= 0:
        return False
    root = int(math.isqrt(n))
    return root * root == n

def _apply_logit_shift(samples, shift):
    shift = float(shift)
    if shift == 0.0:
        return samples
    clipped = np.clip(samples, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    logits = logits + shift
    return 1.0 / (1.0 + np.exp(-logits))

def _resolve_time_shift(time_shift, image_size):
    if time_shift is None:
        return 0.0
    if isinstance(time_shift, (int, float)):
        return float(time_shift)
    if isinstance(time_shift, str):
        if time_shift.lower() == "resolution":
            base = 256
            scale = 1.0
            return scale * math.log2(float(image_size) / base) if image_size > 0 else 0.0
        raise ValueError(f"Unknown time_shift string: {time_shift}")
    if isinstance(time_shift, Mapping):
        mode = str(time_shift.get("mode", "fixed")).lower()
        if mode == "resolution":
            base = float(time_shift.get("base", 256))
            scale = float(time_shift.get("scale", 1.0))
            if base <= 0 or image_size <= 0:
                return 0.0
            return scale * math.log2(float(image_size) / base)
        if mode == "fixed":
            return float(time_shift.get("value", 0.0))
        raise ValueError(f"Unknown time_shift mode: {mode}")
    raise ValueError(f"Unsupported time_shift type: {type(time_shift)}")


def _strip_repa_extra_tokens(tokens, num_register_tokens=0):
    length = tokens.shape[1]
    if _is_square(length):
        return tokens
    candidates = []
    if num_register_tokens:
        candidates.append(num_register_tokens + 1)
        candidates.append(num_register_tokens)
    candidates.append(1)
    for remove in candidates + list(range(2, 9)):
        if length > remove and _is_square(length - remove):
            return tokens[:, remove:]
    return tokens

class MeanFlowSlot(nn.Module):
    def __init__(
        self,
        encoder="vit_base_patch16",
        drop_path_rate=0.1,
        enc_img_size=256,
        enc_causal=True,
        num_slots=16,
        slot_dim=256,
        norm_slots=False,
        enable_nest=False,
        enable_nest_after=-1,
        enable_mf_after=-1,
        enable_cfg_after=-1,
        enable_bernoulli_after=-1,
        cfg_w=1.0,
        vae="stabilityai/sd-vae-ft-ema",
        dit_model="DiT-B-4",
        use_repa=False,
        repa_encoder="dinov2",
        repa_encoder_depth=8,
        repa_encoder_image_size=None,
        repa_loss_weight=1.0,
        bind_t=False,
        force_r_zero=False,
        freeze_vit_encoder=False,
        freeze_encoder2slot=None,
        use_dist=False,
        flow_ratio=0.75,
        time_dist=['lognorm', -0.4, 1.0],
        time_shift=None,
        **kwargs,
    ):
        super().__init__()

        self.flow_ratio = flow_ratio
        self.use_repa = use_repa
        self.repa_loss_weight = repa_loss_weight
        self.force_r_zero = force_r_zero
        self.freeze_vit_encoder = freeze_vit_encoder
        self.freeze_encoder2slot = freeze_vit_encoder if freeze_encoder2slot is None else freeze_encoder2slot
        self.use_vae = True
        if vae is None:
            self.use_vae = False
        elif isinstance(vae, str) and vae.lower() in ["none", "null", ""]:
            self.use_vae = False
        self.repa_encoder_name = None
        self.repa_encoder_image_size = None
        self.repa_patch_size = None
        self.repa_num_register_tokens = 0
        self.repa_image_mean = None
        self.repa_image_std = None
        self.repa_processor = None
        self.repa_encoder = None

        repa_z_dim = 768
        if use_repa:
            if repa_encoder is None:
                raise ValueError("repa_encoder must be set when use_repa=True")
            repa_path = None
            if isinstance(repa_encoder, (str, os.PathLike)):
                repa_path = os.fspath(repa_encoder)
            if repa_path is not None and os.path.isfile(repa_path):
                self.repa_encoder_name = repa_path
                self.repa_processor = None
                self.repa_encoder = build_dinov3_vitb16(repa_path)
                self.repa_encoder_image_size = enc_img_size
                self.repa_patch_size = self.repa_encoder.patch_size
                self.repa_num_register_tokens = self.repa_encoder.num_register_tokens
                repa_z_dim = self.repa_encoder.embed_dim
                self.repa_image_mean = IMAGENET_DEFAULT_MEAN
                self.repa_image_std = IMAGENET_DEFAULT_STD
            else:
                repa_key = repa_encoder.lower() if isinstance(repa_encoder, str) else repa_encoder
                self.repa_encoder_name = _REPA_ENCODER_PRESETS.get(repa_key, repa_encoder)
                self.repa_processor = AutoImageProcessor.from_pretrained(self.repa_encoder_name)
                self.repa_encoder = AutoModel.from_pretrained(self.repa_encoder_name)
                if hasattr(self.repa_encoder, "vision_model"):
                    self.repa_encoder = self.repa_encoder.vision_model
                repa_config = _resolve_repa_config(self.repa_encoder.config)
                self.repa_encoder_image_size = (
                    _resolve_repa_image_size(self.repa_processor, repa_config) or enc_img_size
                )
                self.repa_patch_size = _resolve_repa_patch_size(repa_config)
                self.repa_num_register_tokens = _resolve_repa_num_register_tokens(repa_config)
                repa_z_dim = _resolve_repa_hidden_size(repa_config) or repa_z_dim
                self.repa_image_mean = getattr(self.repa_processor, "image_mean", IMAGENET_DEFAULT_MEAN)
                self.repa_image_std = getattr(self.repa_processor, "image_std", IMAGENET_DEFAULT_STD)
            if repa_encoder_image_size is not None:
                self.repa_encoder_image_size = repa_encoder_image_size
            for param in self.repa_encoder.parameters():
                param.requires_grad = False
            self.repa_encoder.eval()

        if self.use_vae:
            vae_is_mar = isinstance(vae, str) and "mar" in vae
            self.mf_input_size = enc_img_size // 8 if not vae_is_mar else enc_img_size // 16
            self.mf_in_channels = 4 if not vae_is_mar else 16
        else:
            self.mf_input_size = enc_img_size
            self.mf_in_channels = 3

        time_shift_value = _resolve_time_shift(time_shift, enc_img_size)
        self.meanflow = MeanFlow(channels=self.mf_in_channels,
                        image_size=self.mf_input_size,
                        num_classes=None,
                        time_dist=time_dist,
                        time_shift=time_shift_value,
                        cfg_scale=3.0,
                        # experimental
                        cfg_uncond='u')


        if dit_model == 'DiT-L-2':
            depth = 24
            dim = 1024 
            patch_size = 2
            num_heads = 16
            repa_encoder_depth = 8
        elif dit_model == 'DiT-B-4':
            depth = 12
            dim = 768 
            patch_size = 4
            num_heads = 12
            repa_encoder_depth = 4
        elif dit_model == 'DiT-B-16':
            depth = 12
            dim = 768 
            patch_size = 16
            num_heads = 12
            repa_encoder_depth = 4
        elif dit_model == 'DiT-XL-2':
            depth = 28
            dim = 1152 
            patch_size = 2
            num_heads = 16
            repa_encoder_depth = 8
        elif dit_model == 'DiT-XL-1':
            depth = 28
            dim = 1152 
            patch_size = 1
            num_heads = 16
            repa_encoder_depth = 8
        else:
            raise ValueError(f"Unsupported dit_model: {dit_model}")

        self.decoder = MFDiT_with_concept(
                        input_size=self.mf_input_size,
                        patch_size=patch_size,
                        in_channels=self.mf_in_channels,
                        dim=dim,
                        depth=depth,
                        num_heads=num_heads,
                        num_concept=num_slots,
                        concept_dim=slot_dim,
                        use_repa=use_repa,
                        encoder_depth=repa_encoder_depth,
                        z_dim=repa_z_dim,
                    )

        if self.use_vae:
            self.vae = AutoencoderKL.from_pretrained(vae)
            self.scaling_factor = self.vae.config.scaling_factor
            self.vae.eval().requires_grad_(False)
        else:
            self.vae = None
            self.scaling_factor = 1.0

        self.enc_img_size = enc_img_size
        self.enc_causal = enc_causal
        encoder_fn = vision_transformer.__dict__[encoder]

        self.encoder = encoder_fn(
            img_size=[enc_img_size],
            num_slots=num_slots,
            drop_path_rate=drop_path_rate,
        )
        self.num_slots = num_slots
        self.norm_slots = norm_slots
        self.num_channels = self.encoder.num_features
        
        self.encoder2slot = nn.Linear(self.num_channels, slot_dim)

        if self.freeze_vit_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        if self.freeze_encoder2slot and hasattr(self, "encoder2slot"):
            for param in self.encoder2slot.parameters():
                param.requires_grad = False

        self.nested_sampler = NestedSampler(num_slots)
        self.enable_nest = enable_nest
        self.enable_nest_after = enable_nest_after
        self.enable_mf_after = enable_mf_after
        self.enable_cfg_after = enable_cfg_after
        self.enable_bernoulli_after = enable_bernoulli_after

        self.enable_meanflow = 0
        self.cfg_w = cfg_w
        self.temp_cfg_w = 1.0

        self.bind_t = bind_t
        self.use_dist = use_dist
    
    def train(self, mode=True):
        """Keep frozen modules / VAE in eval mode while toggling train() elsewhere."""
        super().train(mode)
        if self.freeze_vit_encoder:
            self.encoder.eval()
        if self.freeze_encoder2slot and hasattr(self, "encoder2slot"):
            self.encoder2slot.eval()
        if self.use_vae:
            self.vae.eval()
        return self

    @torch.no_grad()
    def vae_encode(self, x):
        if not self.use_vae:
            return x * 2 - 1
        x = x * 2 - 1
        x = self.vae.encode(x)
        if hasattr(x, 'latent_dist'):
            x = x.latent_dist
        return x.sample().mul_(self.scaling_factor)

    @torch.no_grad()
    def vae_decode(self, z):
        if not self.use_vae:
            return (z + 1) / 2
        z = self.vae.decode(z / self.scaling_factor)
        if hasattr(z, 'sample'):
            z = z.sample
        return (z + 1) / 2

    @torch.no_grad()
    def repa_encode(self, x):
        mean = torch.tensor(self.repa_image_mean, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.repa_image_std, device=x.device).view(1, -1, 1, 1)
        x = (x - mean) / std
        if self.repa_encoder_image_size is not None and self.repa_encoder_image_size != self.enc_img_size:
            x = torch.nn.functional.interpolate(x, self.repa_encoder_image_size, mode='bicubic')
        with autocast(enabled=False):
            outputs = self.repa_encoder(pixel_values=x.float())
        if torch.is_tensor(outputs):
            tokens = outputs
        else:
            tokens = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if self.repa_patch_size is not None:
            h, w = x.shape[-2:]
            if h % self.repa_patch_size == 0 and w % self.repa_patch_size == 0:
                num_patches = (h // self.repa_patch_size) * (w // self.repa_patch_size)
                extra_tokens = 0
                if tokens.shape[1] == num_patches + self.repa_num_register_tokens + 1:
                    extra_tokens = 1 + self.repa_num_register_tokens
                elif tokens.shape[1] == num_patches + self.repa_num_register_tokens:
                    extra_tokens = self.repa_num_register_tokens
                elif tokens.shape[1] > num_patches:
                    extra_tokens = tokens.shape[1] - num_patches
                if extra_tokens:
                    tokens = tokens[:, extra_tokens:]
        tokens = _strip_repa_extra_tokens(tokens, self.repa_num_register_tokens)
        return tokens

    def encode_slots(self, x):
        if self.use_dist:
            cls_token, patch_tokens, slots = self.encoder(x, is_causal=self.enc_causal, use_dist=self.use_dist)
        else:
            slots = self.encoder(x, is_causal=self.enc_causal)
        slots = self.encoder2slot(slots)
        if self.norm_slots:
            slots_std = torch.std(slots, dim=-1, keepdim=True)
            slots_mean = torch.mean(slots, dim=-1, keepdim=True)
            slots = (slots - slots_mean) / slots_std

        if self.use_dist:
            return cls_token, patch_tokens, slots
        return slots

    def forward_with_latents_mf(self,
                             x,
                             x_vae,
                             slots,
                             z,
                             sample=False,
                             epoch=None,
                             inference_with_n_slots=-1, 
                             inference_begin=0,
                             cfg=1.0,
                             vit_cls=None,
                             vit_patch=None,
                             sample_steps=25):
        losses = {}
        batch_size = x_vae.shape[0]
        device = x_vae.device
        
        if (
            epoch is not None
            and epoch >= self.enable_nest_after
            and self.enable_nest_after != -1
        ):
            self.enable_nest = True

        if (
            epoch is not None
            and epoch >= self.enable_mf_after
            and self.enable_mf_after != -1
        ):
            self.enable_meanflow = True

        if (
            epoch is not None
            and epoch >= self.enable_cfg_after
            and self.enable_cfg_after != -1
        ):
            self.temp_cfg_w = self.cfg_w

        if (
            epoch is not None
            and epoch >= self.enable_bernoulli_after
            and self.enable_cfg_after != -1
        ):
            enable_bernoulli = True
        else:
            enable_bernoulli = False
            
        flow_ratio = self.flow_ratio if self.enable_meanflow else 1.0
        t, r = self.meanflow.sample_t_r(
            batch_size, device, flow_ratio, enable_bernoulli, force_r_zero=self.force_r_zero
        )

        if self.enable_nest or inference_with_n_slots != -1:
            drop_mask = self.nested_sampler(
                batch_size, device, 
                inference_with_n_slots=inference_with_n_slots, 
                inference_begin=inference_begin,
                bind_t=self.bind_t,
                t=t,
                r=r,
            )
        else:
            drop_mask = None
            
        if sample:
            return self.sample(slots, drop_mask=drop_mask, sample_steps=sample_steps, cfg=cfg)
            # return self.sample(slots, drop_mask=drop_mask, sample_steps=1, cfg=cfg)

        meanflow_loss, ref_loss, x_pre = self.meanflow.loss(self.decoder, x_vae, t, r, slots, self.temp_cfg_w, drop_mask)

        losses["meanflow_loss"] = meanflow_loss
        losses["ref_loss"] = ref_loss

        if self.use_repa:
            assert self.decoder._repa_hook is not None and z is not None
            z_tilde = self.decoder._repa_hook
            
            if z_tilde.shape[1] != z.shape[1]:
                z_tilde = interpolate_features(z_tilde, z.shape[1])
            
            z_tilde = F.normalize(z_tilde, dim=-1)
            z = F.normalize(z, dim=-1)
            repa_loss = -torch.sum(z_tilde * z, dim=-1)
            losses["repa_loss"] = repa_loss.mean() * self.repa_loss_weight
        
        if self.use_dist:
            if vit_patch.shape[1] != z.shape[1]:
                vit_patch = interpolate_features(vit_patch, z.shape[1])
            vit_patch = F.normalize(vit_patch, dim=-1)
            dist_loss = -torch.sum(vit_patch * z, dim=-1)
            losses["dist_loss"] = dist_loss.mean() * 0.8

        return losses
        

    def forward(self, 
                x,
                sample=False,
                epoch=None,
                inference_with_n_slots=-1,
                inference_begin=0,
                sample_steps=1,
                cfg=1.0):
        x_vae = self.vae_encode(x)
        z = self.repa_encode(x) if self.use_repa else None
        vit_cls = None
        vit_patch = None
        if self.use_dist:
            vit_cls, vit_patch, slots = self.encode_slots(x)
        else:
            slots = self.encode_slots(x)
        return self.forward_with_latents_mf(x, x_vae, slots, z, sample, epoch, inference_with_n_slots, inference_begin, cfg, vit_cls, vit_patch, sample_steps)


    @torch.no_grad()
    def sample(self, slots, drop_mask=None, sample_steps=25, cfg=1.0):
        batch_size = slots.shape[0]
        device = slots.device

        z = torch.randn(batch_size, self.mf_in_channels, self.mf_input_size, self.mf_input_size, device=device)

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        null_slots = None
        if cfg != 1.0:
            null_slots = self.decoder.null_cond.expand(batch_size, -1, -1)
            if drop_mask is not None:
                null_cond_mask = torch.ones_like(drop_mask)

        for i in range(sample_steps):
            t = torch.full((batch_size,), t_vals[i], device=device)
            r = torch.full((batch_size,), t_vals[i + 1], device=device)

            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            if cfg == 1.0:
                v = self.decoder(z, t, r, slots, drop_mask)
            else:
                z_in = torch.cat([z, z], 0)
                slots_in = torch.cat([slots, null_slots], 0)
                if drop_mask is not None:
                    drop_mask_in = torch.cat([drop_mask, null_cond_mask], 0)
                else:
                    drop_mask_in = None
                v_all = self.decoder(z_in, t.repeat(2), r.repeat(2), slots_in, drop_mask_in)
                v_cond, v_uncond = v_all.chunk(2, dim=0)
                v = v_uncond + cfg * (v_cond - v_uncond)

            z = z - (t_ - r_) * v

        samples = self.vae_decode(z)
        return samples

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def interpolate_features(x, target_len):
    """Interpolate features to match target sequence length.
    Args:
        x: tensor of shape (B, T1, D)
        target_len: desired sequence length T2
    Returns:
        tensor of shape (B, T2, D)
    """
    B, T1, D = x.shape
    H1 = W1 = int(math.sqrt(T1))
    H2 = W2 = int(math.sqrt(target_len))
    
    # Reshape to 2D spatial dimensions and move channels to second dimension
    x = x.reshape(B, H1, W1, D).permute(0, 3, 1, 2)
    
    # Interpolate
    x = F.interpolate(x, size=(H2, W2), mode='bicubic', align_corners=False)
    
    # Reshape back to sequence
    return x.permute(0, 2, 3, 1).reshape(B, target_len, D)
