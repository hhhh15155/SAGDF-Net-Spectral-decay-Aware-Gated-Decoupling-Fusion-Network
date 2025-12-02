# vsda_net.py - VSDANet: Volumetric Spectral Decay Attention Network (updated)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import PatchEmbeddings, PositionalEmbeddings


# ==================== TriScale Feature Extractor ====================

class BasicConv(nn.Module):
    """Basic convolution block with optional BN and activation"""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class TriScaleBlock(nn.Module):
    """
    Triple-Scale Block with three parallel branches

    Args:
        in_planes: Input channels
        out_planes: Output channels
        stride: Stride for convolution
        scale: Residual scaling factor
        map_reduce: Channel reduction ratio
    """

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(TriScaleBlock, self).__init__()
        self.scale = scale
        inter_planes = in_planes // map_reduce

        # Branch 0: Standard 3×3 convolution
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3,
                      stride=1, padding=1, relu=False)
        )

        # Branch 1: Horizontal decomposition (1×3 → 3×1) + dilated conv
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3),
                      stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1),
                      stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3,
                      stride=1, padding=5, dilation=5, relu=False)
        )

        # Branch 2: Vertical decomposition (3×1 → 1×3) + dilated conv
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1),
                      stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3),
                      stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3,
                      stride=1, padding=5, dilation=5, relu=False)
        )

        # Feature fusion
        self.conv_linear = BasicConv(6 * inter_planes, out_planes,
                                     kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes,
                                  kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.conv_linear(out)

        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class TriScale(nn.Module):
    """
    Triple-Scale Feature Extractor

    Args:
        in_channels: Number of input channels
        emb_dim: Embedding dimension
        num_blocks: Number of TriScale blocks
    """

    def __init__(self, in_channels, emb_dim=128, num_blocks=2):
        super(TriScale, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(TriScaleBlock(64, 64, map_reduce=8))

        self.projection = nn.Sequential(
            nn.Conv2d(64, emb_dim, kernel_size=1),
            nn.BatchNorm2d(emb_dim)
        )

        def forward(self, x):
            x = self.stem(x)
            for block in self.blocks:
                x = block(x)
            out = self.projection(x)
            return out

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        out = self.projection(x)
        return out


# ==================== SD-Transformer (with spectral decay bias) ====================

class SpectralDecayPE(nn.Module):
    """
    Spectral Decay Position Encoding (真正可用版本)

    根据 token 序列长度 N 生成 [num_heads, N, N] 的衰减偏置矩阵：
        bias[h, i, j] = |i - j| * decay[h]

    其中每个 head 有不同的衰减率 decay[h]（负值，表示对远距离施加 penalty）。
    """

    def __init__(self, embed_dim, num_heads, initial_value=2.0, heads_range=4.0):
        """
        Args:
            embed_dim: 保留作兼容用，不实际使用
            num_heads: 注意力头数
            initial_value: 初始衰减强度
            heads_range: 各个 head 之间衰减跨度
        """
        super().__init__()
        self.num_heads = num_heads
        self.initial_value = initial_value
        self.heads_range = heads_range

        head_ids = torch.arange(num_heads, dtype=torch.float32)
        # 生成每个 head 一个衰减率（负值）
        decay = torch.log(
            1.0 - 2.0 ** (-initial_value - heads_range * head_ids / num_heads)
        )  # [num_heads]
        self.register_buffer("decay", decay)

    def generate_decay_bias(self, length: int, device=None, dtype=None):
        """
        Args:
            length: 序列长度 N（即 token 个数）
        Returns:
            decay_bias: [num_heads, N, N]
        """
        device = device if device is not None else self.decay.device
        index = torch.arange(length, device=device, dtype=torch.float32)
        distance = (index[:, None] - index[None, :]).abs()  # [N, N]

        decay = self.decay.view(self.num_heads, 1, 1)  # [H, 1, 1]
        bias = distance.unsqueeze(0) * decay  # [H, N, N]
        if dtype is not None:
            bias = bias.to(dtype)
        return bias

    def forward(self, seq_len: int, device=None, dtype=None):
        return self.generate_decay_bias(seq_len, device=device, dtype=dtype)


class GDFN(nn.Module):
    """
    Gated Depthwise Feed-Forward Network

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension
        bias: Whether to use bias
    """

    def __init__(self, dim, hidden_dim, bias=True):
        super(GDFN, self).__init__()

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_dim * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, H, W):
        B, N, C = x.size()
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = x.reshape(B, C, N).transpose(1, 2)
        return x


class SDTransformerLayer(nn.Module):
    """
    单层 Spectral Decay-aware Transformer Block

    结构：
        x -> LN -> MHSA(with spectral decay bias) -> Dropout + Residual
          -> LN -> GDFN(带DWConv) -> Dropout + Residual
    """

    def __init__(self, dim, num_heads, use_spectral_decay=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_spectral_decay = use_spectral_decay

        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # QKV linear projection
        self.qkv = nn.Linear(dim, dim * 3)
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # GDFN
        self.ffn = GDFN(dim, dim * 2, bias=True)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, H, W, spectral_decay_bias=None):
        """
        Args:
            x: [B, N, D]
            H, W: spatial size (H*W = N)
            spectral_decay_bias: [num_heads, N, N] or None
        """
        B, N, D = x.shape

        # ---- Multi-head Self-Attention ----
        residual = x
        x_norm = self.norm1(x)  # [B, N, D]

        qkv = self.qkv(x_norm)  # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # each: [B, N, D]

        # -> [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

        # Add spectral decay bias
        if self.use_spectral_decay and spectral_decay_bias is not None:
            # spectral_decay_bias: [H, N, N] -> [1, H, N, N]
            attn_scores = attn_scores + spectral_decay_bias.unsqueeze(0)

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        out = self.proj(out)
        out = self.proj_dropout(out)

        x = residual + out  # Residual 1

        # ---- Gated Depthwise FFN ----
        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm, H, W)
        x_ffn = self.ffn_dropout(x_ffn)

        x = residual + x_ffn  # Residual 2

        return x


class SD_Transformer(nn.Module):
    """
    Spectral Decay-aware Transformer (真正使用谱衰减偏置版本)

    Args:
        dim: Feature dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        use_spectral_decay: Whether to use spectral decay position encoding
    """

    def __init__(self, dim, num_layers=2, num_heads=4, use_spectral_decay=True):
        super(SD_Transformer, self).__init__()
        self.use_spectral_decay = use_spectral_decay
        self.num_heads = num_heads
        self.dim = dim

        if use_spectral_decay:
            self.spectral_decay_pe = SpectralDecayPE(
                embed_dim=dim,
                num_heads=num_heads,
                initial_value=2.0,
                heads_range=4.0
            )
        else:
            self.spectral_decay_pe = None

        self.layers = nn.ModuleList(
            [
                SDTransformerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    use_spectral_decay=use_spectral_decay,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, D] where N = H * W
        Returns:
            x: [B, N, D]
        """
        B, N, D = x.shape

        # Infer H, W (assuming square)
        H = W = int(math.sqrt(N))
        if H * W != N:
            raise ValueError(f"N = {N} cannot be reshaped to square HxW. Check patching.")

        # Precompute spectral decay bias once, shared across layers
        if self.use_spectral_decay and self.spectral_decay_pe is not None:
            spectral_decay_bias = self.spectral_decay_pe(
                seq_len=N, device=x.device, dtype=x.dtype
            )  # [H, N, N]
        else:
            spectral_decay_bias = None

        for layer in self.layers:
            x = layer(x, H, W, spectral_decay_bias=spectral_decay_bias)

        return x


# ==================== VSDF ====================

class VSDF(nn.Module):
    """
    Volumetric Spectral-Spatial Decoupling Fusion

    Args:
        dim: Feature dimension
        num_patches: Number of spatial patches
        num_heads: Number of attention heads
        reduction: Channel reduction ratio
    """

    def __init__(self, dim, num_patches, num_heads=4, reduction=4):
        super(VSDF, self).__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Spectral attention (for HSI)
        self.spectral_qkv = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.LayerNorm(dim * 3)
        )
        self.spectral_proj = nn.Linear(dim, dim)

        # Spatial attention (for LiDAR)
        self.spatial_qkv = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.LayerNorm(dim * 3)
        )
        self.spatial_proj = nn.Linear(dim, dim)

        # Volumetric fusion
        self.volumetric_fusion = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

        # Adaptive three-way gating
        self.adaptive_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1)
        )

        # Feature recalibration
        self.feature_recalibration = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, hsi_tokens, lidar_tokens):
        B, N, D = hsi_tokens.shape

        # Spectral Attention
        qkv_spectral = self.spectral_qkv(hsi_tokens)
        q_s, k_s, v_s = qkv_spectral.chunk(3, dim=-1)

        q_s = q_s.reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        k_s = k_s.reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        v_s = v_s.reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)

        q_s = F.normalize(q_s, dim=-1)
        k_s = F.normalize(k_s, dim=-1)

        attn_spectral = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_spectral = F.softmax(attn_spectral, dim=-1)

        out_spectral = (attn_spectral @ v_s).transpose(1, 2).reshape(B, N, D)
        out_spectral = self.spectral_proj(out_spectral)

        # Spatial Attention
        qkv_spatial = self.spatial_qkv(lidar_tokens)
        q_sp, k_sp, v_sp = qkv_spatial.chunk(3, dim=-1)

        q_sp = q_sp.reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        k_sp = k_sp.reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        v_sp = v_sp.reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)

        q_sp = F.normalize(q_sp, dim=-1)
        k_sp = F.normalize(k_sp, dim=-1)

        attn_spatial = (q_sp @ k_sp.transpose(-2, -1)) * self.temperature
        attn_spatial = F.softmax(attn_spatial, dim=-1)

        out_spatial = (attn_spatial @ v_sp).transpose(1, 2).reshape(B, N, D)
        out_spatial = self.spatial_proj(out_spatial)

        # Volumetric Fusion
        volumetric_feat = self.volumetric_fusion(out_spectral + out_spatial)

        # Adaptive Gating
        hsi_global = out_spectral.mean(dim=1)
        lidar_global = out_spatial.mean(dim=1)
        concat_feat = torch.cat([hsi_global, lidar_global], dim=-1)
        gate_weights = self.adaptive_gate(concat_feat)

        w_spectral = gate_weights[:, 0].view(B, 1, 1)
        w_spatial = gate_weights[:, 1].view(B, 1, 1)
        w_volumetric = gate_weights[:, 2].view(B, 1, 1)

        fused = (w_spectral * out_spectral +
                 w_spatial * out_spatial +
                 w_volumetric * volumetric_feat)

        # Feature Recalibration
        fused_recal = self.feature_recalibration(fused)
        fused = fused + self.alpha * fused_recal + self.beta * (hsi_tokens + lidar_tokens) / 2

        return fused


# ==================== VSDANet Main Network ====================

class VSDANet(nn.Module):
    """
    VSDANet: Volumetric Spectral Decay Attention Network

    Architecture:
        HSI/LiDAR → TriScale → Patch Embed → SD-Transformer → VSDF → Classifier

    Args:
        input_channels1: Number of HSI input channels
        input_channels2: Number of LiDAR input channels
        n_classes: Number of classification classes
        patch_size: Size of spatial patch
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        patch_size2: Size of sub-patch (default: 1)
        emb_dim: Embedding dimension
        num_blocks: Number of TriScale blocks
        use_spectral_decay: Whether to use spectral decay position encoding
    """

    def __init__(self,
                 input_channels1=30,
                 input_channels2=1,
                 n_classes=11,
                 patch_size=11,
                 num_layers=2,
                 num_heads=4,
                 patch_size2=1,
                 emb_dim=128,
                 num_blocks=2,
                 use_spectral_decay=True):
        super(VSDANet, self).__init__()

        self.emb_dim = emb_dim
        self.num_patches = (patch_size // patch_size2) ** 2
        self.use_spectral_decay = use_spectral_decay

        print("=" * 80)
        print("Initializing VSDANet (Updated SD-Transformer)")
        print(f"  - Triple-Scale Feature Extraction: Enabled")
        print(f"  - Spectral Decay Position Encoding: {'Enabled' if use_spectral_decay else 'Disabled'}")
        print(f"  - Volumetric Decoupling Fusion: Enabled")
        print("=" * 80)

        # Dual-stream feature extractors
        self.hsi_triscale = TriScale(
            in_channels=input_channels1,
            emb_dim=emb_dim,
            num_blocks=num_blocks
        )
        self.lidar_triscale = TriScale(
            in_channels=input_channels2,
            emb_dim=emb_dim,
            num_blocks=num_blocks
        )

        # Patch embedding layers
        self.patch_embeddings = PatchEmbeddings(patch_size2, emb_dim, emb_dim)
        self.pos_embeddings = PositionalEmbeddings(self.num_patches, emb_dim)

        # SD-Transformer (updated)
        self.sd_transformer = SD_Transformer(
            dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_spectral_decay=use_spectral_decay
        )

        # VSDF
        self.vsdf = VSDF(
            dim=emb_dim,
            num_patches=self.num_patches,
            num_heads=num_heads,
            reduction=4
        )

        # Classifier head
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, hsi, lidar):
        """
        Args:
            hsi: [B, 1, C, H, W] or [B, C, H, W]
            lidar: [B, 1, H, W]
        Returns:
            logits: [B, n_classes]
        """
        if hsi.dim() == 5:
            hsi = hsi.squeeze(1)

        # Triple-scale feature extraction
        hsi_feat = self.hsi_triscale(hsi)      # [B, emb_dim, H, W]
        lidar_feat = self.lidar_triscale(lidar)  # [B, emb_dim, H, W]

        # Patch embedding with positional encoding
        hsi_tokens = self.pos_embeddings(self.patch_embeddings(hsi_feat))      # [B, N, D]
        lidar_tokens = self.pos_embeddings(self.patch_embeddings(lidar_feat))  # [B, N, D]

        # Early fusion for global context
        fused_tokens = (hsi_tokens + lidar_tokens) / 2

        # SD-Transformer (global context modeling with spectral decay)
        context_tokens = self.sd_transformer(fused_tokens)   # [B, N, D]

        # VSDF (spectral-spatial decoupling fusion)
        decoupled_tokens = self.vsdf(hsi_tokens, lidar_tokens)  # [B, N, D]

        # Combine
        final_tokens = (context_tokens + decoupled_tokens) / 2

        # Classification
        global_feat = final_tokens.mean(dim=1)
        global_feat = self.norm(global_feat)
        logits = self.classifier(global_feat)

        return logits


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 80)
    print("VSDANet: Volumetric Spectral Decay Attention Network (Updated SD-Transformer)")
    print("=" * 80)

    model = VSDANet(
        input_channels1=30,
        input_channels2=1,
        n_classes=11,
        patch_size=11,
        num_layers=2,
        num_heads=4,
        emb_dim=128,
        num_blocks=2,
        use_spectral_decay=True
    )

    hsi = torch.randn(2, 1, 30, 11, 11)
    lidar = torch.randn(2, 1, 11, 11)

    print("\n--- Testing Forward Pass ---")
    model.eval()
    with torch.no_grad():
        logits = model(hsi, lidar)

    print(f"\nInput Shapes:")
    print(f"  HSI:   {hsi.shape}")
    print(f"  LiDAR: {lidar.shape}")
    print(f"\nOutput Shape: {logits.shape}")
    print(f"Parameters:   {count_parameters(model):,}")
    print("\n" + "=" * 80)
