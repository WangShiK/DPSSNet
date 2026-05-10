import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm import Mamba

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath, trunc_normal_


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)




class FReLU(nn.Module):
    #https://arxiv.org/pdf/2007.11824
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))

class ConvBNFReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, bias=False, inplace=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        self.norm = norm_layer(out_channels)
        self.act = FReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x






class FillterBlock(nn.Module):
    def __init__(self, dim, out_dim):
        super(FillterBlock, self).__init__()
        self.dim = dim
        self.out_dim = out_dim

        self.DWT = DWTForward(J=1, mode='symmetric', wave='sym4')
        self.IWT = DWTInverse(mode='symmetric', wave='sym4')
        self.Conv = ConvBNFReLU(dim, out_dim, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_weight = self.avg_pool(x)
        yL, yH = self.DWT(x)
        y_min = torch.min(yH[0], dim=2)[0]
        y_max = torch.max(yH[0], dim=2)[0]
        y_range = (y_max - y_min) + 1e-6
        y_mean = torch.mean(yH[0], dim=2)
        normal_y = (yL - y_mean + 1e-6) / (yL + y_mean + 1e-6)
        normal_y = torch.cat([normal_y.unsqueeze(2), y_range.unsqueeze(2), y_mean.unsqueeze(2)], dim=2)

        fillter_x = self.IWT((yL, [normal_y])) * x_weight



        return self.Conv(fillter_x)












class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # BCHW

        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        out = x.permute(0, 3, 2, 1)  # BHWC

        return out


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.dwconv = DWConv(self.d_inner)
        self.act = nn.SiLU()
        self.haar_conv = Conv(self.d_inner, d_state * 4, kernel_size=1)
        self.DWT = DWTForward(J=1, mode='symmetric', wave='haar')

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_haar = self.haar_conv(x)
        yL, yH = self.DWT(x_haar)
        y_mean = torch.mean(yH[0], dim=2).view(B, 1, -1, L)
        yL = yL.view(B, 1, -1, L)

        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) * yL + 1e-6 # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) * y_mean + 1e-6 # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        z = self.dwconv(z)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')

        return out


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
# 还原 b和hw
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)




class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # BCHW

        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        out = x.permute(0, 3, 2, 1)  # BHWC

        return out

class Mlp(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNAct(in_features, hidden_features, kernel_size=1)
        # self.fc2 = nn.Sequential(nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features),
        #                          norm_layer(hidden_features),
        #                          act_layer())
        self.fc2 = DWConv(hidden_features)

        self.fc3 = ConvBN(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.drop(x)

        return x



class DMB(nn.Module):
    def __init__(self, dim=64, patch=128, mlp_ratio=4., num_heads=8, drop_rate=0.):
        super(DMB, self).__init__()
        self.FAVMamba = SS2D(d_model=dim, patch=patch, d_state=16)
        self.mamba_block = Mamba(d_model=dim,  # Model dimension d_model
                                 d_state=32,  # SSM state expansion factor
                                 d_conv=4,  # Local convolution width
                                 expand=2,  # Block expansion factor
                                 )
        self.ln = nn.LayerNorm(normalized_shape=dim)
        self.perconv = ConvBNFReLU(dim*2, dim, kernel_size=1)
        self.drop = nn.Dropout(drop_rate)


    def forward(self, x):
        b, c, h, w = x.size()
        mamba_x = self.mamba_block(self.ln(x.reshape(b, -1, c))).reshape(b, c, h, w)
        ss2d_x = self.FAVMamba(x)
        out = self.perconv(torch.cat([ss2d_x, mamba_x], dim=1))
        out = self.drop(out)
        return out + x


class DetailPath(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        dim1 = embed_dim // 4
        dim2 = embed_dim // 2
        self.dp1 = nn.Sequential(ConvBNFReLU(3, dim1, stride=2, inplace=False),
                                 ConvBNFReLU(dim1, dim1, stride=1, inplace=False))
        self.dp2 = nn.Sequential(ConvBNFReLU(dim1, dim2, stride=2, inplace=False),
                                 ConvBNFReLU(dim2, dim2, stride=1, inplace=False))
        self.dp3 = nn.Sequential(ConvBNFReLU(dim2, embed_dim, stride=1, inplace=False),
                                 ConvBNFReLU(embed_dim, embed_dim, stride=1, inplace=False))

    def forward(self, x):
        feats1 = self.dp1(x)
        feats2 = self.dp2(feats1)
        feats3 = self.dp3(feats2)

        return feats1, feats3




class LMFFM(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=256):
        super().__init__()
        self.pre_conv0 = Conv(encoder_channels[0], decoder_channels, kernel_size=1)
        self.pre_conv1 = Conv(encoder_channels[1], decoder_channels, kernel_size=1)
        self.pre_conv2 = Conv(encoder_channels[2], decoder_channels, kernel_size=1)
        self.pre_conv3 = Conv(encoder_channels[3], decoder_channels, kernel_size=1)

        self.post_conv3 = nn.Sequential(ConvBNFReLU(decoder_channels, decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNFReLU(decoder_channels, decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNFReLU(decoder_channels, decoder_channels))

        self.post_conv2 = nn.Sequential(ConvBNFReLU(decoder_channels, decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNFReLU(decoder_channels, decoder_channels))

        self.post_conv1 = ConvBNFReLU(decoder_channels, decoder_channels)
        self.post_conv0 = ConvBNFReLU(decoder_channels, decoder_channels)
        self.cat_weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)

    def upsample_add(self, up, x):
        up = F.interpolate(up, x.size()[-2:], mode='nearest')
        up = up + x
        return up

    def forward(self, x0, x1, x2, x3):
        x3 = self.pre_conv3(x3)
        x2 = self.pre_conv2(x2)
        x1 = self.pre_conv1(x1)
        x0 = self.pre_conv0(x0)

        x2 = self.upsample_add(x3, x2)
        x1 = self.upsample_add(x2, x1)
        x0 = self.upsample_add(x1, x0)

        x3 = self.post_conv3(x3)
        x3 = F.interpolate(x3, x0.size()[-2:], mode='bilinear', align_corners=False)

        x2 = self.post_conv2(x2)
        x2 = F.interpolate(x2, x0.size()[-2:], mode='bilinear', align_corners=False)

        x1 = self.post_conv1(x1)
        x1 = F.interpolate(x1, x0.size()[-2:], mode='bilinear', align_corners=False)

        x0 = self.post_conv0(x0)

        x0 = x3 * self.cat_weight[3] + x2 * self.cat_weight[2] + x1 * self.cat_weight[1]  + x0 * self.cat_weight[0]

        return x0



class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down1 = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU())
        self.conv_down2 = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x1 = self.conv_down1(x)
        x2 = self.conv_down2(x1)
        return x1, x2


class RPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rpe_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.rpe_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.rpe_norm(self.rpe_conv(x))

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.BatchNorm2d, rpe=True):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, 0, bias=False)
        self.rpe = rpe
        if self.rpe:
            self.proj_rpe = RPE(out_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        if self.rpe:
            x = self.proj_rpe(x)
        return x




class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )



class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )

class ConvBNSiLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, bias=False, inplace=False):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            act_layer()
        )

class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, bias=False, inplace=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x











class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        #self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNSiLU(decode_channels, decode_channels, kernel_size=1)
        self.avp_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x= x * self.avp_pool(x)

        x = x + res
        x = self.post_conv(x)
        return x


class DPSSNet(nn.Module):
    def __init__(self,
                 in_dim,
                 decoder_channels = 384,
                 in_chans=3,
                 num_classes=1000,
                 dims=[96, 192, 384, 768],
                 num_heads=[4, 8, 16, 32],
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.stem = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dims[0])  # 输出尺寸为96*128*128 dpr[sum(depths[:i]):sum(depths[:i + 1])]
        self.encoder_channels = dims
        self.local_feat = DetailPath(embed_dim=decoder_channels)
        self.DMB1 = DMB(dim=dims[0], patch=384, num_heads=num_heads[0])
        self.PM1 = PatchMerging(dim=dims[0], out_dim=dims[1], rpe=False)
        self.DMB2 = DMB(dim=dims[1], patch=192, num_heads=num_heads[1])
        self.PM2 = PatchMerging(dim=dims[1], out_dim=dims[2], rpe=True)
        self.DMB3 = DMB(dim=dims[2], patch=96, num_heads=num_heads[2])
        self.PM3 = PatchMerging(dim=dims[2], out_dim=dims[3], rpe=True)
        self.DMB4 = DMB(dim=dims[3], patch=96, num_heads=num_heads[3])
        self.LMFFM = LMFFM(encoder_channels=dims, decoder_channels=decoder_channels)
        self.head = nn.Sequential(ConvBNAct(decoder_channels, dims[0]),
                                  nn.Dropout(0.1),
                                  Conv(dims[0], num_classes, kernel_size=1))
        self.catdim = decoder_channels // 4
        self.catconv = ConvBNFReLU(decoder_channels // 4 + in_dim + decoder_channels, decoder_channels, kernel_size=1)
        self.avg_weight = nn.AdaptiveAvgPool2d(1)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def forward(self, x):
        b, c, h, w = x.size()
        local_feat1, local_feat2 = self.local_feat(x)
        global_feat = []
        feat256, x1 = self.stem(x)
        x2 = self.DMB1(x1)
        global_feat.append(x2)
        x3 = self.PM1(x2)
        x4 = self.DMB2(x3)
        global_feat.append(x4)
        x5 = self.PM2(x4)
        x6 = self.DMB3(x5)
        global_feat.append(x6)
        x7 = self.PM3(x6)
        x8 = self.DMB4(x7)
        global_feat.append(x8)
        output = self.LMFFM(x2, x4, x6, x8)
        output = local_feat2 + output
        out_weight = self.avg_weight(output)
        output = nn.UpsamplingBilinear2d(scale_factor=2)(output) * out_weight
        feat256_cat = torch.cat((feat256, local_feat1, output), dim=1)
        feat256_conv = self.catconv(feat256_cat)
        out = self.head(feat256_conv)
        output = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
        return output












if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512)
    model = DPSSNet(in_dim=64, in_chans=3, num_classes=2, dims=[96, 192, 384, 768], num_heads=[4, 8, 16, 32])
    output = model(x)
    print(output.shape)
    # if 1:
    #     from fvcore.nn import FlopCountAnalysis, parameter_count_table
    #     flops = FlopCountAnalysis(model, x)
    #     print("FLOPs: %.4f G" % (flops.total()/1e9))
    #
    #     total_paramters = 0
    #     for parameter in model.parameters():
    #         i = len(parameter.size())
    #         p = 1
    #         for j in range(i):
    #             p *= parameter.size(j)
    #         total_paramters += p
    #     print("Params: %.4f M" % (total_paramters / 1e6))
