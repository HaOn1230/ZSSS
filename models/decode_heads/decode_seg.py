from ast import Gt
import numpy as np
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy

from models.decode_heads.utils import positional_encoding

def gaussian(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None: # not do
            output = self.norm(output)

        return outputs, attns

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size() 
        Nk = xk.size()[1]
        Nv = xv.size()[1]
        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads
class FPN(nn.Module): 
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            self.lateral_convs.append(lateral_conv)

            output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
            self.output_convs.append(output_conv)
        
        self.fusion_scale = nn.Parameter(torch.ones(1))
        nn.init.constant_(self.fusion_scale, 1.0)

    def forward(self, features):
        features = [f.contiguous() for f in features]
        lateral_feats = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
        fpn_feats = lateral_feats[:]
        
        for i in range(len(lateral_feats) - 2, -1, -1):
            up_feat = F.interpolate(
                fpn_feats[i+1],
                size=lateral_feats[i].shape[2:], 
                mode='bilinear',
                align_corners=True
            )
            mixed = lateral_feats[i] + self.fusion_scale * up_feat
            fpn_feats[i] = self.output_convs[i](mixed)

        return fpn_feats
@HEADS.register_module()
class ATMSingleHeadSeg(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=5,
            fpn_out_channels=512,
            **kwargs,
    ):
        super(ATMSingleHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)
        def create_decoder():
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            return TPN_Decoder(decoder_layer, num_layers)
        self.image_size = img_size
        self.use_stages = use_stages+1
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.unseen_idx = [idx for idx in all_idx if idx not in seen_idx]
        nhead = num_heads
        dim = embed_dims
        in_channels_list = [in_channels] * self.use_stages
        self.fpn = FPN(in_channels_list, fpn_out_channels)
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim//4,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(dim//4, 1,dtype=torch.float32),
            nn.Sigmoid()
        )  
        self.gate_weight = nn.Parameter(torch.zeros(1))    
        # Initialize cascade decoders
        self.cascade_decoders = nn.ModuleList([create_decoder() for _ in range(3)])

        if hasattr(self, 'conv_seg'):
            delattr(self, 'conv_seg')

        self.q_projs = nn.ModuleList([
            nn.Linear(dim*3, dim) for _ in range(3)])

        self.weights_s1 = nn.Parameter(gaussian(torch.tensor([2, 1, 0], device='cuda'), 0, 1))
        self.weights_s2 = nn.Parameter(gaussian(torch.tensor([2, 1, 0], device='cuda'), 0, 1))

    def init_weights(self):
        nn.init.normal_(self.channel_gate[0].weight, std=0.02)
        nn.init.constant_(self.channel_gate[0].bias, 0)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, self_training=False, st_mask=None):
        seg_logits = self.forward(inputs)

        if self_training:
            pseudo_semantic_masks = seg_logits['pred_masks'].clone().detach().sigmoid()
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(dim=1).unsqueeze(1)
            # generate pseudo labels for "transductive" setting
            gt_semantic_seg[gt_semantic_seg==-1] = pseudo_semantic_seg[gt_semantic_seg==-1]
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)
        else:
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, self_training):
        return self.forward(inputs, self_training)

    def decode(self, q, decoder, laterals, decoder_output):
        _, attn_ = decoder(q, laterals.transpose(0, 1))
        for attn in attn_:
            attn = attn.transpose(-1, -2)
            attn = self.d3_to_d4(attn)
            decoder_output.append(attn)
        return decoder_output

    def forward(self, inputs_both, self_training=None):

        inputs = inputs_both[0][0]
        cls_token = inputs_both[0][1]
        text_token = inputs_both[1]
        laterals = self.fpn(inputs[:self.use_stages])
        laterals = [self.d4_to_d3(stage) for stage in laterals]

        q_s1, q_s2, q_s3 = [self.q_projs[i](self.get_qs(text_token, cls_token)).transpose(0, 1) for i in range(3)]
  
        s1, s2 = laterals[:3], laterals[3:6]
        
        decoder_s1 = self.decode(q_s1, self.cascade_decoders[0], sum(w * s for w, s in zip(self.weights_s1, s1)), [])
        decoder_s2 = self.decode(q_s2, self.cascade_decoders[1], sum(w * s for w, s in zip(self.weights_s2, s2)), [])
        decoder_s3 = self.decode(q_s3, self.cascade_decoders[2], laterals[-1], [])

        out = {}
        out["pred_masks"] = F.interpolate((decoder_s1[-1]+decoder_s2[-1]+decoder_s3[-1]),size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False)
 
        if not self.training:
            if self_training:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx)
            else:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.1)

        return out

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            # for a in zip(outputs_seg_masks[:-1])
            for a in outputs_seg_masks[:-1]
        ]

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def get_qs(self, q, cls):
        # q = [q.cls, q]
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        q = q.to(torch.float32)
        temperature = 0.07
        sim_map = torch.einsum("bd,bcd->bcd", cls, q) / temperature
        gate = torch.sigmoid(self.gate_weight)
        enhanced_q = q + gate * sim_map
        channel_gate = torch.sigmoid(self.channel_gate(q))
        selected_q = channel_gate * enhanced_q + (1-channel_gate) * q
        q1 = torch.einsum("bd,bcd->bcd", cls, q)
        gate = torch.sigmoid(self.gate_weight)
        q1 = gate * q1 + (1 - gate) * q  
        q_ = torch.cat([selected_q,q,sim_map], dim=-1)  
        return q_

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, num_classes=None):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)

            loss = self.loss_decode(
                seg_logit,
                seg_label,
                ignore_index = self.ignore_index)

            loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
            return loss