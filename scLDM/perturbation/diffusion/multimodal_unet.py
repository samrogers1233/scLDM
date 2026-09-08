from abc import abstractmethod
import random
import math
import time
from einops import rearrange
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from . import logger
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    normalization_cell,
    timestep_embedding
)
import numpy as np
import os
import torch

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, audio, emb_pert):#
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, audio, emb_pert):#
        for layer in self:
            if isinstance(layer, TimestepBlock):
                audio = layer(audio, emb_pert)
            else:
                if isinstance(layer, SelfAttentionBlock_cell):
                    audio, audio_old, att_vec= layer(audio)
                    return audio, audio_old, att_vec
                else:
                    audio = layer(audio)
        return audio
        


class CellConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        feature_dim,
        kernel_size=3,
        stride = 1,
        padding = "same",
        dilation = 1,
        conv_type = '1d',
    ):
        super().__init__()
        
        self.cell_conv = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        
    def forward(self, audio):
        audio = self.cell_conv(audio)
        return audio


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims=dims
        if dims == 3: 
            # for video
            self.stride = (1,2,2)
        elif dims == 1:
            #for audio
            self.stride = 2
        else:
            # for image
            self.stride = 2

        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 2, padding='same')

    def forward(self, x):
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2] * self.stride[0], x.shape[3] * self.stride[1], x.shape[4] * self.stride[2]), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=self.stride, mode="nearest")

        if self.use_conv:
            x = self.conv(x)
   
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if dims == 3:
            stride = (1,2,2)
        elif dims == 1:
            stride = 2
        else:
            stride = 2

        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 2, stride=stride
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.op(x)
        return x



class SingleModalQKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

class SingleModalAtten(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        feature_dim=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        if feature_dim is not None:
            self.norm = normalization_cell(channels)
        else:
            self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = SingleModalQKVAttention(self.num_heads)
        
        self.proj_out = conv_nd(1, channels, channels, 1)

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        
        b, c, *spatial = x.shape
     
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)



class ResBlock_cell(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
        use_conv=False,
        num_heads=4,
        feature_dim=26,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.audio_in_layers = nn.Sequential(
            normalization_cell(channels),
            nn.SiLU(),
            CellConv(channels, self.out_channels, feature_dim, kernel_size=3,  conv_type='1d'),
        )
        
        self.updown = up or down
        
        self.audio_attention = False #audio_attention

        if up:
            self.ah_upd = Upsample(channels, True, 1)
            self.ax_upd = Upsample(channels, True, 1)
            feature_dim = int(feature_dim*2)
        elif down:
            self.ah_upd = Downsample(channels, True, 1)
            self.ax_upd = Downsample(channels, True, 1)
            feature_dim = int(feature_dim/2)
        else:
            self.ah_upd = self.ax_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.audio_out_layers = nn.Sequential(
            normalization_cell(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            CellConv(self.out_channels, self.out_channels, feature_dim, kernel_size=1, conv_type='1d'),
        )

        if self.out_channels == channels:
            self.audio_skip_connection = nn.Identity()
        elif use_conv:
            self.audio_skip_connection = CellConv(
                 channels, self.out_channels, feature_dim, kernel_size=3, conv_type='1d'
            )
        else:
            self.audio_skip_connection = CellConv(
                 channels, self.out_channels, feature_dim, kernel_size=1, conv_type='1d'
            )

        if self.audio_attention:
            self.audio_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, 
                num_head_channels=-1, use_checkpoint=use_checkpoint,feature_dim=feature_dim)
       

    def forward(self, audio, emb_pert):
        return checkpoint(
            self._forward, (audio, emb_pert), self.parameters(), self.use_checkpoint
        )

    def _forward(self, audio, emb_pert):
        if self.updown:
            audio_h = self.audio_in_layers(audio)
            audio_h = self.ah_upd(audio_h)
            audio = self.ax_upd(audio)

        else:
            audio_h = self.audio_in_layers(audio)
            
        audio_emb = self.emb_layers(emb_pert).type(audio.dtype)
        

        if self.use_scale_shift_norm:
            audio_out_norm, audio_out_rest = self.audio_out_layers[0], self.audio_out_layers[1:]
            audio_emb_out = audio_emb[:,None,:]
            scale, shift = th.chunk(audio_emb_out, 2, dim=-1)
            audio_h = audio_out_norm(audio_h) * (1 + scale) + shift
            audio_h = audio_out_rest(audio_h)

        else:
            audio_emb_out = audio_emb[:,None,:]
            audio_h = audio_h + audio_emb_out
            audio_h = self.audio_out_layers(audio_h)

        

        audio_out = self.audio_skip_connection(audio) + audio_h


        if self.audio_attention:
            audio_out = self.audio_attention_block(audio_out)

        return audio_out


class SelfAttentionBlock_cell(nn.Module):
    def __init__(
        self,
        channels,
  
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        local_window = 1,
        window_shift = False,
        feature_dim=24,
        id = 1,
        specific_type=None,
    ):
        super().__init__()
        self.channels = channels
        self.feature_dim = 64
        self.id = id
        self.time_step = 0
        self.specific_type=specific_type


        self.q = nn.Linear(1,self.feature_dim,bias=False)
        self.k = nn.Linear(1,self.feature_dim,bias=False)
        self.v = nn.Linear(1,self.feature_dim,bias=False)

        self.trans = nn.Linear(self.feature_dim,1,bias=False)


    def forward(self, audio): 
        
        return checkpoint(self._forward, (audio,), self.parameters(), True)
        

    def _forward(self, audio): 
        audio_old1 = audio

        audio = audio.transpose(-1,-2)

        Q = self.q(audio)
        K = self.k(audio)
        V = self.v(audio)


        att = F.softmax(th.matmul(Q, K.transpose(-1,-2))/th.tensor(self.feature_dim).sqrt().to(device=Q.device),dim=-1)
        pert_new = th.matmul(att,V)     
        audio_old2 = att
        audio = audio + self.trans(pert_new)
        audio = audio.transpose(-1,-2)       

        return audio, audio_old1, audio_old2


class InitialBlock_cell(nn.Module):
    def __init__(
        self,
        audio_in_channels,
        audio_out_channels,
        feature_dim,
        kernel_size = 3 
    ):
        super().__init__()
        self.audio_conv = CellConv(audio_in_channels, audio_out_channels,feature_dim=feature_dim, kernel_size=kernel_size, conv_type='1d')

    def forward(self, audio): 
        return self.audio_conv(audio)

class MultimodalUNet(nn.Module):
    def __init__(
        self,
        ctrl_dim,
        pert_dim,
        model_channels,
       
        video_out_channels,
        audio_out_channels,
        num_res_blocks,
        cross_attention_resolutions,
        cross_attention_windows,
        cross_attention_shift,

        dropout=0,
        channel_mult=(4, 2, 1),
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=True,
        specific_type=None,
        use_gene_cond=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.ctrl_dim = ctrl_dim.copy()    #(1,100)
        self.pert_dim = pert_dim.copy()    #(1,100)
        self.specific_type = specific_type
        self.use_gene_cond = use_gene_cond

        self.model_channels = model_channels   #128
        self.video_out_channels = video_out_channels   #100
        self.audio_out_channels = audio_out_channels    #100
        self.num_res_blocks = num_res_blocks
        self.cross_attention_resolutions = cross_attention_resolutions
        self.cross_attention_windows = cross_attention_windows
        self.cross_attention_shift = cross_attention_shift
        self.dropout = dropout
        self.channel_mult = channel_mult   # (4, 2, 1)
        self.num_classes = int(num_classes) if num_classes is not None else num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)
        ch = input_ch = int(channel_mult[0] * model_channels)   

        cond_video_dim = ctrl_dim[-1] 
        cond_audio_dim = pert_dim[-1]
        self.audio_cond_layers = nn.Sequential(
            nn.Linear(cond_audio_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        if self.use_gene_cond:
            gene_emb_dim = 200
            self.audio_gene_layers = nn.Sequential(
                nn.Linear(gene_emb_dim, model_channels),
                nn.SiLU(),
                nn.Linear(model_channels, model_channels),
            )
        
        self._feature_size = ch   #512
        input_block_chans = [ch] 
        feature_dim = self.pert_dim[-1]  #100
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(InitialBlock_cell(self.pert_dim[-1], feature_dim=feature_dim, audio_out_channels=ch))])

        len_audio_conv = 1

        bid = 1
        dilation = 1
       
        for level, mult in enumerate(channel_mult):
            for block_id in range(num_res_blocks):
                layers=[ResBlock_cell(
                        ch,
                        time_embed_dim,
                        dropout,
                        feature_dim=feature_dim,
                        out_channels=int(mult * model_channels),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_heads = num_heads,
                    )]
                
                dilation += len_audio_conv
                ch = int(mult * model_channels)

                if level == len(channel_mult)-2:              
                    layers.append(SelfAttentionBlock_cell(
                            ch, 
                            feature_dim=feature_dim,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            local_window=self.cross_attention_windows[0],
                            window_shift=self.cross_attention_shift,
                            num_head_channels=num_head_channels,  
                            id=1, 
                            specific_type=self.specific_type,
                            )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                bid += 1
                self._feature_size += ch
                input_block_chans.append(ch)
            
        self.middle_blocks = TimestepEmbedSequential(
            SelfAttentionBlock_cell(
                    ch,
                    feature_dim=feature_dim,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    local_window = self.pert_dim[0],
                    window_shift = False,
                    id=2,
                    specific_type=self.specific_type,
                ),
            ResBlock_cell(
                    ch,
                    time_embed_dim,
                    dropout,
                    feature_dim=feature_dim,
                    use_checkpoint=use_checkpoint,   
                    use_scale_shift_norm=use_scale_shift_norm,
                    num_heads=num_heads,
            )
        )           
       
        self._feature_size += ch
        bid=0
        self.output_blocks = nn.ModuleList([])
        dilation -= len_audio_conv

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_id in range(num_res_blocks):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock_cell(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        feature_dim=feature_dim,
                        out_channels=int(model_channels * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_heads=num_heads 
                    )
                ]

                dilation -= len_audio_conv
                ch = int(model_channels * mult)
                if level == len(channel_mult)-2:   
                    layers.append(SelfAttentionBlock_cell(
                            ch,
                            feature_dim=feature_dim,
                            use_checkpoint=use_checkpoint,
                            local_window=self.cross_attention_windows[0],
                            window_shift=self.cross_attention_shift, 
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            id=3,
                            specific_type=self.specific_type,
                    ))
                
                bid += 1
                self._feature_size += ch
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        
        self.audio_out = nn.Sequential(
            normalization_cell(input_ch),
            nn.SiLU(),
            CellConv(input_ch, audio_out_channels, feature_dim, kernel_size=3, conv_type='1d'),
        )
        
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.audio_out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.audio_out.apply(convert_module_to_f32)

    def load_state_dict_(self, state_dict, is_strict=False):
              
        for key, val in self.state_dict().items():
            
            if key in state_dict.keys():
                if val.shape == state_dict[key].shape:
                    continue
                else:
                    state_dict.pop(key)
                    logger.log("{} not matchable with state_dict with shape {}".format(key, val.shape))
            else:
                
                logger.log("{} not exists in state_dict".format(key))

        for key, val in state_dict.items():
            if key in self.state_dict().keys():
                if val.shape == state_dict[key].shape:
                    continue  
            else:
                logger.log("{} not used in state_dict".format(key))
        self.load_state_dict(state_dict, strict=is_strict)
        return 
    
   

    def forward(self, audio,  timesteps,  label=None, audio_cond=None, audio_gene_cond=None, return_attvec=False, cos_sim=None):
        att_vectors = []
        audio_hs = []
        base_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) # 
    
        emb_pert = base_emb.clone() 

        #if self.num_classes is not None:
        if self.num_classes is not None and label is not None:
            assert label.shape[0] == (audio.shape[0])
            emb_pert = emb_pert + self.label_emb(label)

        if audio_cond is not None:
            assert audio_cond.shape[0] == (audio.shape[0])
            a = audio_cond.squeeze(1) if audio_cond.dim() == 3 else audio_cond
            emb_pert = emb_pert + self.audio_cond_layers(a)   
        
        if self.use_gene_cond and audio_gene_cond is not None:
            assert audio_gene_cond.shape[0] == (audio.shape[0])
            g = audio_gene_cond.squeeze(1) if audio_gene_cond.dim() == 3 else audio_gene_cond
            emb_pert = emb_pert + self.audio_gene_layers(g)

        audio = audio.type(self.dtype)
        emb_pert = emb_pert.type(self.dtype)


        for m_id, module in enumerate(self.input_blocks):# 
            if isinstance(module[-1], SelfAttentionBlock_cell):
                module[-1].specific_type = self.specific_type
                audio, audio_old, att_vector = module(audio, emb_pert)
                att_vectors.append(audio_old)
                att_vectors.append(att_vector)
                
            else:
                audio = module(audio, emb_pert)#
            audio_hs.append(audio)

        self.middle_blocks[0].specific_type = self.specific_type
        audio, audio_old, att_vector = self.middle_blocks(audio, emb_pert)
        att_vectors.append(audio_old)
        att_vectors.append(att_vector)

        for m_id, module in enumerate(self.output_blocks):
            audio = th.cat([audio, audio_hs.pop()], dim=-1)
            if isinstance(module[-1], SelfAttentionBlock_cell):
                module[-1].specific_type = self.specific_type
                audio, audio_old, att_vector = module(audio, emb_pert)
                att_vectors.append(audio_old)
                att_vectors.append(att_vector)
            else:
                audio = module(audio, emb_pert)#
       

        audio = self.audio_out(audio)
    
        if return_attvec:
            return audio, att_vectors
        
        return audio



if __name__=='__main__':
    import time
    device = th.device("cuda:7")
    

    model_channels = 192
    emb_channels = 128
    rna_dim= [16,3,64,64]
    atac_dim = [1, 25600]
    video_out_channels = 3
    audio_out_channels = 1
    num_heads = 2
    num_res_blocks = 1
    cross_attention_resolutions = [4,8,16]
    cross_attention_window = [1,1,1]
    cross_attention_shift = False
    video_attention_resolutions = [2,4,8,16]
    audio_attention_resolutions = [2,4,8,16]
    lr=0.0001
    channel_mult=(1,2,3,4)
    model = MultimodalUNet(
        rna_dim,
        atac_dim,
        model_channels,
        video_out_channels,
        audio_out_channels,
        num_res_blocks,
        cross_attention_resolutions = cross_attention_resolutions,
        num_heads = num_heads,
        cross_attention_windows = cross_attention_window,
        cross_attention_shift = cross_attention_shift,
        video_attention_resolutions = video_attention_resolutions,
        audio_attention_resolutions = audio_attention_resolutions,
        use_scale_shift_norm=True,
        use_checkpoint=True
        
    ).to(device)
    
    optim = th.optim.SGD(model.parameters(),lr=lr)
    model.train()
    while True:
        time_start=time.time()
        video = th.randn([1, 16, 3, 64, 64]).to(device)
        audio = th.randn([1, 1, 25600]).to(device)
        time_index = th.tensor([1]).to(device)

        video_out, audio_out = model(video, audio, time_index)
        video_target = th.randn_like(video_out)
        audio_target = th.randn_like(audio_out)
        loss =  F.mse_loss(video_target, video_out)+F.mse_loss(audio_target, audio_out)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"loss:{loss} time:{time.time()-time_start}")
  