# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BigGAN_layers as layers
from networks.utils import init_weights, _len2mask, make_one_hot

# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}


    arch[32] = {'in_channels': [ch * item for item in [4, 2, 1]],
                'out_channels': [ch * item for item in [2, 1, 1]],
                'upsample': [(2,1), (2,2), (2,2)],
                'resolution': [8, 16, 16],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                    for i in range(3, 6)},
                }

    return arch


class Generator(nn.Module):
    def __init__(self, G_ch=64, style_dim=128, bottom_width=4, bottom_height=4, resolution=128,
                 G_kernel_size=3, G_attn='64', n_class=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_shared=True, shared_dim=0, no_hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 BN_eps=1e-5, SN_eps=1e-12, G_fp16=False,
                 init='ortho', G_param='SN', norm_style='bn', bn_linear='embed', input_nc=3,
                 one_hot=False, first_layer=False, one_hot_k=1):
        super(Generator, self).__init__()
        dim_z = style_dim
        self.name = 'G'
        # Use class only in first layer
        self.first_layer = first_layer
        # Use one hot vector representation for input class
        self.one_hot = one_hot
        # Use one hot k vector representation for input class if k is larger than 0. If it's 0, simly use the class number and not a k-hot encoding.
        self.one_hot_k = one_hot_k
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial width dimensions
        self.bottom_width = bottom_width
        # The initial height dimension
        self.bottom_height = bottom_height
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_class
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = not no_hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]
        self.bn_linear = bn_linear

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        if one_hot:
            self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                         eps=self.SN_eps)
        else:
            self.which_embedding = nn.Embedding

        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                     else self.which_embedding)
        if self.bn_linear=='SN':
            bn_linear = functools.partial(self.which_linear, bias=False)
        if self.G_shared:
            input_size = self.shared_dim + self.z_chunk_size
        elif self.hier:
            if self.first_layer:
                input_size = self.z_chunk_size
            else:
                input_size = self.n_classes + self.z_chunk_size
            self.which_bn = functools.partial(layers.ccbn,
                                              which_linear=bn_linear,
                                              cross_replica=self.cross_replica,
                                              mybn=self.mybn,
                                              input_size=input_size,
                                              norm_style=self.norm_style,
                                              eps=self.BN_eps)
        else:
            input_size = self.n_classes
            self.which_bn = functools.partial(layers.bn,
                                              cross_replica=self.cross_replica,
                                              mybn=self.mybn,
                                              eps=self.BN_eps)

        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(self.n_classes, self.shared_dim) if G_shared
                       else layers.identity())
        # First linear layer
        # The parameters for the first linear layer depend on the different input variations.
        if self.first_layer:
            # print('one_hot:{} one_hot_k:{}'.format(self.one_hot, self.one_hot_k) )
            if self.one_hot:
                self.linear = self.which_linear(self.dim_z // self.num_slots + self.n_classes,
                                        self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
            else:
                self.linear = self.which_linear(self.dim_z // self.num_slots + 1,
                                                self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
            if self.one_hot_k==1:
                self.linear = self.which_linear((self.dim_z // self.num_slots) * self.n_classes,
                                        self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
            if self.one_hot_k>1:
                self.linear = self.which_linear(self.dim_z // self.num_slots + self.n_classes*self.one_hot_k,
                                        self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
            if self.one_hot_k == 0:
                self.linear = self.which_linear(self.n_classes,
                                                self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))

        else:
            self.linear = self.which_linear(self.dim_z // self.num_slots,
                                            self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            if 'kernel1' in self.arch.keys():
                padd1 = 1 if self.arch['kernel1'][index]>1 else 0
                padd2 = 1 if self.arch['kernel2'][index]>1 else 0
                conv1 = functools.partial(layers.SNConv2d,
                                                kernel_size=self.arch['kernel1'][index], padding=padd1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
                conv2 = functools.partial(layers.SNConv2d,
                                                kernel_size=self.arch['kernel2'][index], padding=padd2,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
                self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                               out_channels=self.arch['out_channels'][index],
                                               which_conv1=conv1,
                                               which_conv2=conv2,
                                               which_bn=self.which_bn,
                                               activation=self.activation,
                                               upsample=(functools.partial(F.interpolate,
                                                                           scale_factor=self.arch['upsample'][index])
                                                         if index < len(self.arch['upsample']) else None))]]
            else:
                self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                               out_channels=self.arch['out_channels'][index],
                                               which_conv1=self.which_conv,
                                               which_conv2=self.which_conv,
                                               which_bn=self.which_bn,
                                               activation=self.activation,
                                               upsample=(functools.partial(F.interpolate, scale_factor=self.arch['upsample'][index])
                                                         if index < len(self.arch['upsample']) else None))]]

            # If attention on this block, attach it to the end
            # print('index ', index, self.arch['resolution'][index])
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], input_nc))

        # Initialize weights. Optionally skip init for testing.
        if self.init != 'none':
            self = init_weights(self, self.init)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y, y_lens):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            if len(y.shape)<2:
                y = y.unsqueeze(1)
            if self.first_layer:
                ys = zs[1:]
            else:
                ys = [torch.cat([y.type(torch.float32), item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)
        # This is the change we made to the Big-GAN generator architecture.
        # The input goes into classes go into the first layer only.
        if self.first_layer:
            if self.one_hot:
                # y = F.one_hot(y, self.n_classes).float().to(y.device)
                y = make_one_hot(y, y_lens, self.n_classes).float().to(y.device)

            # Each characters filter is modulated by the noise vector
            if self.one_hot_k==1:
                z = z.unsqueeze(1).repeat(1, y.shape[1], y.shape[2]) * torch.repeat_interleave(y, z.shape[1], 2)
                # print('z.shape ', z.shape)
                # if self.training:
                #     z = z.unsqueeze(1).repeat(1, y.shape[1], y.shape[2]) * torch.repeat_interleave(y, z.shape[1], 2)
                # else:
                #     z = torch.randn(z.shape[0], y.shape[1], z.shape[1]).repeat(1, 1, y.shape[2]).to(z.device) * \
                #         torch.repeat_interleave(y, z.shape[1], 2)
            # Each character's filter is a one-hot k (for N char alphabet -
            # the entire vector is N*k long and the k values in the specific character location are equal to 1.
            # The filters are concatenated to the noise vector.
            elif self.one_hot_k>1:
                y = torch.repeat_interleave(y, self.one_hot_k, 2)
                z = torch.cat((z.unsqueeze(1).repeat(1, y.shape[1], 1), y), 2)
            elif self.one_hot_k == 0:
                z = y
            # only the noise vector is used as an input
            else:
                z = torch.cat((z.unsqueeze(1).repeat(1, y.shape[1], 1), y), 2)

        # First linear layer
        # print('self.linear', self.linear)
        # print('z', z.abs().mean([1, 2]))
        # print('z00', z[0, 0].cpu().numpy().tolist())
        h = self.linear(z)
        # print('h.shape', h.shape)
        # print('h.shape', h.abs().mean([1, 2]))
        # Reshape - when y is not a single class value but rather an array of classes, the reshape is needed to create
        # a separate vertical patch for each input.
        if self.first_layer:
            # correct reshape
            h = h.view(h.size(0), h.shape[1] * self.bottom_width, self.bottom_height, -1)
            h = h.permute(0, 3, 2, 1)

        else:
            h = h.view(h.size(0), -1, self.bottom_width, self.bottom_height)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        output = torch.tanh(self.output_layer(h))

        # Mask blanks
        if not self.training:
            out_lens = y_lens * output.size(-2) // 2
            mask = _len2mask(out_lens.int(), output.size(-1), torch.float32).to(z.device).detach()
            mask = mask.unsqueeze(1).unsqueeze(1)
            output = output * mask + (mask - 1)

        return output



# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64', input_nc=3, ksize='333333', dilation='111111'):
    arch = {}
    arch[32] = {'in_channels': [input_nc] + [ch * item for item in [1, 4, 8]],
                 'out_channels': [item * ch for item in [1, 4, 8, 8]],
                 'downsample': [True] * 3 + [False],
                 'resolution': [8, 4, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}
    return arch


class Discriminator(nn.Module):
    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                 D_kernel_size=3, D_attn='64', n_class=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-12, output_dim=1, D_fp16=False,
                 init='ortho', D_param='SN', bn_linear='embed', input_nc=3, one_hot=False):
        super(Discriminator, self).__init__()
        self.name = 'D'
        # one_hot representation
        self.one_hot = one_hot
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_class
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention, input_nc)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
            if bn_linear=='SN':
                self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            # We use a non-spectral-normed embedding here regardless;
            # For some reason applying SN to G's embedding seems to randomly cripple G
            self.which_embedding = nn.Embedding
        if one_hot:
            self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if self.init != 'none':
            self = init_weights(self, self.init)

    def forward(self, x, x_lens=None, y_lens=None, **kwargs):
        # print('Discriminator y:', y)
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        if x_lens is None:
            h = torch.sum(self.activation(h), [2, 3])
        else:
            h = self.activation(h)
            h_lens = x_lens * h.size(-1) // (x.size(-1) + 1e-8)
            mask = _len2mask(h_lens.int(), h.size(-1), torch.float32).to(x.device).detach()
            mask = mask.unsqueeze(1).unsqueeze(1)
            # print('h:{} mask:{}'.format(h.size(), mask.size()))
            # print(h_lens.data)
            h = torch.sum(h * mask, [2, 3])
            h = h / y_lens.unsqueeze(dim=-1)

        # Get initial class-unconditional output
        out = self.linear(h)

        return out

    def get_shared_features(self, x):
        # print('Discriminator y:', y)
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        return h
