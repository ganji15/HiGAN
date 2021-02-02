import torch
from torch import nn
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM
from networks.utils import _len2mask, init_weights


class StyleEncoder(nn.Module):
    # style_dim: 96  resolution: 32  max_dim: 512  in_channel: 1  init: 'N02'  SN_param: false
    def __init__(self, style_dim=32, resolution=16, max_dim=256, in_channel=1, init='N02',
                 SN_param=False, norm='none', share_wid=True):

        super(StyleEncoder, self).__init__()
        self.reduce_len_scale = 16
        self.share_wid = share_wid
        self.style_dim = style_dim

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 1, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', norm, sn=SN_param)]
            cnn_f += [nn.ReflectionPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', norm, sn=SN_param)]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(1):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            df = min([df_out, max_dim])

        df_out = min([int(df * 2), max_dim])
        cnn_f += [ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param)]
        cnn_f += [ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param)]
        self.cnn_backbone = nn.Sequential(*cnn_f)
        # df_out = max_dim
        # df = max_dim // 2
        ######################################
        # Construct StyleEncoder
        ######################################
        cnn_e = [nn.ReflectionPad2d((1, 1, 0, 0)),
                 Conv2dBlock(df_out, df, 3, 2, 0,
                             norm=norm,
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_wid = nn.Sequential(*cnn_e)
        self.linear_style = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU()
        )
        self.mu = nn.Linear(df, style_dim)
        self.logvar = nn.Linear(df, style_dim)

        if init != 'none':
            init_weights(self, init)

        torch.nn.init.constant_(self.logvar.weight.data, 0.)
        torch.nn.init.constant_(self.logvar.bias.data, -10.)

    def forward(self, img, img_len, wid_cnn_backbone=None, vae_mode=False):
        if self.share_wid:
            feat = wid_cnn_backbone(img)
        else:
            feat = self.cnn_backbone(img)

        img_len = img_len // self.reduce_len_scale
        out_e = self.cnn_wid(feat).squeeze(-2)
        img_len_mask = _len2mask(img_len, out_e.size(-1)).unsqueeze(1).float().detach()
        assert img_len.min() > 0, img_len.cpu().numpy()
        style = (out_e * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        style = self.linear_style(style)
        mu = self.mu(style)
        if vae_mode:
            logvar = self.logvar(style)
            encode_z = self.sample(mu, logvar)
            return encode_z, mu, logvar
        else:
            return mu

    @staticmethod
    def sample(mu, logvar):
        std = torch.exp(0.5 * logvar)
        rand_z_score = torch.randn_like(std)
        return mu + rand_z_score * std


class WriterIdentifier(nn.Module):
    def __init__(self, n_writer=284, resolution=16, max_dim=256, in_channel=1, init='N02',
                 SN_param=False, dropout=0.0, norm='bn'):

        super(WriterIdentifier, self).__init__()
        self.reduce_len_scale = 16

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 1, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2)]
            cnn_f += [nn.ReflectionPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2)]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(1):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param, dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param, dropout=dropout)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            df = min([df_out, max_dim])

        df_out = min([int(df * 2), max_dim])
        cnn_f += [ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2)]
        cnn_f += [ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2)]
        self.cnn_backbone = nn.Sequential(*cnn_f)

        ######################################
        # Construct WriterIdentifier
        ######################################
        cnn_w = [nn.ReflectionPad2d((1, 1, 0, 0)),
                 Conv2dBlock(df_out, df, 3, 2, 0,
                             norm=norm,
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_wid = nn.Sequential(*cnn_w)
        self.linear_wid = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU(),
            nn.Linear(df, n_writer),
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len):
        feat = self.cnn_backbone(img)
        img_len = img_len // self.reduce_len_scale
        out_w = self.cnn_wid(feat).squeeze(-2)
        img_len_mask = _len2mask(img_len, out_w.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (out_w * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        wid_logits = self.linear_wid(wid_feat)
        return wid_logits



class Recognizer(nn.Module):
    # resolution: 32  max_dim: 512  in_channel: 1  norm: 'none'  init: 'N02'  dropout: 0.  n_class: 72  rnn_depth: 0
    def __init__(self, n_class, resolution=16, max_dim=256, in_channel=1, norm='none',
                 init='none', rnn_depth=1, dropout=0.0, bidirectional=True):
        super(Recognizer, self).__init__()
        self.len_scale = 8
        self.use_rnn = rnn_depth > 0
        self.bidirectional = bidirectional

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 1, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])

        ######################################
        # Construct Classifier
        ######################################
        cnn_c = [nn.ReLU(),
                 Conv2dBlock(df, df, 3, 1, 0,
                             norm=norm,
                             activation='relu')]

        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.cnn_ctc = nn.Sequential(*cnn_c)
        if self.use_rnn:
            if bidirectional:
                self.rnn_ctc = DeepBLSTM(df, df, rnn_depth, bidirectional=True)
            else:
                self.rnn_ctc = DeepLSTM(df, df, rnn_depth)
        self.ctc_cls = nn.Linear(df, n_class)

        if init != 'none':
            init_weights(self, init)

    def forward(self, x, x_len=None):
        cnn_feat = self.cnn_backbone(x)
        cnn_feat2 = self.cnn_ctc(cnn_feat)
        ctc_feat = cnn_feat2.squeeze(-2).transpose(1, 2)
        if self.use_rnn:
            if self.bidirectional:
                ctc_len = x_len // (self.len_scale  + 1e-8)
            else:
                ctc_len = None
            ctc_feat = self.rnn_ctc(ctc_feat, ctc_len)
        logits = self.ctc_cls(ctc_feat)
        if self.training:
            logits = logits.transpose(0, 1).log_softmax(2)
            logits.requires_grad_(True)
        return logits
