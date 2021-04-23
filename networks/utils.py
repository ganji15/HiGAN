import functools
import numpy as np
from itertools import groupby
import cv2
import torch
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler
from networks.block import AdaptiveInstanceNorm2d, Identity, AdaptiveInstanceLayerNorm2d, InstanceLayerNorm2d
from lib.alphabet import word_capitalize
from PIL import Image, ImageDraw, ImageFont


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        if (isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)
                or isinstance(m, nn.Embedding)):
            if init_type == 'N02':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type in ['glorot', 'xavier']:
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'ortho':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    if init_type in ['N02', 'glorot', 'xavier', 'kaiming', 'ortho']:
        print('initialize network {} with {}'.format(net.__class__.__name__, init_type))
        net.apply(init_func)  # apply the initialization function <init_func>
    return net


def get_norm_layer(norm='in', **kwargs):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm == 'bn':
        norm_layer = functools.partial(nn.BatchNorm2d)
    elif norm == 'gn':
        norm_layer = functools.partial(nn.GroupNorm)
    elif norm == 'in':
        norm_layer = functools.partial(nn.InstanceNorm2d)
    elif norm == 'adain':
        norm_layer = functools.partial(AdaptiveInstanceNorm2d)
    elif norm == 'iln':
        norm_layer = functools.partial(InstanceLayerNorm2d)
    elif norm == 'adailn':
        norm_layer = functools.partial(AdaptiveInstanceLayerNorm2d)
    elif norm == 'none':
        def norm_layer(x): return Identity()
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
    return norm_layer


def get_linear_scheduler(optimizer, start_decay_iter, n_iters_decay):
    def lambda_rule(iter):
        lr_l = 1.0 - max(0, iter - start_decay_iter) / float(n_iters_decay + 1)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.start_decay_epoch) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def _len2mask(length, max_len, dtype=torch.float32):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def get_init_state(deepth, batch_size, hidden_dim, device, bidirectional=False):
    """Get cell states and hidden states."""
    if bidirectional:
        deepth *= 2
        hidden_dim //= 2

    h0_encoder_bi = torch.zeros(
        deepth,
        batch_size,
        hidden_dim, requires_grad=False)
    c0_encoder_bi = torch.zeros(
        deepth,
        batch_size,
        hidden_dim, requires_grad=False)
    return h0_encoder_bi.to(device), c0_encoder_bi.to(device)


def _info(model, detail=False, ret=False):
    nParams = sum([p.nelement() for p in model.parameters()])
    mSize = nParams * 4.0 / 1024 / 1024
    res = "*%-12s  param.: %dK  Stor.: %.4fMB" % (type(model).__name__,  nParams / 1000, mSize)
    if detail:
        res += '\r\n' + str(model)
    if ret:
        return res
    else:
        print(res)


def _info_simple(model, tag=None):
    nParams = sum([p.nelement() for p in model.parameters()])
    mSize = nParams * 4.0 / 1024 / 1024
    if tag is None:
        tag = type(model).__name__
    res = "%-12s P:%6dK  S:%8.4fMB" % (tag,  nParams / 1000, mSize)
    return res


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def idx_to_words(idx, lexicon, capitize_ratio=0.5):
    words = []
    for i in idx:
        word = lexicon[i]
        if np.random.random() < capitize_ratio:
            word = word_capitalize(word)
        words.append(word)
    return words


def pil_text_img(im, text, pos, color=(255, 0, 0), textSize=25):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('font/arial.ttf', textSize)
    fillColor = color  # (255,0,0)
    position = pos  # (100,100)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def words_to_images(texts, img_h, img_w, n_channel=1):
    n_channel = 3
    word_imgs = np.zeros((len(texts), img_h, img_w, n_channel)).astype(np.uint8)
    for i in range(len(texts)):
        # cv2.putText(word_imgs[i], texts[i], (2, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
        word_imgs[i] = pil_text_img(word_imgs[i], texts[i], (1, 1),  textSize=25)
    word_imgs = word_imgs.sum(axis=-1, keepdims=True).astype(np.uint8)
    word_imgs = torch.from_numpy(word_imgs).permute([0, 3, 1, 2]).float() / 128 - 1
    return word_imgs


def ctc_greedy_decoder(probs_seq, blank_index=0):
    """CTC greedy (best path) decoder.
    Path consisting of the most probable tokens are further post-processed to
    remove consecutive repetitions and all blanks.
    :param probs_seq: 2-D list of probabilities over the vocabulary for each
                      character. Each element is a list of float probabilities
                      for one character.
    :type probs_seq: list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :return: Decoding result string.
    :rtype: baseline
    """

    # argmax to get the best index for each time step
    max_index_list = list(np.array(probs_seq).argmax(axis=1))
    # remove consecutive duplicate indexes
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # remove blank indexes
    # blank_index = len(vocabulary)
    index_list = [index for index in index_list if index != blank_index]
    # convert index list to string
    return index_list


def make_one_hot(labels, len_labels, n_class):
    one_hot = torch.zeros((labels.shape[0], labels.shape[1], n_class), dtype=torch.float32)
    for i in range(len(labels)):
        one_hot[i, np.array(range(len_labels[i])), labels[i,:len_labels[i]]-1]=1
    return one_hot


def rand_clip(imgs, img_lens, min_clip_width=64):
    device = imgs.device
    imgs, img_lens = imgs.cpu().numpy(), img_lens.cpu().numpy()
    clip_imgs, clip_img_lens = [], []
    for img, img_len in zip(imgs, img_lens):
        if img_len <= min_clip_width:
            clip_imgs.append(img[:, :, :img_len])
            clip_img_lens.append(img_len)
        else:
            crop_width = np.random.randint(min_clip_width, img_len)
            crop_width = crop_width - crop_width % (min_clip_width // 4)
            rand_pos = np.random.randint(0, img_len - crop_width)
            clip_img = img[:, :, rand_pos: rand_pos + crop_width]
            clip_imgs.append(clip_img)
            clip_img_lens.append(clip_img.shape[-1])

    max_img_len = max(clip_img_lens)
    pad_imgs = -np.ones((imgs.shape[0], 1, imgs.shape[2], max_img_len))
    for i, (clip_img, clip_img_len) in enumerate(zip(clip_imgs, clip_img_lens)):
        pad_imgs[i, 0, :, :clip_img_len] = clip_img
    return torch.from_numpy(pad_imgs).float().to(device), torch.Tensor(clip_img_lens).int().to(device)