import torch, os
from PIL import Image
from munch import Munch
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.nn import CTCLoss, CrossEntropyLoss
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter

from fid_kid.fid_kid_pad import calculate_kid_fid
from networks.utils import _info, set_requires_grad, get_scheduler, idx_to_words, words_to_images
from networks.BigGAN_networks import Generator, Discriminator
from networks.module import Recognizer, WriterIdentifier, StyleEncoder
from lib.datasets import get_dataset, get_collect_fn, Hdf5Dataset
from lib.alphabet import strLabelConverter, get_lexicon, get_true_alphabet, Alphabets
from lib.utils import draw_image, get_logger, AverageMeterManager, option_to_string
from networks.rand_dist import prepare_z_dist, prepare_y_dist


class BaseModel(object):
    def __init__(self, opt, log_root='./'):
        self.opt = opt
        self.device = torch.device(opt.device)
        self.models = Munch()
        self.models_ema = Munch()
        self.log_root = log_root
        self.logger = None
        self.writer = None
        alphabet_key = 'rimes_word' if opt.dataset.startswith('rimes') else 'all'
        self.alphabet = Alphabets[alphabet_key]
        self.label_converter = strLabelConverter(alphabet_key)
        self.collect_fn = get_collect_fn(opt.training.sort_input)

    def print(self, info):
        if self.logger is None:
            print(info)
        else:
            self.logger.info(info)

    def create_logger(self):
        if self.logger or self.writer:
            return

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        self.writer = SummaryWriter(log_dir=self.log_root)
        opt_str = option_to_string(self.opt)
        with open(os.path.join(self.log_root, 'config.txt'), 'w') as f:
            f.writelines(opt_str)
        self.logger = get_logger(self.log_root)

    def info(self, extra=None):
        self.print("RUNDIR: {}".format(self.log_root))
        opt_str = option_to_string(self.opt)
        self.print(opt_str)
        for model in self.models.values():
            self.print(_info(model, ret=True))
        if extra is not None:
            self.print(extra)
        self.print('=' * 20)

    def save(self, tag='best', epoch_done=0, **kwargs):
        ckpt = {}
        if len(self.models_ema.values()) == 0:
            for model in self.models.values():
                ckpt[type(model).__name__] = model.state_dict()
        else:
            for model in self.models_ema.values():
                ckpt[type(model).__name__] = model.state_dict()

        for key, val in kwargs.items():
            ckpt[key] = val

        ckpt['Epoch'] = epoch_done
        ckpt_save_path = os.path.join(self.log_root, self.opt.training.ckpt_dir, tag + '.pth')
        torch.save(ckpt, ckpt_save_path)

    def load(self, ckpt, map_location=None, modules=None):
        if modules is None:
            modules = []
        elif not isinstance(modules, list):
            modules = [modules]

        print('load checkpoint from ', ckpt)
        if map_location is None:
            ckpt = torch.load(ckpt)
        else:
            ckpt = torch.load(ckpt, map_location=map_location)

        if len(modules) == 0:
            for model in self.models.values():
                model.load_state_dict(ckpt[type(model).__name__])
        else:
            for model in modules:
                model.load_state_dict(ckpt[type(model).__name__])

    def set_mode(self, mode='eval'):
        for model in self.models.values():
            if mode == 'eval':
                model.eval()
            elif mode == 'train':
                model.train()
            else:
                raise NotImplementedError()

    def validate(self):
        yield NotImplementedError()

    def train(self):
        yield NotImplementedError()


class AdversarialModel(BaseModel):
    def __init__(self, opt, log_root='./'):
        super(AdversarialModel, self).__init__(opt, log_root)

        device = self.device
        self.lexicon = get_lexicon(self.opt.training.lexicon,
                                   get_true_alphabet(opt.dataset),
                                   max_length=self.opt.training.max_word_len)
        self.max_valid_image_width = self.opt.char_width * self.opt.training.max_word_len
        self.noise_dim = self.opt.GenModel.style_dim - self.opt.EncModel.style_dim

        generator = Generator(**opt.GenModel).to(device)
        style_encoder = StyleEncoder(**opt.EncModel).to(device)
        writer_identifier = WriterIdentifier(**opt.WidModel).to(device)
        discriminator = Discriminator(**opt.DiscModel).to(device)
        recognizer = Recognizer(**opt.OcrModel).to(device)
        self.models = Munch(
            G=generator,
            D=discriminator,
            R=recognizer,
            E=style_encoder,
            W=writer_identifier
        )

        self.ctc_loss = CTCLoss(zero_infinity=True, reduction='mean')
        self.classify_loss = CrossEntropyLoss()

    def train(self):
        self.info()

        def KLloss(mu, logvar):
            return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        opt = self.opt
        self.z = prepare_z_dist(opt.training.batch_size, opt.GenModel.style_dim, self.device,
                                seed=self.opt.seed)
        self.y = prepare_y_dist(opt.training.batch_size, len(self.lexicon), self.device, seed=self.opt.seed)

        self.eval_z = prepare_z_dist(opt.training.eval_batch_size, opt.GenModel.style_dim, self.device,
                                     seed=self.opt.seed)
        self.eval_y = prepare_y_dist(opt.training.eval_batch_size, len(self.lexicon), self.device,
                                     seed=self.opt.seed)

        self.train_loader = DataLoader(
            get_dataset(opt.dataset, opt.training.dset_split),
            batch_size=opt.training.batch_size,
            shuffle=True,
            collate_fn=self.collect_fn,
            num_workers=4,
            drop_last=True
        )

        self.tst_loader = DataLoader(
            get_dataset(opt.dataset, opt.valid.dset_split),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn
        )

        self.tst_loader2 = DataLoader(
            get_dataset(opt.dataset, opt.training.dset_split),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn,
            num_workers=4
        )

        self.optimizers = Munch(
            G=torch.optim.Adam(chain(self.models.G.parameters(), self.models.E.parameters()),
                               lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2)),
            D=torch.optim.Adam(
                chain(self.models.D.parameters(), self.models.R.parameters(), self.models.W.parameters()),
                lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2)),
        )

        self.lr_schedulers = Munch(
            G=get_scheduler(self.optimizers.G, opt.training),
            D=get_scheduler(self.optimizers.D, opt.training)
        )

        self.averager_meters = AverageMeterManager(['adv_loss', 'fake_disc_loss',
                                                    'real_disc_loss', 'info_loss',
                                                    'fake_ctc_loss', 'real_ctc_loss',
                                                    'fake_wid_loss', 'real_wid_loss',
                                                    'kl_loss', 'gp_ctc', 'gp_info', 'gp_wid'])
        device = self.device

        ctc_len_scale = 8
        best_kid = np.inf
        iter_count = 0
        for epoch in range(1, self.opt.training.epochs):
            for i, (imgs, img_lens, lbs, lb_lens, wids) in enumerate(self.train_loader):
                #############################
                # Prepare inputs & Network Forward
                #############################
                self.set_mode('train')
                real_imgs, real_img_lens, real_wids = imgs.to(device), img_lens.to(device), wids.to(device)
                real_lbs, real_lb_lens = lbs.to(device), lb_lens.to(device)

                #############################
                # Optimizing Recognizer & Writer Identifier & Discriminator
                #############################
                self.optimizers.D.zero_grad()
                set_requires_grad([self.models.G, self.models.E], False)
                set_requires_grad([self.models.R, self.models.D, self.models.W], True)

                ### Compute CTC loss for real samples###
                real_ctc = self.models.R(real_imgs)
                real_ctc_lens = real_img_lens // ctc_len_scale
                real_ctc_loss = self.ctc_loss(real_ctc, real_lbs, real_ctc_lens, real_lb_lens)
                self.averager_meters.update('real_ctc_loss', real_ctc_loss.item())

                real_wid_logits = self.models.W(real_imgs, real_img_lens)
                real_wid_loss = self.classify_loss(real_wid_logits, real_wids)
                self.averager_meters.update('real_wid_loss', real_wid_loss.item())

                with torch.no_grad():
                    self.y.sample_()
                    sampled_words = idx_to_words(self.y, self.lexicon, self.opt.training.capitalize_ratio)
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                    fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()

                    self.z.sample_()
                    fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                    enc_styles, _, _ = self.models.E(real_imgs, real_img_lens,
                                                     self.models.W.cnn_backbone, vae_mode=True)
                    noises = torch.randn((real_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                    style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)

                    cat_fake_imgs = torch.cat([fake_imgs, style_imgs], dim=0)
                    cat_fake_lb_lens = fake_lb_lens.repeat(2,).detach()
                    cat_fake_img_lens = cat_fake_lb_lens * self.opt.char_width

                ### Compute discriminative loss for real & fake samples ###
                fake_disc = self.models.D(cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens)
                fake_disc_loss = torch.mean(F.relu(1.0 + fake_disc))

                real_disc= self.models.D(real_imgs, real_img_lens, real_lb_lens)
                real_disc_loss = torch.mean(F.relu(1.0 - real_disc))

                disc_loss = real_disc_loss + fake_disc_loss
                self.averager_meters.update('real_disc_loss', real_disc_loss.item())
                self.averager_meters.update('fake_disc_loss', fake_disc_loss.item())

                (real_ctc_loss + disc_loss + real_wid_loss).backward()
                self.optimizers.D.step()

                #############################
                # Optimizing Generator
                #############################
                if iter_count % self.opt.training.num_critic_train == 0:
                    self.optimizers.G.zero_grad()
                    set_requires_grad([self.models.D, self.models.R, self.models.W], False)
                    set_requires_grad([self.models.G, self.models.E], True)

                    ##########################
                    # Prepare Fake Inputs
                    ##########################
                    self.y.sample_()
                    sampled_words = idx_to_words(self.y, self.lexicon, self.opt.training.capitalize_ratio)
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                    fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()
                    fake_img_lens = fake_lb_lens * self.opt.char_width

                    self.z.sample_()
                    fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                    enc_styles, enc_mu, enc_logvar = self.models.E(real_imgs, real_img_lens,
                                                                   self.models.W.cnn_backbone, vae_mode=True)
                    noises = torch.randn((real_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                    style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)
                    style_img_lens = fake_lb_lens * self.opt.char_width

                    ### Concatenating all generated images in a batch ###
                    cat_fake_imgs = torch.cat([fake_imgs, style_imgs], dim=0)
                    cat_fake_lbs = fake_lbs.repeat(2, 1).detach()
                    cat_fake_lb_lens = fake_lb_lens.repeat(2,).detach()
                    cat_fake_img_lens = cat_fake_lb_lens * self.opt.char_width

                    ###################################################
                    # Calculating G Losses
                    ####################################################
                    ### deal with fake samples ###
                    ### Compute Adversarial loss ###
                    cat_fake_disc = self.models.D(cat_fake_imgs, cat_fake_img_lens, cat_fake_lb_lens)
                    adv_loss = -torch.mean(cat_fake_disc)

                    ### CTC Auxiliary loss ###
                    cat_fake_ctc = self.models.R(cat_fake_imgs)
                    cat_fake_ctc_lens = cat_fake_img_lens // ctc_len_scale
                    fake_ctc_loss = self.ctc_loss(cat_fake_ctc, cat_fake_lbs,
                                                  cat_fake_ctc_lens, cat_fake_lb_lens)

                    ### Latent Style Reconstruction ###
                    styles = self.models.E(fake_imgs, fake_img_lens, self.models.W.cnn_backbone)
                    info_loss = torch.mean(torch.abs(styles - self.z[:, -self.opt.EncModel.style_dim:].detach()))

                    ### Writer Identify Loss ###
                    recn_wid_logits = self.models.W(style_imgs, style_img_lens)
                    fake_wid_loss = self.classify_loss(recn_wid_logits, real_wids)

                    ### KL-Divergence Loss ###
                    kl_loss = KLloss(enc_mu, enc_logvar)

                    ### Gradient balance ###
                    grad_fake_adv = torch.autograd.grad(adv_loss, cat_fake_imgs, create_graph=True, retain_graph=True)[0]
                    grad_fake_OCR = torch.autograd.grad(fake_ctc_loss, cat_fake_ctc, create_graph=True, retain_graph=True)[0]
                    grad_fake_info = torch.autograd.grad(info_loss, fake_imgs, create_graph=True, retain_graph=True)[0]
                    grad_fake_wid = torch.autograd.grad(fake_wid_loss, recn_wid_logits, create_graph=True, retain_graph=True)[0]

                    std_grad_adv = torch.std(grad_fake_adv)
                    gp_ctc = torch.div(std_grad_adv, torch.std(grad_fake_OCR) + 1e-8).detach() + 1
                    gp_info = torch.div(std_grad_adv, torch.std(grad_fake_info) + 1e-8).detach() + 1
                    gp_wid = torch.div(std_grad_adv, torch.std(grad_fake_wid) + 1e-8).detach() + 1
                    self.averager_meters.update('gp_ctc', gp_ctc.item())
                    self.averager_meters.update('gp_info', gp_info.item())
                    self.averager_meters.update('gp_wid', gp_wid.item())

                    g_loss = 2 * adv_loss + \
                             gp_ctc * fake_ctc_loss + \
                             gp_info * info_loss + \
                             gp_wid * fake_wid_loss + \
                             self.opt.training.lambda_kl * kl_loss
                    g_loss.backward()
                    self.averager_meters.update('adv_loss', adv_loss.item())
                    self.averager_meters.update('fake_ctc_loss', fake_ctc_loss.item())
                    self.averager_meters.update('info_loss', info_loss.item())
                    self.averager_meters.update('fake_wid_loss', fake_wid_loss.item())
                    self.averager_meters.update('kl_loss', kl_loss.item())
                    self.optimizers.G.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    meter_vals = self.averager_meters.eval_all()
                    self.averager_meters.reset_all()
                    info = "[%3d|%3d]-[%4d|%4d] G:%.4f D-fake:%.4f D-real:%.4f " \
                           "CTC-fake:%.4f CTC-real:%.4f Wid-fake:%.4f Wid-real:%.4f " \
                           "Recn-z:%.4f Kl:%.4f" \
                           % (epoch, self.opt.training.epochs,
                              iter_count % len(self.train_loader), len(self.train_loader),
                              meter_vals['adv_loss'],
                              meter_vals['fake_disc_loss'], meter_vals['real_disc_loss'],
                              meter_vals['fake_ctc_loss'], meter_vals['real_ctc_loss'],
                              meter_vals['fake_wid_loss'], meter_vals['real_wid_loss'],
                              meter_vals['info_loss'], meter_vals['kl_loss'])
                    self.print(info)

                    if self.writer:
                        for key, val in meter_vals.items():
                            self.writer.add_scalar('loss/%s' % key, val, iter_count + 1)

                if (iter_count + 1) % self.opt.training.sample_iter_val == 0:
                    if not (self.logger and self.writer):
                        self.create_logger()

                    sample_root = os.path.join(self.log_root, self.opt.training.sample_dir)
                    if not os.path.exists(sample_root):
                        os.makedirs(sample_root)
                    self.sample_images(iter_count + 1)

                iter_count += 1

            if epoch:
                ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root)

                self.save('last', epoch)
                if epoch >= self.opt.training.start_save_epoch_val and \
                        epoch % self.opt.training.save_epoch_val == 0:
                    self.print('Calculate FID_KID')
                    scores = self.validate()
                    fid, kid = scores['FID'], scores['KID']
                    self.print('FID:{} KID:{}'.format(fid, kid))

                    if kid < best_kid:
                        best_kid = kid
                        self.save('best', epoch, KID=kid, FID=fid)
                    if self.writer:
                        self.writer.add_scalar('valid/FID', fid, epoch)
                        self.writer.add_scalar('valid/KID', kid, epoch)

            for scheduler in self.lr_schedulers.values():
                scheduler.step(epoch)

    def sample_images(self, iteration_done=0):
        self.set_mode('eval')

        device = self.device
        batchA = next(iter(self.tst_loader))
        batchB = next(iter(self.tst_loader2))
        batch = Hdf5Dataset.merge_batch(batchA, batchB, device)
        imgs, img_lens, lbs, lb_lens, wids = batch

        real_imgs, real_img_lens = imgs.to(device), img_lens.to(device)
        real_lbs, real_lb_lens = lbs.to(device), lb_lens.to(device)

        with torch.no_grad():
            self.eval_z.sample_()
            recn_imgs = None
            if 'E' in self.models:
                enc_styles = self.models.E(real_imgs, real_img_lens, self.models.W.cnn_backbone)
                noises = torch.randn((real_imgs.size(0), self.opt.GenModel.style_dim
                                      - self.opt.EncModel.style_dim)).float().to(device)
                enc_z = torch.cat([noises, enc_styles], dim=-1)
                recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

            fake_real_imgs = self.models.G(self.eval_z, real_lbs, real_lb_lens)

            self.eval_y.sample_()
            sampled_words = idx_to_words(self.eval_y, self.lexicon, self.opt.training.capitalize_ratio)
            sampled_words[-2] = sampled_words[-1]
            fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
            fake_lbs, fake_lb_lens = fake_lbs.to(device), fake_lb_lens.to(device)
            fake_imgs = self.models.G(self.eval_z, fake_lbs, fake_lb_lens)

            max_img_len = max([real_imgs.size(-1), fake_real_imgs.size(-1), fake_imgs.size(-1)])
            img_shape = [real_imgs.size(2), max_img_len, real_imgs.size(1)]

            real_imgs = F.pad(real_imgs, [0, max_img_len - real_imgs.size(-1), 0, 0], value=-1.)
            fake_real_imgs = F.pad(fake_real_imgs, [0, max_img_len - fake_real_imgs.size(-1), 0, 0], value=-1.)
            fake_imgs = F.pad(fake_imgs, [0, max_img_len - fake_imgs.size(-1), 0, 0], value=-1.)
            recn_imgs = F.pad(recn_imgs, [0, max_img_len - recn_imgs.size(-1), 0, 0], value=-1.) \
                        if recn_imgs is not None else None

            real_words = self.label_converter.decode(real_lbs, real_lb_lens)
            real_labels = words_to_images(real_words, *img_shape)
            rand_labels = words_to_images(sampled_words, *img_shape)

            try:
                sample_img_list = [real_labels.cpu(), real_imgs.cpu(), fake_real_imgs.cpu(),
                                   fake_imgs.cpu(), rand_labels.cpu()]
                if recn_imgs is not None:
                    sample_img_list.insert(2, recn_imgs.cpu())
                sample_imgs = torch.cat(sample_img_list, dim=2).repeat(1, 3, 1, 1)
                res_img = draw_image(1 - sample_imgs.data, nrow=self.opt.training.sample_nrow, normalize=True)
                save_path = os.path.join(self.log_root, self.opt.training.sample_dir,
                                         'iter_{}.png'.format(iteration_done))
                im = Image.fromarray(res_img)
                im.save(save_path)
                if self.writer:
                    self.writer.add_image('Image', res_img.transpose((2, 0, 1)), iteration_done)
            except RuntimeError as e:
                print(e)

    def image_generator(self, source_dloader, style_guided=True):
        device = self.device

        with torch.no_grad():
            for style_imgs, style_img_lens, style_lbs, style_lb_lens, style_wids in source_dloader:
                content_lbs, content_lb_lens = style_lbs.to(device), style_lb_lens.to(device)

                if style_guided:
                    enc_styles = self.models.E(style_imgs.to(device), style_img_lens.to(device),
                                               self.models.W.cnn_backbone)
                    noises = torch.randn((style_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(style_imgs.size(0), self.opt.GenModel.style_dim).to(device)

                fake_imgs = self.models.G(enc_z, content_lbs.long(), content_lb_lens.long())
                fake_img_lens = content_lb_lens * self.opt.char_width
                yield fake_imgs, fake_img_lens, content_lbs, content_lb_lens, style_wids.to(device)


    def validate(self, guided=True):
        self.set_mode('eval')
        dset_name = self.opt.valid.dset_name if self.opt.valid.dset_name \
                    else self.opt.dataset
        dset = get_dataset(dset_name, self.opt.valid.dset_split)
        dloader = DataLoader(
            dset,
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        # style images are resized
        source_dloader = DataLoader(
            get_dataset(self.opt.valid.dset_name.strip('_org'), self.opt.valid.dset_split),
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        generator = self.image_generator(source_dloader, guided)
        fid_kid = calculate_kid_fid(self.opt.valid, dloader, generator, self.max_valid_image_width, self.device)
        return fid_kid

    def eval_interp(self):
        self.set_mode('eval')

        with torch.no_grad():
            interp_num = self.opt.test.interp_num
            nrow, ncol = 1, interp_num
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                fake_lbs = self.label_converter.encode(text)
                fake_lbs = torch.LongTensor(fake_lbs)
                fake_lb_lens = torch.IntTensor([len(text)])

                style0 = torch.randn((1, self.opt.GenModel.style_dim))
                style1 = torch.randn(style0.size())
                noise = torch.randn((1, self.noise_dim)).repeat(interp_num, 1).to(self.device)

                styles = [torch.lerp(style0, style1, i / (interp_num - 1)) for i in range(interp_num)]
                styles = torch.cat(styles, dim=0).float().to(self.device)
                styles = torch.cat([noise, styles], dim=1).to(self.device)

                fake_lbs, fake_lb_lens = fake_lbs.repeat(nrow * ncol, 1).to(self.device),\
                                         fake_lb_lens.repeat(nrow * ncol).to(self.device)
                gen_imgs = self.models.G(styles, fake_lbs, fake_lb_lens)
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                plt.figure()
                for i in range(nrow * ncol):
                    plt.subplot(nrow, ncol, i + 1)
                    plt.imshow(gen_imgs[i], cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.show()

    def eval_style(self):
        self.set_mode('eval')

        tst_loader = DataLoader(
            get_dataset('iam_word', self.opt.training.dset_split),
            batch_size=self.opt.test.nrow,
            shuffle=True,
            collate_fn=self.collect_fn
        )

        with torch.no_grad():
            nrow, ncol = self.opt.test.nrow, 2
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                texts = text.split(' ')
                ncol = len(texts)
                batch = next(iter(tst_loader))
                imgs, img_lens, lbs, lb_lens, wids = batch
                real_imgs, real_img_lens = imgs.to(self.device), img_lens.to(self.device)
                if len(texts) == 1:
                    fake_lbs = self.label_converter.encode(texts)
                    fake_lbs = torch.LongTensor(fake_lbs)
                    fake_lb_lens = torch.IntTensor([len(texts[0])])
                else:
                    fake_lbs, fake_lb_lens = self.label_converter.encode(texts)

                fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                fake_lb_lens = fake_lb_lens.repeat(nrow,).to(self.device)
                enc_styles = self.models.E(real_imgs, real_img_lens, self.models.W.cnn_backbone).unsqueeze(1).\
                                repeat(1, ncol, 1).view(nrow * ncol, self.opt.EncModel.style_dim)
                noises = torch.randn((nrow, self.noise_dim)).unsqueeze(1).\
                                repeat(1, ncol, 1).view(nrow * ncol, self.noise_dim).to(self.device)
                enc_styles = torch.cat([noises, enc_styles], dim=-1)

                gen_imgs = self.models.G(enc_styles, fake_lbs, fake_lb_lens)
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                real_imgs = (1 - real_imgs).squeeze().cpu().numpy() * 127
                plt.figure()
                for i in range(nrow):
                    plt.subplot(nrow, 1 + ncol, i * (1 + ncol) + 1)
                    plt.imshow(real_imgs[i], cmap='gray')
                    plt.axis('off')
                    for j in range(ncol):
                        plt.subplot(nrow, 1 + ncol, i * (1 + ncol) + 2 + j)
                        plt.imshow(gen_imgs[i * ncol + j], cmap='gray')
                        plt.axis('off')
                plt.tight_layout()
                plt.show()

    def eval_rand(self):
        self.set_mode('eval')

        with torch.no_grad():
            nrow, ncol = self.opt.test.nrow, 2
            rand_z = prepare_z_dist(nrow, self.opt.GenModel.style_dim, self.device)
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                texts = text.split(' ')
                ncol = len(texts)
                if len(texts) == 1:
                    fake_lbs = self.label_converter.encode(texts)
                    fake_lbs = torch.LongTensor(fake_lbs)
                    fake_lb_lens = torch.IntTensor([len(texts[0])])
                else:
                    fake_lbs, fake_lb_lens = self.label_converter.encode(texts)

                fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                fake_lb_lens = fake_lb_lens.repeat(nrow, ).to(self.device)

                rand_z.sample_()
                rand_styles = rand_z.unsqueeze(1).repeat(1, ncol, 1).view(nrow * ncol, self.opt.GenModel.style_dim)
                gen_imgs = self.models.G(rand_styles, fake_lbs, fake_lb_lens)
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                plt.figure()
                for i in range(nrow):
                    for j in range(ncol):
                        plt.subplot(nrow, ncol, i * ncol + 1 + j)
                        plt.imshow(gen_imgs[i * ncol + j], cmap='gray')
                        plt.axis('off')
                plt.tight_layout()
                plt.show()

    def eval_text(self):
        self.set_mode('eval')

        tst_loader = DataLoader(
            get_dataset('iam_word', self.opt.training.dset_split),
            batch_size=self.opt.test.nrow,
            shuffle=True,
            collate_fn=self.collect_fn
        )

        def get_space_index(text):
            idxs = []
            for i, ch in enumerate(text):
                if ch == ' ':
                    idxs.append(i)
            return idxs

        with torch.no_grad():
            nrow = self.opt.test.nrow
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                batch = next(iter(tst_loader))
                imgs, img_lens, lbs, lb_lens, wids = batch
                real_imgs, real_img_lens = imgs.to(self.device), img_lens.to(self.device)
                fake_lbs = self.label_converter.encode(text)
                fake_lbs = torch.LongTensor(fake_lbs)
                fake_lb_lens = torch.IntTensor([len(text)])

                fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                fake_lb_lens = fake_lb_lens.repeat(nrow,).to(self.device)
                enc_styles = self.models.E(real_imgs, real_img_lens, self.models.W.cnn_backbone)
                noises = torch.randn((nrow, self.noise_dim)).to(self.device)
                enc_styles = torch.cat([noises, enc_styles], dim=-1)

                real_imgs = (1 - real_imgs).squeeze().cpu().numpy() * 127
                gen_imgs = self.models.G(enc_styles, fake_lbs, fake_lb_lens)
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                space_indexs = get_space_index(text)
                for idx in space_indexs:
                    gen_imgs[:, :, idx * 16: (idx + 1) * 16] = 255

                plt.figure()

                for i in range(nrow):
                    plt.subplot(nrow * 2, 1, i * 2 + 1)
                    plt.imshow(real_imgs[i], cmap='gray')
                    plt.axis('off')
                    plt.subplot(nrow * 2, 1, i * 2 + 2)
                    plt.imshow(gen_imgs[i], cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.show()
