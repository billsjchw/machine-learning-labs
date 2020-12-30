import numpy as np
import tables
import json
import os
import sys
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
import torch.nn.utils as nnutils
import torch.nn.utils.rnn as rnnutils
import torch.optim as optim
import torch.utils.data as data
from misc import *

PAD, SOS, EOS, UNK = 0, 1, 2, 3
DEVICE = torch.device('cuda:0')
CONFIG = {
    'max_utt_len': 40,
    'max_ctx_len': 10,
    'emb_size': 200,
    'hid_size': 300,
    'z_size': 200,
    'noise_std': 0.2,
    'comp_num': 3,
    'gumbel_temp': 0.1,
    'batch_size': 32,
    'epoch_num': 100,
    'auto_enc_lr': 1.0,
    'gen_lr': 5e-5,
    'disc_lr': 1e-5,
    'max_norm': 1.,
    'lambda_gp': 10,
    'disc_iter_num': 5,
    'log_every': 25,
}


class DialogDataset(data.Dataset):
    def __init__(self, filename, max_ctx_len, max_utt_len):
        self.max_ctx_len = max_ctx_len
        self.max_utt_len = max_utt_len

        file = tables.open_file(filename)
        self.data = file.get_node('/sentences')[:]
        self.index = file.get_node('/indices')[:]

    def __getitem__(self, offset):
        pos_utt = self.index[offset]['pos_utt']
        ctx_len = self.index[offset]['ctx_len']
        resp_len = self.index[offset]['res_len']

        ctx_arr = self.data[pos_utt - ctx_len:pos_utt]
        resp_arr = self.data[pos_utt:pos_utt + resp_len]

        def align_utt(utt):
            if len(utt) < self.max_utt_len:
                return np.concatenate([utt, [PAD] * (self.max_utt_len - len(utt))])
            else:
                return np.concatenate([utt[:self.max_utt_len - 1], [EOS]])

        def align_ctx(ctx):
            if len(ctx) < self.max_ctx_len:
                return np.concatenate(
                    [ctx, [[SOS, EOS] + [PAD] * (self.max_utt_len - 2)] * (self.max_ctx_len - len(ctx))])
            else:
                return ctx[:self.max_ctx_len]

        ctx = np.split(ctx_arr, np.where(ctx_arr == EOS)[0] + 1)[:-1]
        ctx = [align_utt(utt[1:]) for utt in ctx]
        ctx = np.array(ctx)
        ctx_len = min(len(ctx), self.max_ctx_len)
        ctx = align_ctx(ctx)
        utt_lens = np.array([np.where(utt == EOS)[0][0] + 1 for utt in ctx])
        resp = resp_arr[1:]
        resp_len = min(len(resp), self.max_utt_len)
        resp = align_utt(resp)
        floors = np.array(([resp_arr[0] + 2, -1 - resp_arr[0]] * self.max_ctx_len)[:self.max_ctx_len])

        return ctx, ctx_len, utt_lens, floors, resp, resp_len

    def __len__(self):
        return self.index.shape[0]


def load_word2vec(filename, vocab, emb_size):
    lines = open(filename, 'r', encoding='UTF-8').readlines()
    raw_word2vec = {line.split(' ', 1)[0]: line.split(' ', 1)[1] for line in lines}
    return np.array([
        np.fromstring(raw_word2vec[word], sep=' ') if word in raw_word2vec else np.random.randn(emb_size) * 0.1
        for word in vocab
    ])


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hid_size, bidirectional):
        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.hid_size = hid_size
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(input_size, hid_size, batch_first=True, bidirectional=bidirectional)

        for param in self.rnn.parameters():
            if param.dim() > 1:
                init.orthogonal_(param)

    def forward(self, seqs, seq_lens):
        batch_size = seqs.size()[0]
        rnn_input = rnnutils.pack_padded_sequence(seqs, seq_lens, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(rnn_input)
        enc_seqs = h_n.transpose(1, 0).contiguous().view(batch_size, -1)
        return enc_seqs


class UtteranceEncoder(nn.Module):
    def __init__(self, embedding, rnn_encoder, noise_std):
        super(UtteranceEncoder, self).__init__()

        self.noise_std = noise_std

        self.embedding = embedding
        self.rnn_encoder = rnn_encoder

    def forward(self, utts, utt_lens):
        emb_utts = self.embedding(utts)

        emb_utts = functional.dropout(emb_utts, 0.5, self.training)
        enc_utts = self.rnn_encoder(emb_utts, utt_lens)

        noises = torch.normal(torch.zeros(enc_utts.size()), self.noise_std).to(DEVICE)
        enc_utts = enc_utts + noises

        return enc_utts


class ContextEncoder(nn.Module):
    def __init__(self, utt_encoder, rnn_encoder, noise_std):
        super(ContextEncoder, self).__init__()

        self.noise_std = noise_std

        self.utt_encoder = utt_encoder
        self.rnn_encoder = rnn_encoder

    def forward(self, ctxs, ctx_lens, utt_lenss, floorss):
        batch_size, max_ctx_len, max_utt_len = ctxs.size()

        utts = ctxs.view(-1, max_utt_len)
        utt_lens = utt_lenss.view(-1)
        enc_utts = self.utt_encoder(utts, utt_lens)
        enc_uttss = enc_utts.view(batch_size, max_ctx_len, -1)

        floor_one_hotss = torch.zeros(floorss.numel(), 2, device=DEVICE) \
            .scatter_(1, floorss.view(-1, 1), 1).view(-1, max_ctx_len, 2)
        enc_utt_and_floorss = torch.cat([enc_uttss, floor_one_hotss], 2)

        enc_utt_and_floorss = functional.dropout(enc_utt_and_floorss, 0.25, self.training)
        enc_ctxs = self.rnn_encoder(enc_utt_and_floorss, ctx_lens)

        noises = torch.normal(torch.zeros(enc_ctxs.size()), self.noise_std).to(DEVICE)
        enc_ctxs = enc_ctxs + noises

        return enc_ctxs


class Variation(nn.Module):
    def __init__(self, input_size, z_size):
        super(Variation, self).__init__()

        self.input_size = input_size
        self.z_size = z_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-5, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-5, momentum=0.1),
            nn.Tanh(),
        )
        self.linear_mu = nn.Linear(z_size, z_size)
        self.linear_sigma = nn.Linear(z_size, z_size)

        def init_weights(module):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.uniform_(-0.02, 0.02)  # noqa
                    module.bias.fill_(0)

        self.fc.apply(init_weights)
        init_weights(self.linear_mu)
        init_weights(self.linear_sigma)

    def forward(self, inputs):
        batch_size = inputs.size()[0]

        tensor = self.fc(inputs)
        mus = self.linear_mu(tensor)
        log_sqr_sigmas = self.linear_sigma(tensor)
        stds = torch.exp(0.5 * log_sqr_sigmas)

        epsilons = torch.randn(batch_size, self.z_size, device=DEVICE) * stds + mus

        return epsilons


class MixVariation(nn.Module):
    def __init__(self, input_size, z_size, comp_num, gumbel_temp):
        super(MixVariation, self).__init__()

        self.input_size = input_size
        self.z_size = z_size
        self.comp_num = comp_num
        self.gumbel_temp = gumbel_temp

        self.pi_net = nn.Sequential(
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-5, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, comp_num),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-5, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-5, momentum=0.1),
            nn.Tanh(),
        )
        self.linear_mu = nn.Linear(z_size, comp_num * z_size)
        self.linear_sigma = nn.Linear(z_size, comp_num * z_size)

        def init_weights(module):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.uniform_(-0.05, 0.05)  # noqa
                    module.bias.fill_(0)

        self.pi_net.apply(init_weights)
        self.fc.apply(init_weights)
        init_weights(self.linear_mu)
        init_weights(self.linear_sigma)

    def forward(self, inputs):
        batch_size = inputs.size()[0]

        tensor = self.fc(inputs)
        pis = functional.gumbel_softmax(self.pi_net(tensor), self.gumbel_temp, True).unsqueeze(1)
        mus = self.linear_mu(tensor)
        log_sqr_sigmas = self.linear_sigma(tensor)
        stds = torch.exp(0.5 * log_sqr_sigmas)

        epsilonis = (torch.randn(batch_size, self.comp_num * self.z_size, device=DEVICE) * stds + mus) \
            .view(batch_size, self.comp_num, self.z_size)
        epsilons = torch.bmm(pis, epsilonis).squeeze(1)

        return epsilons


class Generator(nn.Module):
    def __init__(self, z_size):
        super(Generator, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Linear(z_size, z_size),
        )

        def init_weights(module):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.uniform_(-0.02, 0.02)  # noqa
                    module.bias.fill_(0)

        self.module.apply(init_weights)

    def forward(self, epsilons):
        return self.module(epsilons)


class Discriminator(nn.Module):
    def __init__(self, c_size, z_size):
        super(Discriminator, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(c_size + z_size, c_size * 2),
            nn.BatchNorm1d(c_size * 2, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(c_size * 2, c_size * 2),
            nn.BatchNorm1d(c_size * 2, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(c_size * 2, 1),
        )

        def init_weights(module):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.uniform_(-0.02, 0.02)  # noqa
                    module.bias.fill_(0)

        self.module.apply(init_weights)

    def forward(self, zs, cs):
        return self.module(torch.cat([zs, cs], 1))


class RNNDecoder(nn.Module):
    def __init__(self, input_size, hid_size, vocab_size):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.hid_size = hid_size
        self.vocab_size = vocab_size

        self.rnn = nn.GRU(input_size, hid_size, batch_first=True)
        self.out = nn.Linear(hid_size, vocab_size)

        for param in self.rnn.parameters():
            if param.dim() > 1:
                init.orthogonal_(param)
        with torch.no_grad():
            self.out.weight.uniform_(-0.1, 0.1)  # noqa
            self.out.bias.fill_(0)


class TrainingDecoder(nn.Module):
    def __init__(self, embedding, rnn_decoder):
        super(TrainingDecoder, self).__init__()

        self.embedding = embedding
        self.rnn_decoder = rnn_decoder

    def forward(self, zs, cs, resps):
        zcs = torch.cat([zs, cs], 1)
        batch_size, max_utt_len = resps.size()

        emb_resps = self.embedding(resps)

        emb_resps = functional.dropout(emb_resps, 0.5, self.training)
        rnn_output, _ = self.rnn_decoder.rnn(emb_resps, zcs.unsqueeze(0))
        dec_resps = self.rnn_decoder.out(rnn_output.contiguous().view(-1, self.rnn_decoder.hid_size)) \
            .view(batch_size, max_utt_len, self.rnn_decoder.vocab_size)

        return dec_resps


class SamplingDecoder(nn.Module):
    def __init__(self, embedding, rnn_decoder):
        super(SamplingDecoder, self).__init__()

        self.embedding = embedding
        self.rnn_decoder = rnn_decoder

    def forward(self, zs, cs, max_utt_len):
        zcs = torch.cat([zs, cs], 1)
        batch_size = zcs.size()[0]

        resps = torch.zeros(batch_size, max_utt_len).to(torch.long)

        rnn_input = self.embedding(torch.tensor([SOS] * batch_size).to(DEVICE).view(batch_size, 1))
        h_n = zcs.unsqueeze(0)
        for i in range(max_utt_len):
            rnn_output, h_n = self.rnn_decoder.rnn(rnn_input, h_n)
            top_word_idxes = self.rnn_decoder.out(rnn_output)[:, -1].max(1, keepdim=True)[1]
            rnn_input = self.embedding(top_word_idxes)
            resps[:, i] = top_word_idxes.squeeze().to(torch.long)

        resp_lens = torch.tensor([
            torch.nonzero(torch.cat([resp, torch.tensor([EOS])]).eq(EOS))[0, 0] + 1
            for resp in resps
        ])

        return resps, resp_lens


class DialogWAE(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, z_size, noise_std, comp_num, gumbel_temp):
        super(DialogWAE, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.z_size = z_size
        self.noise_std = noise_std
        self.comp_num = comp_num
        self.gumbel_temp = gumbel_temp

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD)
        self.utt_encoder = UtteranceEncoder(self.embedding, RNNEncoder(emb_size, hid_size, True), noise_std)
        self.ctx_encoder = ContextEncoder(self.utt_encoder, RNNEncoder(hid_size * 2 + 2, hid_size, False), noise_std)
        self.post_net = Variation(hid_size * 3, z_size)
        self.prior_net = MixVariation(hid_size, z_size, comp_num, gumbel_temp)
        self.post_generator = Generator(z_size)
        self.prior_generator = Generator(z_size)
        rnn_decoder = RNNDecoder(emb_size, hid_size + z_size, vocab_size)
        self.training_decoder = TrainingDecoder(self.embedding, rnn_decoder)
        self.sampling_decoder = SamplingDecoder(self.embedding, rnn_decoder)
        self.discriminator = Discriminator(hid_size, z_size)


class Chatbot:
    def __init__(self, dialog_wae):
        super(Chatbot, self).__init__()

        self.dialog_wae = dialog_wae

    def train(self, training_dataset, validation_dataset, batch_size, epoch_num, auto_enc_lr, gen_lr, disc_lr, max_norm,
              lambda_gp, disc_iter_num, log_every, logger, saving_path, start_epoch_no):
        auto_enc_params = [
            *self.dialog_wae.ctx_encoder.parameters(),
            *self.dialog_wae.post_net.parameters(),
            *self.dialog_wae.post_generator.parameters(),
            *self.dialog_wae.training_decoder.parameters(),
        ]
        gen_params = [
            *self.dialog_wae.post_net.parameters(),
            *self.dialog_wae.post_generator.parameters(),
            *self.dialog_wae.prior_net.parameters(),
            *self.dialog_wae.prior_generator.parameters(),
        ]
        disc_params = [
            *self.dialog_wae.discriminator.parameters()
        ]
        auto_enc_optimizer = optim.SGD(auto_enc_params, lr=auto_enc_lr)
        gen_optimizer = optim.RMSprop(gen_params, lr=gen_lr)
        disc_optimizer = optim.RMSprop(disc_params, lr=disc_lr)
        auto_enc_lr_scheduler = optim.lr_scheduler.StepLR(auto_enc_optimizer, step_size=10, gamma=0.6)

        def calc_auto_enc_loss(cs, xs, resps):
            post_epsilons = self.dialog_wae.post_net(torch.cat([xs, cs], 1))
            post_zs = self.dialog_wae.post_generator(post_epsilons)
            dec_resps = self.dialog_wae.training_decoder(post_zs, cs, resps[:, :-1])
            mask = resps[:, 1:].contiguous().view(-1).gt(0)
            cross_ent_input = dec_resps.view(-1, self.dialog_wae.vocab_size) \
                .masked_select(mask.unsqueeze(1).expand(mask.size()[0], self.dialog_wae.vocab_size)) \
                .view(-1, self.dialog_wae.vocab_size)
            cross_ent_target = resps[:, 1:].contiguous().view(-1).masked_select(mask)
            return nn.CrossEntropyLoss()(cross_ent_input, cross_ent_target)

        def calc_post_disc_err(cs, xs):
            post_epsilons = self.dialog_wae.post_net(torch.cat([xs, cs], 1))
            post_zs = self.dialog_wae.post_generator(post_epsilons)
            return torch.mean(self.dialog_wae.discriminator(post_zs, cs))

        def calc_prior_disc_err(cs):
            prior_epsilons = self.dialog_wae.prior_net(cs)
            prior_zs = self.dialog_wae.prior_generator(prior_epsilons)
            return torch.mean(self.dialog_wae.discriminator(prior_zs, cs))

        def calc_gradient_penalty(cs, xs):
            post_epsilons = self.dialog_wae.post_net(torch.cat([xs, cs], 1))
            post_zs = self.dialog_wae.post_generator(post_epsilons)
            prior_epsilons = self.dialog_wae.prior_net(cs)
            prior_zs = self.dialog_wae.prior_generator(prior_epsilons)
            alphas = torch.rand(batch_size, 1, device=DEVICE).expand(prior_zs.size())
            interps = alphas * prior_zs.detach() + (1 - alphas) * post_zs.detach()
            interps.requires_grad = True
            interp_disc_errs = torch.mean(self.dialog_wae.discriminator(interps, cs))
            gradients = autograd.grad(interp_disc_errs, interps,
                                      torch.ones(interp_disc_errs.size(), device=DEVICE), True, True)[0]
            return torch.mean((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2) * lambda_gp

        def train_auto_encoder(ctxs, ctx_lens, utt_lenss, floorss, resps, resp_lens):
            self.dialog_wae.train()
            auto_enc_optimizer.zero_grad()

            cs = self.dialog_wae.ctx_encoder(ctxs, ctx_lens, utt_lenss, floorss)
            xs = self.dialog_wae.utt_encoder(resps, resp_lens)

            auto_enc_loss = calc_auto_enc_loss(cs, xs, resps)
            auto_enc_loss.backward()

            clip_params = [
                *self.dialog_wae.ctx_encoder.parameters(),
                *self.dialog_wae.training_decoder.parameters(),
            ]
            nnutils.clip_grad_norm_(clip_params, max_norm)
            auto_enc_optimizer.step()

            return auto_enc_loss.item()

        def train_generator(ctxs, ctx_lens, utt_lenss, floorss, resps, resp_lens):
            self.dialog_wae.train()
            self.dialog_wae.ctx_encoder.eval()
            gen_optimizer.zero_grad()

            cs = self.dialog_wae.ctx_encoder(ctxs, ctx_lens, utt_lenss, floorss).detach()
            xs = self.dialog_wae.utt_encoder(resps, resp_lens).detach()

            post_disc_err = calc_post_disc_err(cs, xs)
            post_disc_err.backward(torch.tensor(-1.).to(DEVICE))

            prior_disc_err = calc_prior_disc_err(cs)
            prior_disc_err.backward(torch.tensor(1.).to(DEVICE))

            gen_optimizer.step()

            gen_loss = prior_disc_err - post_disc_err
            return gen_loss.item()

        def train_discriminator(ctxs, ctx_lens, utt_lenss, floorss, resps, resp_lens):
            self.dialog_wae.train()
            self.dialog_wae.ctx_encoder.eval()
            disc_optimizer.zero_grad()

            cs = self.dialog_wae.ctx_encoder(ctxs, ctx_lens, utt_lenss, floorss).detach()
            xs = self.dialog_wae.utt_encoder(resps, resp_lens).detach()

            post_disc_err = calc_post_disc_err(cs, xs)
            post_disc_err.backward(torch.tensor(1.).to(DEVICE))

            prior_disc_err = calc_prior_disc_err(cs)
            prior_disc_err.backward(torch.tensor(-1.).to(DEVICE))

            gradient_penalty = calc_gradient_penalty(cs, xs)
            gradient_penalty.backward()

            disc_optimizer.step()

            disc_loss = post_disc_err - prior_disc_err + gradient_penalty
            return disc_loss.item()

        def calc_loss(ctxs, ctx_lens, utt_lenss, floorss, resps, resp_lens):
            self.dialog_wae.eval()

            cs = self.dialog_wae.ctx_encoder(ctxs, ctx_lens, utt_lenss, floorss)
            xs = self.dialog_wae.utt_encoder(resps, resp_lens)

            post_disc_err = calc_post_disc_err(cs, xs)
            prior_disc_err = calc_prior_disc_err(cs)
            disc_loss = post_disc_err - prior_disc_err
            gen_loss = -disc_loss

            auto_enc_loss = calc_auto_enc_loss(cs, xs, resps)

            return auto_enc_loss.item(), gen_loss.item(), disc_loss.item()

        iter_num = (len(training_dataset) // batch_size // disc_iter_num) // log_every * log_every

        def log(epoch_no, iter_no, loss_records):
            if iter_no % log_every != 0:
                return

            logger.info(f'[training] epoch: [{epoch_no}/{epoch_num}], iter: [{iter_no}/{iter_num}]')
            logger.info(' '.join([f'({loss_name}, {loss_value:.4f})' for loss_name, loss_value in loss_records]))

        def save(epoch_no):
            torch.save(self, f'{saving_path}chatbot_epoch{epoch_no}.pckl')

            logger.info(f'[model saved] epoch: [{epoch_no}/{epoch_num}]')

        def validate(epoch_no):
            data_loader = data.DataLoader(validation_dataset, batch_size, True, num_workers=1, drop_last=True)
            loss_tuples = [calc_loss(*tuple(tensor.to(DEVICE) for tensor in batch)) for batch in data_loader]
            avg_loss_tuple = [
                np.mean([loss_tuple[i] for loss_tuple in loss_tuples])
                for i in range(3)
            ]

            logger.info(f'[validation] epoch: [{epoch_no}/{epoch_num}]')
            logger.info(f'(AE, {avg_loss_tuple[0]:.4f}) (G, {avg_loss_tuple[1]:.4f}) (D, {avg_loss_tuple[2]:.4f})')

        for epoch_no in range(start_epoch_no + 1, epoch_num + 1):
            data_loader = data.DataLoader(training_dataset, batch_size, True, num_workers=1, drop_last=True)
            data_loader_iter = iter(data_loader)
            for iter_no in range(1, iter_num + 1):
                batches = [
                    tuple(tensor.to(DEVICE) for tensor in next(data_loader_iter))
                    for _ in range(disc_iter_num)
                ]
                loss_records = [
                    ('AE', train_auto_encoder(*batches[0])),
                    ('G', train_generator(*batches[0])),
                    *[('D', train_discriminator(*batch)) for batch in batches],
                ]
                log(epoch_no, iter_no, loss_records)
            auto_enc_lr_scheduler.step()
            save(epoch_no)
            validate(epoch_no)

    def sample(self, ctxs, ctx_lens, utt_lenss, floorss, repeat, max_utt_len):
        self.dialog_wae.eval()

        cs = self.dialog_wae.ctx_encoder(ctxs, ctx_lens, utt_lenss, floorss)
        repeated_cs = cs.expand(repeat, -1)
        prior_epsilons = self.dialog_wae.prior_net(repeated_cs)
        prior_zs = self.dialog_wae.prior_generator(prior_epsilons)

        return self.dialog_wae.sampling_decoder(prior_zs, repeated_cs, max_utt_len)

    def init_embedding(self, word2vec):
        with torch.no_grad():
            self.dialog_wae.embedding.weight.copy_(torch.from_numpy(word2vec))
            self.dialog_wae.embedding.weight[0].fill_(0)


def main_training(dataset_name, start_epoch_no=0, word2vec_filename=None):
    os.makedirs(f'./output/{dataset_name}/models/', exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(message)s')
    log_file_handler = logging.FileHandler(f'./output/{dataset_name}/log.txt')
    log_file_handler.setFormatter(log_formatter)
    log_stdout_handler = logging.StreamHandler(sys.stdout)
    log_stdout_handler.setFormatter(log_formatter)
    logger.addHandler(log_file_handler)
    logger.addHandler(log_stdout_handler)

    vocab = json.loads(open(f'./data/{dataset_name}/vocab.json').read())
    vocab_size = len(vocab)

    training_dataset = DialogDataset(
        f'./data/{dataset_name}/train.h5',
        CONFIG['max_ctx_len'],
        CONFIG['max_utt_len'],
    )
    validation_dataset = DialogDataset(
        f'./data/{dataset_name}/valid.h5',
        CONFIG['max_ctx_len'],
        CONFIG['max_utt_len'],
    )

    if start_epoch_no > 0:
        chatbot = torch.load(f'./output/{dataset_name}/models/chatbot_epoch{start_epoch_no}.pckl')
    else:
        dialog_wae = DialogWAE(
            vocab_size,
            CONFIG['emb_size'],
            CONFIG['hid_size'],
            CONFIG['z_size'],
            CONFIG['noise_std'],
            CONFIG['comp_num'],
            CONFIG['gumbel_temp'],
        ).to(DEVICE)
        chatbot = Chatbot(dialog_wae)

    if word2vec_filename:
        word2vec = load_word2vec(f'./data/{word2vec_filename}', vocab, CONFIG['emb_size'])
        chatbot.init_embedding(word2vec)

    chatbot.train(
        training_dataset,
        validation_dataset,
        CONFIG['batch_size'],
        CONFIG['epoch_num'],
        CONFIG['auto_enc_lr'],
        CONFIG['gen_lr'],
        CONFIG['disc_lr'],
        CONFIG['max_norm'],
        CONFIG['lambda_gp'],
        CONFIG['disc_iter_num'],
        CONFIG['log_every'],
        logger,
        f'./output/{dataset_name}/models/',
        start_epoch_no,
    )


def main_evaluation(dataset_name, epoch_no):
    vocab = json.loads(open(f'./data/{dataset_name}/vocab.json').read())

    chatbot = torch.load(f'./output/{dataset_name}/models/chatbot_epoch{epoch_no}.pckl')

    test_dataset = DialogDataset(
        f'./data/{dataset_name}/test.h5',
        CONFIG['max_ctx_len'],
        CONFIG['max_utt_len'],
    )

    evaluate(
        chatbot,
        Metrics(),
        data.DataLoader(test_dataset, 1, False, num_workers=1),
        vocab,
        10,
        open(f'./output/{dataset_name}/result.txt', 'w'),
    )


if __name__ == '__main__':
    # main_training('dailydialog', 0, 'glove.twitter.27B.200d.txt')
    main_evaluation('dailydialog', 4)
