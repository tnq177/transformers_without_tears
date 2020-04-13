import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from layers import Encoder, Decoder
import utils as ut
import all_constants as ac


class Transformer(nn.Module):
    """Transformer https://arxiv.org/pdf/1706.03762.pdf"""
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args

        embed_dim = args.embed_dim
        fix_norm = args.fix_norm
        joint_vocab_size = args.joint_vocab_size
        lang_vocab_size = args.lang_vocab_size
        use_bias = args.use_bias
        self.scale = embed_dim ** 0.5

        if args.mask_logit:
            # mask logits separately per language
            self.logit_mask = None
        else:
            # otherwise, use the same mask for all
            # this only masks out BOS and PAD
            mask = [True] * joint_vocab_size
            mask[ac.BOS_ID] = False
            mask[ac.PAD_ID] = False
            self.logit_mask = torch.tensor(mask).type(torch.bool)

        self.word_embedding = Parameter(torch.Tensor(joint_vocab_size, embed_dim))
        self.lang_embedding = Parameter(torch.Tensor(lang_vocab_size, embed_dim))
        self.out_bias = Parameter(torch.Tensor(joint_vocab_size)) if use_bias else None

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        # initialize
        nn.init.normal_(self.lang_embedding, mean=0, std=embed_dim ** -0.5)
        if fix_norm:
            d = 0.01
            nn.init.uniform_(self.word_embedding, a=-d, b=d)
        else:
            nn.init.normal_(self.word_embedding, mean=0, std=embed_dim ** -0.5)

        if use_bias:
            nn.init.constant_(self.out_bias, 0.)

    def replace_with_unk(self, toks):
        # word-dropout
        p = self.args.word_dropout
        if self.training and 0 < p < 1:
            non_pad_mask = toks != ac.PAD_ID
            mask = (torch.rand(toks.size()) <= p).type(non_pad_mask.type())
            mask = mask & non_pad_mask
            toks[mask] = ac.UNK_ID

    def get_input(self, toks, lang_idx, word_embedding, pos_embedding):
        # word dropout, but replace with unk instead of zero-ing embed
        self.replace_with_unk(toks)
        word_embed = F.embedding(toks, word_embedding) * self.scale # [bsz, len, dim]
        lang_embed = self.lang_embedding[lang_idx].unsqueeze(0).unsqueeze(1) # [1, 1, dim]
        pos_embed = pos_embedding[:toks.size(-1), :].unsqueeze(0) # [1, len, dim]

        return word_embed + lang_embed + pos_embed

    def forward(self, src, tgt, targets, src_lang_idx, tgt_lang_idx, logit_mask):
        embed_dim = self.args.embed_dim
        max_len = max(src.size(1), tgt.size(1))
        pos_embedding = ut.get_positional_encoding(embed_dim, max_len)
        word_embedding = F.normalize(self.word_embedding, dim=-1) if self.args.fix_norm else self.word_embedding

        encoder_inputs = self.get_input(src, src_lang_idx, word_embedding, pos_embedding)
        encoder_mask = (src == ac.PAD_ID).unsqueeze(1).unsqueeze(2)
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)

        decoder_inputs = self.get_input(tgt, tgt_lang_idx, word_embedding, pos_embedding)
        decoder_mask = torch.triu(torch.ones((tgt.size(-1), tgt.size(-1))), diagonal=1).type(tgt.type()) == 1
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(1)
        decoder_outputs = self.decoder(decoder_inputs, decoder_mask, encoder_outputs, encoder_mask)

        logit_mask = logit_mask == 1 if self.logit_mask is None else self.logit_mask
        logits = self.logit_fn(decoder_outputs, word_embedding, logit_mask)
        neglprobs = F.log_softmax(logits, -1) * logit_mask.type(logits.type()).reshape(1, -1)
        targets = targets.reshape(-1, 1)
        non_pad_mask = targets != ac.PAD_ID

        nll_loss = neglprobs.gather(dim=-1, index=targets)[non_pad_mask]
        smooth_loss = neglprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        # label smoothing: https://arxiv.org/pdf/1701.06548.pdf
        nll_loss = -(nll_loss.sum())
        smooth_loss = -(smooth_loss.sum())
        label_smoothing = self.args.label_smoothing
        if label_smoothing > 0:
            loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss / logit_mask.type(nll_loss.type()).sum()
        else:
            loss = nll_loss

        num_words = non_pad_mask.type(loss.type()).sum()
        opt_loss = loss / num_words
        return {
            'opt_loss': opt_loss,
            'loss': loss,
            'nll_loss': nll_loss,
            'num_words': num_words
        }

    def logit_fn(self, decoder_output, softmax_weight, logit_mask):
        logits = F.linear(decoder_output, softmax_weight, bias=self.out_bias)
        logits = logits.reshape(-1, logits.size(-1))
        logits[:, ~logit_mask] = -1e9
        return logits

    def beam_decode(self, src, src_lang_idx, tgt_lang_idx, logit_mask):
        embed_dim = self.args.embed_dim
        max_len = src.size(1) + 51
        pos_embedding = ut.get_positional_encoding(embed_dim, max_len)
        word_embedding = F.normalize(self.word_embedding, dim=-1) if self.args.fix_norm else self.word_embedding
        logit_mask = logit_mask == 1 if self.logit_mask is None else self.logit_mask
        tgt_lang_embed = self.lang_embedding[tgt_lang_idx]

        encoder_inputs = self.get_input(src, src_lang_idx, word_embedding, pos_embedding)
        encoder_mask = (src == ac.PAD_ID).unsqueeze(1).unsqueeze(2)
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)

        def get_tgt_inp(tgt, time_step):
            word_embed = F.embedding(tgt.type(src.type()), word_embedding) * self.scale
            pos_embed = pos_embedding[time_step, :].reshape(1, 1, -1)
            return word_embed + tgt_lang_embed + pos_embed

        def logprob_fn(decoder_output):
            logits = self.logit_fn(decoder_output, word_embedding, logit_mask)
            return F.log_softmax(logits, dim=-1)

        # following Attention is all you need, we decode up to src_len + 50 tokens only
        max_lengths = torch.sum(src != ac.PAD_ID, dim=-1).type(src.type()) + 50
        return self.decoder.beam_decode(encoder_outputs, encoder_mask, get_tgt_inp, logprob_fn, ac.BOS_ID, ac.EOS_ID, max_lengths, beam_size=self.args.beam_size, alpha=self.args.beam_alpha)
