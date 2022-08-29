import numpy as np
import torch
import all_constants as ac


class DataManager(object):
    def __init__(self, args, io):
        super(DataManager, self).__init__()
        self.args = args
        self.io = io
        self.pairs = args.pairs.split(',')
        self.lang_vocab, self.lang_ivocab = io.load_lang_vocab()
        self.vocab, self.ivocab = io.load_vocab()
        self.logit_masks = {}
        for lang in self.lang_vocab:
            self.logit_masks[lang] = io.load_logit_mask(lang)

    def load_data(self):
        self.data = {}
        batch_size = self.args.batch_size
        for pair in self.pairs:
            self.data[pair] = {}
            for mode in [ac.TRAIN, ac.DEV]:
                src = self.io.load_npy_data(pair, mode, src=True)
                tgt = self.io.load_npy_data(pair, mode, src=False)
                self.args.logger.info(f'Loading {pair}-{mode}')
                self.data[pair][mode] = NMTDataset(src, tgt, batch_size)

        # batch sampling similar to in preprocessing.py
        ns = np.array([len(self.data[pair][ac.TRAIN]) for pair in self.pairs])
        ps = ns / sum(ns)
        ps = ps ** self.args.alpha
        ps = ps / sum(ps)
        self.ps = ps
        self.args.logger.info('Sampling batches with probs:')
        for idx, pair in enumerate(self.pairs):
            self.args.logger.info(f'{pair}, n={ns[idx]}, p={ps[idx]}')

        self.train_iters = {}
        for pair in self.pairs:
            self.train_iters[pair] = self.data[pair][ac.TRAIN].get_iter(shuffle=True)

        # load dev translate batches
        self.translate_data = {}
        for pair in self.pairs:
            self.args.logger.info('Loading dev translate batches')
            src_batches, sorted_idxs = self.get_translate_batches(pair, ac.DEV)
            self.translate_data[pair] = {
                'src_batches': src_batches,
                'sorted_idxs': sorted_idxs,
            }

    def get_batch(self):
        pair = np.random.choice(self.pairs, p=self.ps)
        try:
            src, tgt, targets = next(self.train_iters[pair])
        except StopIteration:
            self.train_iters[pair] = self.data[pair][ac.TRAIN].get_iter(shuffle=True)
            src, tgt, targets = next(self.train_iters[pair])

        src_lang, tgt_lang = pair.split('2')
        return {
            'src': src,
            'tgt': tgt,
            'targets': targets,
            'src_lang_idx': self.lang_vocab[src_lang],
            'tgt_lang_idx': self.lang_vocab[tgt_lang],
            'pair': pair,
            'logit_mask': self.logit_masks[tgt_lang]
        }

    def get_translate_batches(self, pair, mode, input_file=None):    
        data = []
        lens = []
        raw_data = self.io.load_bpe_data(pair, mode, src=True, input_file=input_file)
        for tokenized_line in raw_data:
            ids = [self.vocab.get(tok, ac.UNK_ID) for tok in tokenized_line] + [ac.EOS_ID]
            data.append(ids)
            lens.append(len(ids))

        lens = np.array(lens)
        data = np.array(data)
        sorted_idxs = np.argsort(lens)
        lens = lens[sorted_idxs]
        data = data[sorted_idxs]

        # create batches
        batch_size = self.args.decode_batch_size
        src_batches = []
        s_idx = 0
        length = data.shape[0]
        while s_idx < length:
            e_idx = s_idx + 1
            max_in_batch = lens[s_idx]
            while e_idx < length:
                max_in_batch = max(max_in_batch, lens[e_idx])
                count = (e_idx - s_idx + 1) * 2 * max_in_batch
                if count > batch_size:
                    break
                else:
                    e_idx += 1

            max_in_batch = max(lens[s_idx:e_idx])
            src = np.zeros((e_idx - s_idx, max_in_batch), dtype=np.int32)
            for i in range(s_idx, e_idx):
                src[i - s_idx] = list(data[i]) + (max_in_batch - lens[i]) * [ac.PAD_ID]
            src_batches.append(torch.from_numpy(src).type(torch.long))
            s_idx = e_idx

        return src_batches, sorted_idxs


class NMTDataset(object):
    def __init__(self, src, tgt, batch_size):
        super(NMTDataset, self).__init__()
        if src.shape[0] != tgt.shape[0]:
            raise ValueError('src and tgt must have the same size')

        self.batch_size = batch_size
        self.batches = []

        sorted_idxs = np.argsort([len(x) for x in src])
        src = src[sorted_idxs]
        tgt = tgt[sorted_idxs]
        src_lens = [len(x) for x in src]
        tgt_lens = [len(x) for x in tgt]

        # prepare batches
        s_idx = 0
        while s_idx < src.shape[0]:
            e_idx = s_idx + 1
            max_src_in_batch = src_lens[s_idx]
            max_tgt_in_batch = tgt_lens[s_idx]
            while e_idx < src.shape[0]:
                max_src_in_batch = max(max_src_in_batch, src_lens[e_idx])
                max_tgt_in_batch = max(max_tgt_in_batch, tgt_lens[e_idx])
                num_toks = (e_idx - s_idx + 1) * max(max_src_in_batch, max_tgt_in_batch)
                if num_toks > self.batch_size:
                    break
                else:
                    e_idx += 1

            batch = self.prepare_one_batch(
                src[s_idx:e_idx],
                tgt[s_idx:e_idx],
                src_lens[s_idx:e_idx],
                tgt_lens[s_idx:e_idx])
            self.batches.append(batch)
            s_idx = e_idx

        self.indices = list(range(len(self.batches)))

    def __len__(self):
        return len(self.batches)

    def prepare_one_batch(self, src, tgt, src_lens, tgt_lens):
        num_sents = len(src)
        max_src_len = max(src_lens)
        max_tgt_len = max(tgt_lens)

        src_batch = np.zeros([num_sents, max_src_len], dtype=np.int32)
        tgt_batch = np.zeros([num_sents, max_tgt_len], dtype=np.int32)
        target_batch = np.zeros([num_sents, max_tgt_len], dtype=np.int32)

        for i in range(num_sents):
            src_batch[i] = src[i] + (max_src_len - src_lens[i]) * [ac.PAD_ID]
            tgt_batch[i] = tgt[i] + (max_tgt_len - tgt_lens[i]) * [ac.PAD_ID]
            target_batch[i] = tgt[i][1:] + [ac.EOS_ID] + (max_tgt_len - tgt_lens[i]) * [ac.PAD_ID]

        src_batch = torch.from_numpy(src_batch).type(torch.long)
        tgt_batch = torch.from_numpy(tgt_batch).type(torch.long)
        target_batch = torch.from_numpy(target_batch).type(torch.long)
        return src_batch, tgt_batch, target_batch

    def get_iter(self, shuffle=False):
        if shuffle:
            np.random.shuffle(self.indices)

        for idx in self.indices:
            yield self.batches[idx]
