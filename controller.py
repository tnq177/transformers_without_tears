import time
import numpy as np
import torch
import all_constants as ac
import utils as ut


class Controller(object):
    def __init__(self, args, model, data_manager, io):
        super(Controller, self).__init__()
        self.args = args
        self.model = model
        self.data_manager = data_manager
        self.io = io
        self.logger = args.logger

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # learning rate
        self.lr = args.lr
        self.lr_scale = args.lr_scale
        self.lr_decay = args.lr_decay
        self.lr_scheduler = args.lr_scheduler
        self.warmup_steps = args.warmup_steps
        # heuristic
        self.stop_lr = args.stop_lr
        self.patience = args.patience
        self.eval_metric = args.eval_metric
        # others
        self.epoch_size = args.epoch_size
        self.max_epochs = args.max_epochs

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # logging
        self.log_freq = args.log_freq
        self.pairs = args.pairs.split(',')
        self.stats = {
            'words': 0.,
            'time': 0.,
            'avg_smppls': [],
            'avg_ppls': [],
            'avg_bleus': [],
            'gnorms': [],
            'step': 0.
        }
        for pair in self.pairs:
            self.stats[pair] = {
                'log_loss': 0.,
                'log_nll_loss': 0.,
                'log_weight': 0.,
                'epoch_loss': 0.,
                'epoch_nll_loss': 0.,
                'epoch_weight': 0.,
                'train_smppls': [],
                'train_ppls': [],
                'dev_smppls': [],
                'dev_ppls': [],
                'dev_bleus': []
            }

    def train(self):
        # load data
        self.data_manager.load_data()
        for epoch_num in range(1, self.max_epochs + 1):
            for batch_num in range(1, self.epoch_size + 1):
                self.run_log(batch_num, epoch_num)

            self.report_epoch(epoch_num)
            self.eval_and_decay()

            if self.lr_scheduler == ac.NO_WU:
                cond = self.lr < self.stop_lr
            else:
                # with warmup
                cond = self.stats['step'] > self.warmup_steps and self.lr < self.stop_lr
            if cond:
                self.logger.info(f'lr = {self.lr:1.2e} <= stop_lr = {self.stop_lr:1.2e}. Stop training.')
                break

        self.logger.info('XONGGGGGGG!!! FINISHEDDDD!!!')
        self.io.save_train_stats(self.stats)

        self.logger.info('All pairs avg smppls:')
        self.logger.info(self.stats['avg_smppls'])
        self.logger.info('All pairs avg ppls:')
        self.logger.info(self.stats['avg_ppls'])
        self.logger.info('All pairs avg BLEUs:')
        self.logger.info(self.stats['avg_bleus'])
        for pair in self.pairs:
            self.logger.info('{}:'.format(pair.upper()))
            self.logger.info('--> train_smppls: {}'.format(','.join(map(str, self.stats[pair]['train_smppls']))))
            self.logger.info('--> train_ppls:   {}'.format(','.join(map(str, self.stats[pair]['train_smppls']))))
            self.logger.info('--> dev_smppls:   {}'.format(','.join(map(str, self.stats[pair]['dev_smppls']))))
            self.logger.info('--> dev_ppls:     {}'.format(','.join(map(str, self.stats[pair]['dev_ppls']))))
            self.logger.info('--> dev_bleus:    {}'.format(','.join(map(str, self.stats[pair]['dev_bleus']))))

    def run_log(self, batch_num, epoch_num):
        start = time.time()

        batch_data = self.data_manager.get_batch()
        src = batch_data['src']
        tgt = batch_data['tgt']
        targets = batch_data['targets']
        src_lang_idx = batch_data['src_lang_idx']
        tgt_lang_idx = batch_data['tgt_lang_idx']
        pair = batch_data['pair']
        logit_mask = batch_data['logit_mask']

        # zero grads
        self.optimizer.zero_grad()
        # move data to GPU
        src_cuda = src.to(self.device)
        tgt_cuda = tgt.to(self.device, non_blocking=True)
        targets_cuda = targets.to(self.device, non_blocking=True)
        logit_mask_cuda = logit_mask.to(self.device, non_blocking=True)
        # run
        ret = self.model(src_cuda, tgt_cuda, targets_cuda, src_lang_idx, tgt_lang_idx, logit_mask_cuda)
        opt_loss = ret['opt_loss']
        # back-prob
        opt_loss.backward()
        # clip grad before update
        if self.args.clip_grad > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
        else:
            gnorm = -1.

        # adjust lr before update
        self.adjust_lr()
        # actually update
        self.optimizer.step()

        num_words = ret['num_words'].item()
        loss = ret['loss'].item()
        nll_loss = ret['nll_loss'].item()

        # update stats
        self.stats['step'] += 1.
        self.stats['words'] += num_words
        self.stats['time'] += time.time() - start
        self.stats['gnorms'].append(gnorm)

        # per pair stats
        self.stats[pair]['log_loss'] += loss
        self.stats[pair]['log_nll_loss'] += nll_loss
        self.stats[pair]['log_weight'] += num_words

        self.stats[pair]['epoch_loss'] += loss
        self.stats[pair]['epoch_nll_loss'] += nll_loss
        self.stats[pair]['epoch_weight'] += num_words

        # write to logger every now and then
        if batch_num % self.log_freq == 0:
            self.logger.info(f'Batch {batch_num}/{self.epoch_size}, epoch {epoch_num}/{self.max_epochs}')

            speed = self.stats['words'] / self.stats['time']
            gnorm = self.stats['gnorms'][-1]
            self.logger.info(f'    lr = {self.lr:1.2e}')
            self.logger.info(f'    gnorm = {self.stats["gnorms"][-1]:.2f}')
            self.logger.info(f'    wps = {speed:.2f}')

            # per pair
            for pair in self.pairs:
                if self.stats[pair]['log_weight'] <= 0:
                    continue

                smppl = self.stats[pair]['log_loss'] / self.stats[pair]['log_weight']
                smppl = np.exp(smppl) if smppl < 300 else 1e9
                ppl = self.stats[pair]['log_nll_loss'] / self.stats[pair]['log_weight']
                ppl = np.exp(ppl) if ppl < 300 else 1e9
                self.logger.info(f'    {pair}: smppl = {smppl:.3f}, ppl = {ppl:.3f}')

                self.stats[pair]['log_loss'] = 0.
                self.stats[pair]['log_nll_loss'] = 0.
                self.stats[pair]['log_weight'] = 0.

    def adjust_lr(self):
        if self.lr_scheduler == ac.NO_WU:
            return

        step = self.stats['step'] + 1.
        embed_dim = self.args.embed_dim
        if step < self.warmup_steps:
            # both UPFLAT_WU and ORG_WU follows this lr during warmup
            self.lr = self.lr_scale * embed_dim ** -0.5 * step * self.warmup_steps ** -1.5
        elif self.lr_scheduler == ac.ORG_WU:
            # only ORG_WU decays lr with this formula. UPFLAT_WU decays like NO_WU
            self.lr = self.lr_scale * (embed_dim * step) ** -0.5

        for p in self.optimizer.param_groups:
            p['lr'] = self.lr

    def report_epoch(self, epoch_num):
        self.logger.info(f'Finish epoch {epoch_num}')

        speed = self.stats['words'] / self.stats['time']
        self.stats['words'] = 0.
        self.stats['time'] = 0.
        self.logger.info(f'    wps = {speed:.2f}')

        for pair in self.pairs:
            if self.stats[pair]['epoch_weight'] <= 0:
                smppl = 1e9
                ppl = 1e9
            else:
                smppl = self.stats[pair]['epoch_loss'] / self.stats[pair]['epoch_weight']
                ppl = self.stats[pair]['epoch_nll_loss'] / self.stats[pair]['epoch_weight']
                smppl = np.exp(smppl) if smppl < 300 else 1e9
                ppl = np.exp(ppl) if ppl < 300 else 1e9

            self.stats[pair]['train_smppls'].append(smppl)
            self.stats[pair]['train_ppls'].append(ppl)
            self.stats[pair]['epoch_loss'] = 0.
            self.stats[pair]['epoch_nll_loss'] = 0.
            self.stats[pair]['epoch_weight'] = 0.
            self.logger.info(f'    {pair}: train_smppl={smppl:.3f}, train_ppl={ppl:.3f}')

    def eval_and_decay(self):
        self.eval_ppl()
        self.eval_bleu()

        # save current ckpt
        self.io.save_current_ckpt(self.model.state_dict())

        # save per-language-pair best ckpt
        for pair in self.pairs:
            self.io.update_best_ckpt(self.model.state_dict(), pair)

        # save all-language-pair best ckpt
        self.io.update_best_ckpt(self.model.state_dict())

        # it's we do warmup and it's still in warmup phase, don't anneal
        if self.lr_scheduler == ac.ORG_WU or self.lr_scheduler == ac.UPFLAT_WU and self.stats['step'] < self.warm_steps:
            return

        # we decay learning rate wrt avg_bleu (or a different evaluation metric)
        if self.eval_metric == ac.DEV_BLEU:
            stat = 'avg_bleus'
        elif self.eval_metric == ac.DEV_PPL:
            stat = 'avg_ppls'
        else:
            stat = 'avg_smppls'

        cond1 = len(self.stats[stat]) > self.patience
        if self.eval_metric == ac.DEV_BLEU:
            cond = cond1 and self.stats[stat][-1] < min(self.stats[stat][-1 - self.patience: -1])
        else:
            cond = cond1 and self.stats[stat][-1] > max(self.stats[stat][-1 - self.patience: -1])
        
        if cond:
            past_stats = self.stats[stat][-1 - self.patience:]
            past_stats = map(str, past_stats)
            past_stats = ','.join(past_stats)
            self.logger.info(f'Stat is {stat}, past numbers are {past_stats}')
            self.logger.info(f'Anneal lr from {self.lr} to {self.lr * self.lr_decay}')
            self.lr = self.lr * self.lr_decay
            for p in self.optimizer.param_groups:
                p['lr'] = self.lr

    def eval_ppl(self):
        self.logger.info('Evaluate dev perplexity')
        start = time.time()
        self.model.eval()

        avg_smppls = []
        avg_ppls = []
        with torch.no_grad():
            for pair in self.pairs:
                src_lang, tgt_lang = pair.split('2')
                src_lang_idx = self.data_manager.lang_vocab[src_lang]
                tgt_lang_idx = self.data_manager.lang_vocab[tgt_lang]
                loss = 0.
                nll_loss = 0.
                weight = 0.

                it = self.data_manager.data[pair][ac.DEV].get_iter()
                for src, tgt, targets in it:
                    src_cuda = src.to(self.device)
                    tgt_cuda = tgt.to(self.device)
                    targets_cuda = targets.to(self.device)
                    logit_mask_cuda = self.data_manager.logit_masks[tgt_lang].to(self.device)

                    ret = self.model(src_cuda, tgt_cuda, targets_cuda, src_lang_idx, tgt_lang_idx, logit_mask_cuda)
                    loss += ret['loss'].item()
                    nll_loss += ret['nll_loss'].item()
                    weight += ret['num_words'].item()

                smppl = loss / weight
                smppl = np.exp(smppl) if smppl < 300 else 1e9
                ppl = nll_loss / weight
                ppl = np.exp(ppl) if ppl < 300 else 1e9
                avg_smppls.append(smppl)
                avg_ppls.append(ppl)
                self.io.save_score(ac.DEV_SMPPL, smppl, pair)
                self.io.save_score(ac.DEV_PPL, ppl, pair)
                self.stats[pair]['dev_smppls'].append(smppl)
                self.stats[pair]['dev_ppls'].append(ppl)
                self.logger.info(f'    {pair}: dev_smppl={smppl:.3f}, dev_ppl={ppl:.3f}')

        avg_smppl = sum(avg_smppls) / len(avg_smppls)
        avg_ppl = sum(avg_ppls) / len(avg_ppls)
        self.io.save_score(ac.DEV_SMPPL, avg_smppl)
        self.io.save_score(ac.DEV_PPL, avg_ppl)
        self.stats['avg_smppls'].append(avg_smppls)
        self.stats['avg_ppls'].append(avg_ppl)
        self.logger.info(f'Done evaluating dev ppl, it takes {int(time.time() - start)} seconds')
        self.model.train()

    def eval_bleu(self):
        self.logger.info('Evaluate dev BLEU')
        start = time.time()
        self.model.eval()
        avg_bleus = []
        with torch.no_grad():
            for pair in self.pairs:
                self.logger.info(f'--> {pair}')
                all_best_trans, all_beam_trans = self.translate(pair, ac.DEV)
                bleu = self.io.print_dev_translations_and_calculate_BLEU(pair, all_best_trans, all_beam_trans)
                avg_bleus.append(bleu)
                self.stats[pair]['dev_bleus'].append(bleu)
        avg_bleu = sum(avg_bleus) / len(avg_bleus)
        self.io.save_score(ac.DEV_BLEU, avg_bleu)
        self.stats['avg_bleus'].append(avg_bleu)
        self.logger.info(f'avg_bleu = {avg_bleu}')
        self.logger.info(f'Done evaluating dev BLEU, it takes {ut.format_seconds(time.time() - start)} seconds')

    def get_trans(self, probs, scores, symbols):
        def ids_to_trans(trans_ids):
            words = []
            for idx in trans_ids:
                if idx == ac.EOS_ID:
                    break
                words.append(self.data_manager.ivocab[idx])

            return words

        # if beam search, we want scores sorted in score order.
        # if sampling, better to have them unsorted
        if self.args.decode_method == ac.BEAM_SEARCH:
            sorted_rows = np.argsort(scores)[::-1]
        else:
            sorted_rows = range(scores.shape[0])
        best_trans = None
        beam_trans = []
        best_score = float('-inf')
        for i, r in enumerate(sorted_rows):
            trans_ids = symbols[r]
            trans_out = ids_to_trans(trans_ids)
            beam_trans.append([trans_out, scores[r], probs[r]])
            if scores[r] > best_score: # highest prob trans
                best_trans = trans_out
                best_score = scores[r]

        return best_score, best_trans, beam_trans

    # when sampling, allows you to request an arbitrarily large number of
    # samples per line, by duplicating batches to avoid out-of-memory errors
    def split_batch(self, src):
        batch_size = src.size(0)
        beam_size = self.args.beam_size
        max_beams = self.args.max_parallel_beams
        if (self.args.decode_method == ac.BEAM_SEARCH) or \
           (max_beams == 0) or \
           (batch_size * beam_size <= max_beams):
            return ([src], 1, beam_size)

        sent_length = src.size(1)
        srcs = []
        i = 0

        # TODO: remove unnecessary padding
        curr = torch.zeros(0,sent_length).type(torch.long)
        space_left = max_beams
        beams_left = beam_size

        while i < batch_size:
            num_copies = min(space_left, beams_left)
            copies = src[i].unsqueeze(0).expand(num_copies, -1)
            curr = torch.cat((curr, copies), 0)

            space_left -= num_copies
            if space_left == 0:
                srcs.append(curr)
                curr = torch.zeros(0,sent_length).type(torch.long)
                space_left = max_beams

            beams_left -= num_copies
            if beams_left == 0:
                i += 1
                beams_left = beam_size

        if curr.size(0) > 0:
            srcs.append(curr)

        return (srcs, beam_size, 1)

    def _translate(self, src_batches, sorted_idxs, src_lang_idx, tgt_lang_idx, logit_mask):
        all_best_trans = [[] for i in range(sorted_idxs.shape[0])]
        all_beam_trans = [[] for i in range(sorted_idxs.shape[0])]

        start = time.time()
        count = 0
        saved_best_score = float('-inf')
        self.model.eval()
        with torch.no_grad():
            for orig_src in src_batches:
                (new_batches, inc, beam_size) = self.split_batch(orig_src)
                mini_count = 0
                for src in new_batches:
                    src_cuda = src.to(self.device)
                    logit_mask = logit_mask.to(self.device)
                    ret = self.model.beam_decode(src_cuda, src_lang_idx, tgt_lang_idx, logit_mask, beam_size)
                    for x in ret:
                        probs = x['probs'].cpu().detach().numpy().reshape([-1])
                        scores = x['scores'].cpu().detach().numpy().reshape([-1])
                        symbols = x['symbols'].cpu().detach().numpy()

                        best_score, best_trans, beam_trans = self.get_trans(probs, scores, symbols)
                        if best_score > saved_best_score:
                            all_best_trans[sorted_idxs[count]] = best_trans
                            saved_best_score = best_score
                        all_beam_trans[sorted_idxs[count]].extend(beam_trans)

                        mini_count += 1
                        if mini_count == inc:
                            mini_count = 0
                            count += 1
                            saved_best_score = float('-inf')
                            if count % 100 == 0:
                                self.logger.info(f'   Translating line {count}, avg {count / (time.time() - start):.2f} sents/second')

        self.model.train()

        return all_best_trans, all_beam_trans

    def translate(self, pair, mode, input_file=None):
        src_lang, tgt_lang = pair.split('2')

        src_lang_idx = self.data_manager.lang_vocab[src_lang]
        tgt_lang_idx = self.data_manager.lang_vocab[tgt_lang]
        logit_mask = self.data_manager.logit_masks[tgt_lang]
        if mode == ac.DEV:
            data = self.data_manager.translate_data[pair]
            src_batches = data['src_batches']
            sorted_idxs = data['sorted_idxs']
        else:
            src_batches, sorted_idxs = self.data_manager.get_translate_batches(pair, mode, input_file=input_file)

        return self._translate(src_batches, sorted_idxs, src_lang_idx, tgt_lang_idx, logit_mask)
