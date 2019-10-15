import os
import shutil
import pickle
import time
import numpy as np
import torch
from os.path import join
import all_constants as ac
import utils as ut

if torch.cuda.is_available():
    torch.cuda.manual_seed(ac.SEED)
else:
    torch.manual_seed(ac.SEED)


class Controller(object):
    def __init__(self, args, model, data_manager):
        super(Controller, self).__init__()
        self.args = args
        self.model = model
        self.data_manager = data_manager
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
                self.logger.info('lr = {0:1.2e} <= stop_lr = {0:1.2e}. Stop training.'.format(self.lr, self.stop_lr))
                break

        self.logger.info('XONGGGGGGG!!! FINISHEDDDD!!!')
        train_stats_file = join(self.args.dump_dir, 'train_stats.pkl')
        self.logger.info('Dump stats to {}'.format(train_stats_file))
        open(train_stats_file, 'w').close()
        with open(train_stats_file, 'wb') as fout:
            pickle.dump(self.stats, fout)

        self.logger.info('All pairs avg BLEUs:')
        self.logger.info(self.stats['avg_bleus'])
        for pair in self.pairs:
            self.logger.info('{}:'.format(pair.upper()))
            self.logger.info('--> train_smppls: {}'.format(','.join(map(str, self.stats[pair]['train_smppls']))))
            self.logger.info('--> train_ppls:   {}'.format(','.join(map(str, self.stats[pair]['train_smppls']))))
            self.logger.info('--> dev_smppls:   {}'.format(','.join(map(str, self.stats[pair]['dev_smppls']))))
            self.logger.info('--> dev_ppls:     {}'.format(','.join(map(str, self.stats[pair]['dev_ppls']))))
            self.logger.info('--> dev_bleus:    {}'.format(','.join(map(str, self.stats[pair]['dev_bleus']))))

        # translate test
        self.logger.info('Translating test file')
        for pair in self.pairs:
            # Load best ckpt
            best_score = max(self.stats[pair]['dev_bleus'])
            ckpt_file = join(self.args.dump_dir, '{}-{}.pth'.format(pair, best_score))
            self.logger.info('Reload {}'.format(ckpt_file))
            self.model.load_state_dict(torch.load(ckpt_file))

            src_lang, tgt_lang = pair.split('2')
            test_file = join(self.args.data_dir, '{}/test.{}.bpe'.format(pair, src_lang))
            self.translate(test_file, src_lang, tgt_lang)

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
            self.logger.info('Batch {}/{}, epoch {}/{}'.format(batch_num, self.epoch_size, epoch_num, self.max_epochs))

            speed = self.stats['words'] / self.stats['time']
            gnorm = self.stats['gnorms'][-1]
            self.logger.info('    lr = {0:1.2e}'.format(self.lr))
            self.logger.info('    gnorm = {:.2f}'.format(self.stats['gnorms'][-1]))
            self.logger.info('    wps = {:.2f}'.format(speed))

            # per pair
            for pair in self.pairs:
                if self.stats[pair]['log_weight'] <= 0:
                    continue

                smppl = self.stats[pair]['log_loss'] / self.stats[pair]['log_weight']
                smppl = np.exp(smppl) if smppl < 300 else 1e9
                ppl = self.stats[pair]['log_nll_loss'] / self.stats[pair]['log_weight']
                ppl = np.exp(ppl) if ppl < 300 else 1e9
                self.logger.info('    {}: smppl = {:.3f}, ppl = {:.3f}'.format(pair, smppl, ppl))

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
        self.logger.info('Finish epoch {}'.format(epoch_num))

        speed = self.stats['words'] / self.stats['time']
        self.stats['words'] = 0.
        self.stats['time'] = 0.
        self.logger.info('    wps = {:.2f}'.format(speed))

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
            self.logger.info('    {}: train_smppl={:.3f}, train_ppl={:.3f}'.format(pair, smppl, ppl))

    def eval_and_decay(self):
        self.eval_ppl()
        self.eval_bleu()

        # save current ckpt
        self.save_ckpt('model')

        # save per-language-pair best ckpt
        for pair in self.pairs:
            if self.stats[pair]['dev_bleus'][-1] == max(self.stats[pair]['dev_bleus']):
                # remove previous best ckpt
                if len(self.stats[pair]['dev_bleus']) > 1:
                    self.remove_ckpt(pair, max(self.stats[pair]['dev_bleus'][:-1]))
                self.save_ckpt(pair, self.stats[pair]['dev_bleus'][-1])

        # save all-language-pair best ckpt
        if self.stats['avg_bleus'][-1] == max(self.stats['avg_bleus']):
            # remove previous best ckpt
            if len(self.stats['avg_bleus']) > 1:
                self.remove_ckpt('model', max(self.stats['avg_bleus'][:-1]))
            self.save_ckpt('model', self.stats['avg_bleus'][-1])

        # it's we do warmup and it's still in warmup phase, don't anneal
        if self.lr_scheduler == ac.ORG_WU or self.lr_scheduler == ac.UPFLAT_WU and self.stats['step'] < self.warm_steps:
            return

        # we decay learning rate wrt avg_bleu
        cond = len(self.stats['avg_bleus']) > self.patience and self.stats['avg_bleus'][-1] < min(self.stats['avg_bleus'][-1 - self.patience: -1])
        if cond:
            past_bleus = self.stats['avg_bleus'][-1 - self.patience:]
            past_bleus = map(str, past_bleus)
            past_bleus = ','.join(past_bleus)
            self.logger.info('Past BLEUs are {}'.format(past_bleus))
            self.logger.info('Anneal lr from {} to {}'.format(self.lr, self.lr * self.lr_decay))
            self.lr = self.lr * self.lr_decay
            for p in self.optimizer.param_groups:
                p['lr'] = self.lr

    def save_ckpt(self, model_name, score=None):
        dump_dir = self.args.dump_dir
        if score is None:
            ckpt_path = join(dump_dir, '{}.pth'.format(model_name))
            self.logger.info('Save current ckpt to {}'.format(ckpt_path))
        else:
            ckpt_path = join(dump_dir, '{}-{}.pth'.format(model_name, score))
            self.logger.info('Save best ckpt for {} to {}'.format(model_name, ckpt_path))

        torch.save(self.model.state_dict(), ckpt_path)

    def remove_ckpt(self, model_name, score):
        # never remove current ckpt so always ask for score
        ckpt_path = join(self.args.dump_dir, '{}-{}.pth'.format(model_name, score))
        if os.path.exists(ckpt_path):
            self.logger.info('rm {}'.format(ckpt_path))
            os.remove(ckpt_path)

    def eval_ppl(self):
        self.logger.info('Evaluate dev perplexity')
        start = time.time()
        self.model.eval()

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
                self.stats[pair]['dev_smppls'].append(smppl)
                self.stats[pair]['dev_ppls'].append(ppl)
                self.logger.info('    {}: dev_smppl={:.3f}, dev_ppl={:.3f}'.format(pair, smppl, ppl))

        self.logger.info('It takes {} seconds'.format(int(time.time() - start)))
        self.model.train()

    def eval_bleu(self):
        self.logger.info('Evaluate dev BLEU')
        start = time.time()
        self.model.eval()
        avg_bleus = []
        dump_dir = self.args.dump_dir
        with torch.no_grad():
            for pair in self.pairs:
                self.logger.info('--> {}'.format(pair))
                src_lang, tgt_lang = pair.split('2')
                src_lang_idx = self.data_manager.lang_vocab[src_lang]
                tgt_lang_idx = self.data_manager.lang_vocab[tgt_lang]
                logit_mask = self.data_manager.logit_masks[tgt_lang]
                data = self.data_manager.translate_data[pair]
                src_batches = data['src_batches']
                sorted_idxs = data['sorted_idxs']
                ref_file = data['ref_file']

                all_best_trans, all_beam_trans = self._translate(src_batches, sorted_idxs, src_lang_idx, tgt_lang_idx, logit_mask)

                all_best_trans = ''.join(all_best_trans)
                best_trans_file = join(dump_dir, '{}_val_trans.txt.bpe'.format(pair))
                open(best_trans_file, 'w').close()
                with open(best_trans_file, 'w') as fout:
                    fout.write(all_best_trans)

                all_beam_trans = ''.join(all_beam_trans)
                beam_trans_file = join(dump_dir, '{}_beam_trans.txt.bpe'.format(pair))
                open(beam_trans_file, 'w').close()
                with open(beam_trans_file, 'w') as fout:
                    fout.write(all_beam_trans)

                # merge BPE
                nobpe_best_trans_file = join(dump_dir, '{}_val_trans.txt'.format(pair))
                ut.remove_bpe(best_trans_file, nobpe_best_trans_file)
                nobpe_beam_trans_file = join(dump_dir, '{}_beam_trans.txt'.format(pair))
                ut.remove_bpe(beam_trans_file, nobpe_beam_trans_file)

                # calculate BLEU
                bleu, msg = ut.calc_bleu(self.args.bleu_script, nobpe_best_trans_file, ref_file)
                self.logger.info(msg)
                avg_bleus.append(bleu)
                self.stats[pair]['dev_bleus'].append(bleu)

                # save translation with BLEU score for future reference
                trans_file = '{}-{}'.format(nobpe_best_trans_file, bleu)
                shutil.copyfile(nobpe_best_trans_file, trans_file)
                beam_file = '{}-{}'.format(nobpe_beam_trans_file, bleu)
                shutil.copyfile(nobpe_beam_trans_file, beam_file)

        avg_bleu = sum(avg_bleus) / len(avg_bleus)
        self.stats['avg_bleus'].append(avg_bleu)
        self.logger.info('avg_bleu = {}'.format(avg_bleu))
        self.logger.info('Done evaluating dev BLEU, it takes {} seconds'.format(ut.format_seconds(time.time() - start)))

    def get_trans(self, probs, scores, symbols):
        def ids_to_trans(trans_ids):
            words = []
            for idx in trans_ids:
                if idx == ac.EOS_ID:
                    break
                words.append(self.data_manager.ivocab[idx])

            return ' '.join(words)

        sorted_rows = np.argsort(scores)[::-1]
        best_trans = None
        beam_trans = []
        for i, r in enumerate(sorted_rows):
            trans_ids = symbols[r]
            trans_out = ids_to_trans(trans_ids)
            beam_trans.append('{} ||| {:.3f} {:.3f}'.format(trans_out, scores[r], probs[r]))
            if i == 0: # highest prob trans
                best_trans = trans_out

        return best_trans, '\n'.join(beam_trans)

    def _translate(self, src_batches, sorted_idxs, src_lang_idx, tgt_lang_idx, logit_mask):
        all_best_trans = [''] * sorted_idxs.shape[0]
        all_beam_trans = [''] * sorted_idxs.shape[0]

        start = time.time()
        count = 0
        self.model.eval()
        with torch.no_grad():
            for src in src_batches:
                src_cuda = src.to(self.device)
                logit_mask = logit_mask.to(self.device)
                ret = self.model.beam_decode(src_cuda, src_lang_idx, tgt_lang_idx, logit_mask)
                for x in ret:
                    probs = x['probs'].cpu().detach().numpy().reshape([-1])
                    scores = x['scores'].cpu().detach().numpy().reshape([-1])
                    symbols = x['symbols'].cpu().detach().numpy()

                    best_trans, beam_trans = self.get_trans(probs, scores, symbols)
                    all_best_trans[sorted_idxs[count]] = best_trans + '\n'
                    all_beam_trans[sorted_idxs[count]] = beam_trans + '\n\n'

                    count += 1
                    if count % 100 == 0:
                        self.logger.info('   Translaing line {}, avg {:.2f} sents/second'.format(count, count / (time.time() - start)))

        self.model.train()

        return all_best_trans, all_beam_trans

    def translate(self, src_file, src_lang, tgt_lang, batch_size=4096):
        src_batches, sorted_idxs = self.data_manager.get_translate_batches(src_file, batch_size=batch_size)
        src_lang_idx = self.data_manager.lang_vocab[src_lang]
        tgt_lang_idx = self.data_manager.lang_vocab[tgt_lang]
        logit_mask = self.data_manager.logit_masks[tgt_lang]
        all_best_trans, all_beam_trans = self._translate(src_batches, sorted_idxs, src_lang_idx, tgt_lang_idx, logit_mask)

        # write to file
        all_best_trans = ''.join(all_best_trans)
        best_trans_file = src_file + '.best_trans'
        open(best_trans_file, 'w').close()
        with open(best_trans_file, 'w') as fout:
            fout.write(all_best_trans)

        all_beam_trans = ''.join(all_beam_trans)
        beam_trans_file = src_file + '.beam_trans'
        open(beam_trans_file, 'w').close()
        with open(beam_trans_file, 'w') as fout:
            fout.write(all_beam_trans)

        self.logger.info('Finish decode {}'.format(src_file))
        self.logger.info('Best --> {}'.format(best_trans_file))
        self.logger.info('Beam --> {}'.format(beam_trans_file))
