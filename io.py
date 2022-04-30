import re
import logging
from os.path import join, exists
from subprocess import Popen, PIPE
import shutil
import pickle
import numpy as np
import torch
import all_constants as ac

class IO:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.dump_dir = args.dump_dir
        self.bleu_script = args.bleu_script
        if not exists(args.bleu_script):
            raise ValueError(f'Bleu script not found at {self.bleu_script}')
        
        # vocab files
        self.vocab_file      = join(self.data_dir, 'vocab.joint')
        self.lang_vocab_file = join(self.data_dir, 'lang.vocab')
        if not exists(self.vocab_file):
            raise ValueError(f'Vocab file not found at {self.vocab_file}')
        if not exists(self.lang_vocab_file):
            raise ValueError(f'Vocab file not found at {self.lang_vocab_file}')

        lang_vocab, _ = self.load_lang_vocab()
        self.langs = lang_vocab.keys()        
        self.pairs = args.pairs.split(',')
        
        # data files
        self.data_files = _construct_data_filenames()
        
        # dump files
        dump_dir = args.dump_dir
        Popen('mkdir -p %s' % dump_dir, shell=True).wait()
        
        self.logfile          = join(self.dump_dir, 'DEBUG.log')
        self.train_stats_file = join(self.dump_dir, 'train_stats.pkl')
        
        self.dev_bleus = {}
        for pair in self.pairs:
            self.dev_bleus[pair] = []
        self.dev_bleus['all'] = []
        
    def _construct_data_filenames(self):
        data_dir = self.data_dir
        data_files = {}

        for lang in self.langs:
            data_files[lang] = {}
            mask_file = join(data_dir, f'mask.{lang}.npy')
            if not exists(mask_file):
                raise ValueError(f'Mask file for {pair} not found at {mask_file}')
            data_files[lang]['mask'] = mask_file
        
        for pair in self.pairs:
            src_lang, tgt_lang = pair.split('2')
            data_files[pair] = {}
            for mode in [ac.TRAIN, ac.DEV, ac.TEST]:
                data_files[pair][mode] = {}
                data_files[pair][mode]['src_orig'] = join(data_dir, f'{pair}/{mode}.{src_lang}')
                data_files[pair][mode]['tgt_orig'] = join(data_dir, f'{pair}/{mode}.{tgt_lang}')
                data_files[pair][mode]['src_bpe'] = join(data_dir, f'{pair}/{mode}.{src_lang}.bpe')
                data_files[pair][mode]['tgt_bpe'] = join(data_dir, f'{pair}/{mode}.{tgt_lang}.bpe')
                if mode != ac.TEST:
                    data_files[pair][mode]['src_npy'] = join(data_dir, f'{pair}/{mode}.{src_lang}.npy')
                    data_files[pair][mode]['tgt_npy'] = join(data_dir, f'{pair}/{mode}.{tgt_lang}.npy')
                for filename in data_files[pair][mode]:
                    if not exists(filename):
                        raise ValueError(f'Data file for {pair} not found at {mask_file}')

        return data_files
        
    def _construct_ckpt_path(self, pair, score):
        dump_dir = self.dump_dir
        if pair is None:
            ckpt_path = join(dump_dir, 'model.pth')
        else:
            if score is None:
                ckpt_path = join(dump_dir, f'{pair}.pth')
            else:
                ckpt_path = join(dump_dir, f'{pair}-{score}.pth')
        return ckpt_path
    
    def _construct_dev_trans_path(self, pair, best, bpe, score=None):
        dump_dir = self.dump_dir
        if bpe:
            if best:
                trans_path = join(dump_dir, f'{pair}_val_trans.txt.bpe')
            else:
                trans_path = join(dump_dir, f'{pair}_beam_trans.txt.bpe')
        else:
            if best:
                trans_path = join(dump_dir, f'{pair}_val_trans.txt')
            else:
                trans_path = join(dump_dir, f'{pair}_beam_trans.txt')
        if score is not None:
            trans_path += f'-{score}'
        return trans_path
    
    def _construct_test_trans_path(self, pair, best, input_file):
        src_file = input_file if input_file else data_files[pair][mode]['src_orig']
        if best:
            return src_file + '.best_trans'
        else:
            return src_file + '.beam_trans'

    def get_logger(self):
        """Global logger for every logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s:%(filename)s:%(lineno)s: %(message)s', '%m/%d %H:%M:%S')
        if not logger.handlers:
            debug_handler = logging.FileHandler(self.logfile)
            debug_handler.setFormatter(formatter)
            debug_handler.setLevel(logging.DEBUG)
            logger.addHandler(debug_handler)

        self.logger = logger
        return logger

    def _init_vocab(self, vocab_file):
        vocab = {}
        ivocab = {}
        with open(vocab_file, 'r') as f:
            for line in f:
                temp = line.strip().split()
                if temp:
                    vocab[temp[0]] = int(temp[1])
                    ivocab[int(temp[1])] = temp[0]

        return vocab, ivocab

    def load_vocab(self):
        return self._init_vocab(self.vocab_file)

    def load_lang_vocab(self):
        return self._init_vocab(self.lang_vocab_file)
    
    def load_logit_mask(self, lang):
        mask_file = self.data_files[lang]['mask']
        mask = np.load(mask_file, allow_pickle=True)
        return torch.from_numpy(mask)
    
    def load_bpe_data(self, pair, mode, src, input_file=None):
        if input_file:
            bpe_file = input_file
        else:
            if src:
                bpe_file = self.data_files[pair][mode]['src_bpe']
            else:
                bpe_file = self.data_files[pair][mode]['tgt_bpe']

        data = []
        with open(bpe_file, 'r') as fin:
            for line in fin:
                toks = line.strip().split()
                if toks:
                    data.append(toks)
        
        return data
    
    def load_npy_data(self, pair, mode, src):
        if src:
            npy_file = self.data_files[pair][mode]['src_npy']
        else:
            npy_file = self.data_files[pair][mode]['tgt_npy']
        return np.load(npy_file, allow_pickle=True)

    def save_train_stats(self, stats):
        train_stats_file = self.train_stats_file
        self.logger.info(f'Dump stats to {train_stats_file}')
        open(train_stats_file, 'w').close()
        with open(train_stats_file, 'wb') as fout:
            pickle.dump(stats, fout)

    def _load_ckpt(self, pair, score):
        ckpt_path = self._construct_ckpt_path(pair, score)
        self.logger.info(f'Reload {ckpt_path}')
        return torch.load(ckpt_path)

    def _save_ckpt(self, state_dict, pair=None, score=None):
        ckpt_path = self._construct_ckpt_path(pair, score)
        if pair:
            self.logger.info(f'Save best ckpt for {pair} to {ckpt_path}')
        else:
            self.logger.info(f'Save current ckpt to {ckpt_path}')
        torch.save(state_dict, ckpt_path)
    
    def _remove_ckpt(self, pair, score):
        ckpt_path = self._construct_ckpt_path(pair, score)
        if os.path.exists(ckpt_path):
            self.logger.info('rm {}'.format(ckpt_path))
            os.remove(ckpt_path)

    def load_best_ckpt(self, pair):
        best_score = max(self.dev_bleus[pair])
        return self._load_ckpt(pair, best_score)

    def save_current_checkpoint(self, state_dict):
        self._save_ckpt(self, state_dict)

    def update_best_ckpt(self, state_dict, pair=None):
        pair_name = pair if pair else 'all'
        if self.dev_bleus[pair_name][-1] == max(self.dev_bleus[pair_name]):
            # remove previous best ckpt
            if len(self.dev_bleus[pair_name]) > 1:
                self._remove_ckpt(pair_name, max(self.dev_bleus[pair_name][:-1]))
            self._save_ckpt(state_dict, pair, self.dev_bleus[pair_name][-1])

    def _line_string(self, line):
        (words, score, prob) = line
        joined_words = ' '.join(words)
        return f'{joined_words} ||| {score:.3f} {prob:.3f}'
    
    def _print_best_trans(self, pair, best_trans, best_trans_file):
        best_trans_strings = []
        for line in best_trans:
            best_trans_strings.append(self._line_string(line))
        all_best_trans = '\n'.join(best_trans_strings)
        
        open(best_trans_file, 'w').close()
        with open(best_trans_file, 'w') as fout:
            fout.write(all_best_trans)        
    
    def _print_beam_trans(self, pair, beam_trans, beam_trans_file):
        new_beam_trans = []
        for beam in beam_trans:
            beam_strings = map(self._line_string, beam)
            beam_strings = '\n'.join(beam_strings)
            new_beam_trans.append(beam_strings)
        all_beam_trans = '\n\n'.join(new_beam_trans)

        open(beam_trans_file, 'w').close()
        with open(beam_trans_file, 'w') as fout:
            fout.write(all_beam_trans)

    def _remove_bpe(self, infile, outfile):
        open(outfile, 'w').close()
        Popen("sed -r 's/(@@ )|(@@ ?$)//g' < {} > {}".format(infile, outfile), shell=True, stdout=PIPE).communicate()

    def _calc_bleu(self, trans_file, ref_file):
        # compute BLEU
        multibleu_cmd = ['perl', self.bleu_script, ref_file, '<', trans_file]
        p = Popen(' '.join(multibleu_cmd), shell=True, stdout=PIPE)
        output, _ = p.communicate()
        output = output.decode('utf-8')
        msg = output + '\n'
        out_parse = re.match(r'BLEU = [-.0-9]+', output)

        bleu = 0.
        if out_parse is None:
            msg += '\n    Error extracting BLEU score, out_parse is None'
            msg += '\n    It is highly likely that your model just produces garbage.'
            msg += '\n    Be patient yo, it will get better.'
        else:
            bleu = float(out_parse.group()[6:])

        return bleu, msg

    def print_dev_translations_and_calculate_BLEU(self, pair, best_trans, beam_trans):
        # make filenames
        ref_file = self.data_files[pair][ac.DEV]['tgt_orig']
        bpe_best_trans_file = self._construct_dev_trans_path(pair, best=True, bpe=True)
        nobpe_best_trans_file = self._construct_dev_trans_path(pair, best=True, bpe=False)
        bpe_beam_trans_file = self._construct_dev_trans_path(pair, best=False, bpe=True)
        nobpe_beam_trans_file = self._construct_dev_trans_path(pair, best=False, bpe=False)

        # print the translations
        self._print_best_trans(pair, best_trans, bpe_best_trans_file)
        self._print_beam_trans(pair, beam_trans, bpe_beam_trans_file)
        
        # merge BPE
        self._remove_bpe(best_trans_file, nobpe_best_trans_file)
        self._remove_bpe(beam_trans_file, nobpe_beam_trans_file)
        
        # calculate BLEU
        bleu, msg = ut.calc_bleu(nobpe_best_trans_file, ref_file)
        self.logger.info(msg)
        self.dev_bleus[pair].append(bleu)

        # save translation with BLEU score for future reference
        final_best_trans_file = self._construct_dev_trans_path(pair, best=True, bpe=False, score=bleu)
        final_beam_trans_file = self._construct_dev_trans_path(pair, best=False, bpe=False, score=bleu)
        shutil.copyfile(nobpe_best_trans_file, final_best_trans_file)
        shutil.copyfile(nobpe_beam_trans_file, final_beam_trans_file)

        return bleu

    def save_avg_bleu_score(self, score):
        self.dev_bleus['all'].append(score)

    def print_test_translations(self, pair, best_trans, beam_trans, input_file=None):
        best_trans_file = self._construct_test_trans_path(pair, best=True, input_file)
        beam_trans_file = self._construct_test_trans_path(pair, best=False, input_file)
        
        self._print_best_trans(pair, best_trans, best_trans_file)
        self._print_beam_trans(pair, beam_trans, beam_trans_file)

        self.logger.info('Finish decode {}'.format(src_file))
        self.logger.info('Best --> {}'.format(best_trans_file))
        self.logger.info('Beam --> {}'.format(beam_trans_file))
