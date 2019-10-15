import argparse
from os.path import join, exists
from subprocess import Popen

import numpy as np
import torch

from controller import Controller
from data_manager import DataManager
from model import Transformer

import utils as ut
import configurations


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'translate'], default='train')
    parser.add_argument('--files-langs', type=str, nargs='+',
                        help='Used if do translate, format {file_path,src_lang,tgt_lang...}. e.g.: data/test.en,en,vi data/test.vi,vi,en')
    parser.add_argument('--model-file', type=str,
                        help='Used if do translate, path to checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--dump-dir', type=str, required=True,
                        help='Path to dump directory (store all stats, translations, checkpoints...)')
    parser.add_argument('--pairs', type=str, required=True,
                        help='Command separated list of pairs in format src2tgt, e.g. en2vi,hu2en,uz2en')
    parser.add_argument('--bleu-script', type=str, default='./scripts/multi-bleu.perl',
                        help='Path to multi-bleu.perl script')
    parser.add_argument('--log-freq', type=int, default=100,
                        help='How often do we log training progress (# of batches)')
    parser.add_argument('--config', type=str, required=True,
                        help='Model and training configuration, see configurations.py')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    config = getattr(configurations, args.config)()
    for k, v in config.items():
        setattr(args, k, v)

    if not exists(args.bleu_script):
        raise ValueError('Bleu script not found at {}'.format(args.bleu_script))

    dump_dir = args.dump_dir
    Popen('mkdir -p %s' % dump_dir, shell=True).wait()

    # model needs these vocab sizes, but it's better to be calculated here
    vocab_file = join(args.data_dir, 'vocab.joint')
    vocab, _ = ut.init_vocab(vocab_file)
    args.joint_vocab_size = len(vocab)

    lang_vocab_file = join(args.data_dir, 'lang.vocab')
    lang_vocab, _ = ut.init_vocab(lang_vocab_file)
    args.lang_vocab_size = len(lang_vocab)

    # since args is passed to many modules, keep logger with it instead of reinit everytime
    log_file = join(dump_dir, 'DEBUG.log')
    logger = args.logger = ut.get_logger(log_file)

    # log args for future reference
    logger.info(args)

    model = Transformer(args)
    # TODO: nicer formatting?
    logger.info(model)
    param_count = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info('Model has {:,} parameters'.format(param_count))

    # controller
    data_manager = DataManager(args)
    controller = Controller(args, model, data_manager)
    if args.mode == 'train':
        controller.train()
    elif args.mode == 'translate':
        controller.model.load_state_dict(torch.load(args.model_file))
        files_langs = args.files_langs
        for fl in files_langs:
            input_file, src_lang, tgt_lang = fl.split(',')
            controller.translate(input_file, src_lang, tgt_lang)
    else:
        raise ValueError('Unknown mode. Only train/translate.')
