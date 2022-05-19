import argparse

import numpy as np
import torch

from controller import Controller
from data_manager import DataManager
from model import Transformer
from io_and_bleu import IO

import all_constants as ac
import utils as ut
import configurations

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'translate'], default='train')
    parser.add_argument('--files-langs', type=str, nargs='+',
                        help='Used if do translate, format {file_path,src_lang,tgt_lang...}. e.g.: data/test.en,en,vi data/test.vi,vi,en')
    parser.add_argument('--model-file', type=str,
                        help='Used if do translate, path to checkpoint')
    parser.add_argument('--raw-data-dir', type=str, required=True,
                        help='Path to original data directory')
    parser.add_argument('--processed-data-dir', type=str, required=True,
                        help='Path to preprocessed data directory')
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
    parser.add_argument('--fix-random-seed', action='store_true',
                        help='Use a fixed random seed, for reproducibility')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.fix_random_seed:
        np.random.seed(ac.SEED)
        torch.manual_seed(ac.SEED)
        torch.cuda.manual_seed(ac.SEED)

    config = getattr(configurations, args.config)()
    for k, v in config.items():
        setattr(args, k, v)
    io = IO(args)

    # model needs these vocab sizes, but it's better to be calculated here
    vocab, _ = io.load_vocab()
    args.joint_vocab_size = len(vocab)

    lang_vocab, _ = io.load_lang_vocab()
    args.lang_vocab_size = len(lang_vocab)

    # since args is passed to many modules, keep logger with it instead of reinit everytime
    logger = args.logger = io.get_logger()

    # log args for future reference
    logger.info(args)

    model = Transformer(args)
    # TODO: nicer formatting?
    logger.info(model)
    param_count = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info(f'Model has {param_count:,} parameters')

    # controller
    data_manager = DataManager(args, io)
    controller = Controller(args, model, data_manager, io)
    if args.mode == 'train':
        controller.train()
    elif args.mode == 'translate':
        controller.model.load_state_dict(torch.load(args.model_file))
        files_langs = args.files_langs
        for fl in files_langs:
            input_file, src_lang, tgt_lang = fl.split(',')
            pair = f'{src_lang}2{tgt_lang}'
            all_best_trans, all_beam_trans = controller.translate(pair, ac.TEST, input_file=input_file)
            io.print_test_translations(pair, all_best_trans, all_beam_trans, input_file=input_file)
    else:
        raise ValueError('Unknown mode. Only train/translate.')
