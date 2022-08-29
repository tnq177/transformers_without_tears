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

    parser.add_argument('--mode', choices=['train', 'train_and_translate', 'translate'], default='train')
    parser.add_argument('--model-file', type=str,
                        help='Used if in translate mode, path to checkpoint')
    parser.add_argument('--translate-test', choices=('True','False'), default='False',
                        help='Used if in train_and_translate or translate mode, says whether to translate the test.[lang].bpe files in the processed data directory')
    parser.add_argument('--files-langs', type=str, nargs='+',
                        help='Used if in train_and_translate or translate node, specifies which files to translate, format {input_filepath,output_filename,src_lang2tgt_lang...}. e.g.: data/test.en.bpe,test.en2vi.bpe,en2vi data/test.vi.bpe,test.vi2en.bpe,vi2en')

    parser.add_argument('--raw-data-dir', type=str, required=True,
                        help='Path to original data directory')
    parser.add_argument('--processed-data-dir', type=str, required=True,
                        help='Path to preprocessed data directory')
    parser.add_argument('--dump-dir', type=str, required=True,
                        help='Path to dump directory (store all stats, checkpoints...)')
    parser.add_argument('--translate-dir', type=str,
                        help='Path to translate directory (where to output the translations)')

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

    # sanity check arguments
    if args.mode == 'train_and_translate' or args.mode == 'translate':
        if args.translate_test == 'False' and args.files_langs is None:
            print('In train_and_translate or translate mode, must provide files to translate using the \'--translate-test\' or \'--files-langs\' parameters')
            exit()
        if args.translate_dir is None:
            print('In train_and_translate or translate mode, must use \'--translate-dir\' to specify where to output translations')
            exit()
    if args.mode == 'translate':
        if args.model_file is None:
            print('In translate mode, must use the \'--model-file\' parameter to specify the trained model')
            exit()

    # fix random seed
    if args.fix_random_seed:
        np.random.seed(ac.SEED)
        torch.manual_seed(ac.SEED)
        torch.cuda.manual_seed(ac.SEED)

    # load options from config file
    config = getattr(configurations, args.config)()
    for k, v in config.items():
        setattr(args, k, v)

    # initialize IO
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

    # initialize model
    model = Transformer(args)
    logger.info(model)
    param_count = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info(f'Model has {param_count:,} parameters')

    # initialize controller
    data_manager = DataManager(args, io)
    controller = Controller(args, model, data_manager, io)

    # train
    if args.mode == 'train' or args.mode == 'train_and_translate':
        logger.info('Beginning training')
        controller.train()

    # files to translate
    files_langs = {}
    if args.files_langs is not None:
        for fl in args.files_langs:
            input_file, output_file, pair = fl.split(',')
            if pair not in files_langs:
                files_langs[pair] = []
            files_langs[pair].append((input_file, output_file))

    # translate
    if args.mode == 'train_and_translate' or args.mode == 'translate':
        # for translate mode, just need to load state dict once
        if args.mode == 'translate':
            model.load_state_dict(torch.load(args.model_file))
        # translate test files
        if args.translate_test == 'True':
            logger.info('Translating test files')
            for pair in args.pairs.split(','):
                # for train_and_translate mode, need to reload best checkpoint for that pair
                if args.mode == 'train_and_translate':
                    model.load_state_dict(io.load_best_ckpt(pair))
                all_best_trans, all_beam_trans = controller.translate(pair, ac.TEST)
                io.print_test_translations(pair, all_best_trans, all_beam_trans)
        # translate other files
        if args.files_langs is not None:
            logger.info('Translating user-specified files')
            for pair in files_langs:
                # for train_and_translate mode, need to reload best checkpoint for that pair
                if args.mode == 'train_and_translate':
                    model.load_state_dict(io.load_best_ckpt(pair))
                for (input_file, output_file) in files_langs[pair]:
                    all_best_trans, all_beam_trans = controller.translate(pair, ac.TEST)
                    io.print_test_translations(pair, all_best_trans, all_beam_trans, input_file=input_file, output_file=output_file)
