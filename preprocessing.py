import argparse
import random
from os.path import join
from collections import Counter
import subprocess
import numpy as np
import all_constants as ac
import utils as ut

np.random.seed(ac.SEED)
random.seed(ac.SEED)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='path to parent data directory')
    parser.add_argument('--fast', type=str, required=True,
                        help='path to fastBPE binary')
    parser.add_argument('--pairs', type=str, required=True,
                        help='command-separated list of language pairs, e.g. en2vi,ar2en,gl2en')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='oversampling prob when learning bpe, see https://arxiv.org/pdf/1901.07291.pdf')
    parser.add_argument('--num-ops', type=int, required=True,
                        help='number of BPE operations')
    parser.add_argument('--max-vocab-size', type=int, default=0,
                        help='maximum vocab size')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    data_dir = args.data_dir
    pairs = args.pairs.split(',')
    langs = []
    for pair in pairs:
        langs.extend(pair.split('2'))
    langs = list(set(langs))
    print('Languages: ', langs)

    # save language vocab
    lang_vocab_file = join(data_dir, 'lang.vocab')
    open(lang_vocab_file, 'w').close()
    with open(lang_vocab_file, 'w') as fout:
        for idx, lang in enumerate(langs):
            fout.write('{} {}\n'.format(lang, idx))

    # group sentences in training data by languages instead of pairs
    datas = {lang: [] for lang in langs}
    for pair in pairs:
        for lang in pair.split('2'):
            infile = join(data_dir, '{}/train.{}'.format(pair, lang))
            with open(infile, 'r') as fin:
                datas[lang].extend(fin.readlines())

    # shuffle training data before sampling
    for lang in langs:
        random.shuffle(datas[lang])

    # calculate sampling probs
    ns = [len(datas[lang]) for lang in langs]
    sum_ns = sum(ns)
    ns = np.array(ns)
    ps = ns / sum_ns
    ps = ps ** args.alpha
    ps = ps / sum(ps)
    print('Alpha = {}'.format(args.alpha))
    for idx, lang in enumerate(langs):
        print('{}: n={}, p={}'.format(lang, len(datas[lang]), ps[idx]))

    """
    Following https://arxiv.org/pdf/1901.07291.pdf, instead of concat all training data
    which might make rare languages (less data) broken into short BPE segments,
    we sample from each languages according to ps above.

    The paper doesn't say how many times should we sample, so I just sample as many as
    the total number of sentences in the whole training data.

    Since a language can appear in many pairs, we also save the sampled data per language
    (called subjoint). This is later on used to extract the BPE vocabulary per language (for BPE encoding).
    """
    joint_all_file = join(data_dir, 'joint_all.txt')
    open(joint_all_file, 'w').close()
    subjoint_files = {}
    subjoint_fouts = {}
    for lang in langs:
        subjoint_files[lang] = join(data_dir, 'subjoint_{}.txt'.format(lang))
        open(subjoint_files[lang], 'w').close()
        subjoint_fouts[lang] = open(subjoint_files[lang], 'w')

    iters = [0] * len(langs)
    print('Sample {} sentences from all training data'.format(sum_ns))
    with open(joint_all_file, 'w') as fout:
        for _ in range(sum_ns):
            lang_idx = np.random.choice(range(len(langs)), p=ps)
            lang = langs[lang_idx]
            idx = iters[lang_idx]
            line = datas[lang][idx]
            fout.write(line)
            subjoint_fouts[lang].write(line)

            iters[lang_idx] = (idx + 1) % len(datas[lang])
            if iters[lang_idx] == 0:
                print('re-shuffle {}'.format(lang))
                random.shuffle(datas[lang])

    for lang in langs:
        subjoint_fouts[lang].close()

    print('Finish sampling')
    print('Learn BPE')
    code_file = join(data_dir, 'joint.bpe')
    open(code_file, 'w').close()
    num_ops = args.num_ops
    fast = args.fast
    command = '{} learnbpe {} {} > {}'.format(fast, num_ops, joint_all_file, code_file)
    print(command)
    subprocess.check_call(command, shell=True)

    # encode each subjoint file and extract vocab for encoding
    print('Extract BPE vocabs')
    bpe_vocab_files = {}
    for lang in langs:
        encoded_file = '{}.{}'.format(subjoint_files[lang], num_ops)
        command = '{} applybpe {} {} {}'.format(fast, encoded_file, subjoint_files[lang], code_file)
        print(command)
        subprocess.check_call(command, shell=True)

        bpe_vocab_file = '{}.vocab'.format(encoded_file)
        command = '{} getvocab {} > {}'.format(fast, encoded_file, bpe_vocab_file)
        print(command)
        subprocess.check_call(command, shell=True)
        bpe_vocab_files[lang] = bpe_vocab_file

    # applying BPE to train,dev,test
    print('Apply BPE to all data')
    for pair in pairs:
        for lang in pair.split('2'):
            bpe_vocab_file = bpe_vocab_files[lang]
            for mode in [ac.TRAIN, ac.DEV, ac.TEST]:
                infile = join(data_dir, '{}/{}.{}'.format(pair, mode, lang))
                encoded_file = '{}.bpe'.format(infile)
                command = '{} applybpe {} {} {} {}'.format(fast, encoded_file, infile, code_file, bpe_vocab_file)
                print(command)
                subprocess.check_call(command, shell=True)

    # now, we extract a joint vocabulary from the encoded train data
    # at the same time, we save each language's vocab
    # this is used to get the logit mask
    joint_vocab = Counter()
    sub_vocabs = {lang: Counter() for lang in langs}
    for pair in pairs:
        for lang in pair.split('2'):
            infile = join(data_dir, '{}/train.{}.bpe'.format(pair, lang))
            with open(infile, 'r') as fin:
                for line in fin:
                    toks = line.strip().split()
                    if toks:
                        joint_vocab.update(toks)
                        sub_vocabs[lang].update(toks)

    start_vocab = ac._START_VOCAB
    sorted_keys = joint_vocab.most_common()
    sorted_keys = [kv[0] for kv in sorted_keys]
    vocab_keys = start_vocab + sorted_keys
    max_vocab_size = args.max_vocab_size
    if 0 < max_vocab_size < len(vocab_keys):
        print('Cut off vocab to top {} types'.format(max_vocab_size))
        vocab_keys = vocab_keys[:max_vocab_size]

    joint_vocab_file = join(data_dir, 'vocab.joint')
    open(joint_vocab_file, 'w').close()
    with open(joint_vocab_file, 'w') as fout:
        for idx, key in enumerate(vocab_keys):
            fout.write('{} {} {}\n'.format(key, idx, joint_vocab.get(key, 0)))

    joint_vocab, _ = ut.init_vocab(joint_vocab_file)

    # get logit mask for each language
    for lang in langs:
        # 0 means masked out, 1 means kept
        mask = np.zeros(len(joint_vocab), dtype=np.uint8)
        mask[ac.UNK_ID] = 1
        mask[ac.EOS_ID] = 1
        for key in sub_vocabs[lang]:
            mask[joint_vocab.get(key, ac.UNK_ID)] = 1

        mask_file = join(data_dir, 'mask.{}.npy'.format(lang))
        np.save(mask_file, mask, allow_pickle=True)

    # save all training data as npy files
    for pair in pairs:
        for mode in [ac.TRAIN, ac.DEV]:
            # read in parallel to make sure we remove empty lines
            src_data = []
            tgt_data = []
            src_lang, tgt_lang = pair.split('2')
            src_infile = join(data_dir, '{}/{}.{}.bpe'.format(pair, mode, src_lang))
            tgt_infile = join(data_dir, '{}/{}.{}.bpe'.format(pair, mode, tgt_lang))
            with open(src_infile, 'r') as f_src, open(tgt_infile, 'r') as f_tgt:
                for src_line, tgt_line in zip(f_src, f_tgt):
                    src_toks = src_line.strip().split()
                    tgt_toks = tgt_line.strip().split()

                    if src_toks and tgt_toks:
                        src_toks = [joint_vocab.get(tok, ac.UNK_ID) for tok in src_toks] + [ac.EOS_ID]
                        tgt_toks = [ac.BOS_ID] + [joint_vocab.get(tok, ac.UNK_ID) for tok in tgt_toks]
                        src_data.append(src_toks)
                        tgt_data.append(tgt_toks)

            src_data = np.array(src_data)
            tgt_data = np.array(tgt_data)
            src_npy_file = join(data_dir, '{}/{}.{}.npy'.format(pair, mode, src_lang))
            tgt_npy_file = join(data_dir, '{}/{}.{}.npy'.format(pair, mode, tgt_lang))
            np.save(src_npy_file, src_data, allow_pickle=True)
            np.save(tgt_npy_file, tgt_data, allow_pickle=True)
