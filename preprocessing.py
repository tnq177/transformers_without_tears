import argparse
import random
from os import makedirs
from os.path import join, exists
from collections import Counter
import subprocess
import numpy as np
import all_constants as ac

np.random.seed(ac.SEED)
random.seed(ac.SEED)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-dir', type=str, required=True,
                        help='path to parent data directory')
    parser.add_argument('--processed-data-dir', type=str, required=True,
                        help='directory to output processed data')
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

    pairs = list(sorted(args.pairs.split(',')))
    langs = []
    for pair in pairs:
        langs.extend(pair.split('2'))
    langs = list(sorted(set(langs)))
    print('Languages: ', langs)

    # all hardcoded filenames
    raw_dir = args.raw_data_dir
    proc_dir = args.processed_data_dir
    if not exists(proc_dir):
        makedirs(proc_dir)
    for pair in pairs:
        subdir = join(proc_dir, pair)
        if not exists(subdir):
            makedirs(subdir)
        

    lang_vocab_file = join(proc_dir, 'lang.vocab')
    joint_vocab_file = join(proc_dir, 'vocab.joint')
    
    mask_files = {}
    for lang in langs:
        mask_files[lang] = join(proc_dir, f'mask.{lang}.npy')    

    input_files = {}
    bpe_files = {}
    npy_files = {}
    for pair in pairs:
        for lang in pair.split('2'):
            for mode in [ac.TRAIN, ac.DEV, ac.TEST]:
                input_files[(pair, lang, mode)] = join(raw_dir, f'{pair}/{mode}.{lang}')
                bpe_files[(pair, lang, mode)] = join(proc_dir, f'{pair}/{mode}.{lang}.bpe')
                npy_files[(pair, lang, mode)] = join(proc_dir, f'{pair}/{mode}.{lang}.npy')
    
    num_ops = args.num_ops
    fast = args.fast
    joint_all_file = join(proc_dir, 'joint_all.txt')
    code_file = join(proc_dir, 'joint.bpe')
    subjoint_files = {}
    encoded_files = {}
    bpe_vocab_files = {}
    for lang in langs:
        subjoint_files[lang]  = join(proc_dir, f'subjoint_{lang}.txt')
        encoded_files[lang]   = join(proc_dir, f'subjoint_{lang}.txt.{num_ops}')
        bpe_vocab_files[lang] = join(proc_dir, f'subjoint_{lang}.txt.{num_ops}.vocab')

    # save language vocab
    open(lang_vocab_file, 'w').close()
    with open(lang_vocab_file, 'w') as fout:
        for idx, lang in enumerate(langs):
            fout.write(f'{lang} {idx}\n')

    # group sentences in training data by languages instead of pairs
    datas = {lang: [] for lang in langs}
    for pair in pairs:
        for lang in pair.split('2'):
            infile = input_files[(pair, lang, ac.TRAIN)]
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
    print(f'Alpha = {args.alpha}')
    for idx, lang in enumerate(langs):
        print(f'{lang}: n={len(datas[lang])}, p={ps[idx]}')

    """
    Following https://arxiv.org/pdf/1901.07291.pdf, instead of concat all training data
    which might make rare languages (less data) broken into short BPE segments,
    we sample from each languages according to ps above.

    The paper doesn't say how many times should we sample, so I just sample as many as
    the total number of sentences in the whole training data.

    Since a language can appear in many pairs, we also save the sampled data per language
    (called subjoint). This is later on used to extract the BPE vocabulary per language (for BPE encoding).
    """
    open(joint_all_file, 'w').close()
    subjoint_fouts = {}
    for lang in langs:
        open(subjoint_files[lang], 'w').close()
        subjoint_fouts[lang] = open(subjoint_files[lang], 'w')

    iters = [0] * len(langs)
    print(f'Sample {sum_ns} sentences from all training data')
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
                print(f're-shuffle {lang}')
                random.shuffle(datas[lang])

    for lang in langs:
        subjoint_fouts[lang].close()

    print('Finish sampling')
    print('Learn BPE')
    open(code_file, 'w').close()
    fast = args.fast
    command = f'{fast} learnbpe {num_ops} {joint_all_file} > {code_file}'
    print(command)
    subprocess.check_call(command, shell=True)

    # encode each subjoint file and extract vocab for encoding
    print('Extract BPE vocabs')
    for lang in langs:
        subjoint_file = subjoint_files[lang]
        encoded_file = encoded_files[lang]
        command = f'{fast} applybpe {encoded_file} {subjoint_file} {code_file}'
        print(command)
        subprocess.check_call(command, shell=True)

        bpe_vocab_file = bpe_vocab_files[lang]
        command = f'{fast} getvocab {encoded_file} > {bpe_vocab_file}'
        print(command)
        subprocess.check_call(command, shell=True)

    # applying BPE to train,dev,test
    print('Apply BPE to all data')
    for pair in pairs:
        for lang in pair.split('2'):
            bpe_vocab_file = bpe_vocab_files[lang]
            for mode in [ac.TRAIN, ac.DEV, ac.TEST]:
                infile = input_files[(pair, lang, mode)]
                encoded_file = bpe_files[(pair, lang, mode)]
                command = f'{fast} applybpe {encoded_file} {infile} {code_file} {bpe_vocab_file}'
                print(command)
                subprocess.check_call(command, shell=True)

    # now, we extract a joint vocabulary from the encoded train data
    # at the same time, we save each language's vocab
    # this is used to get the logit mask
    joint_vocab = Counter()
    sub_vocabs = {lang: Counter() for lang in langs}
    for pair in pairs:
        for lang in pair.split('2'):
            infile = bpe_files[(pair, lang, ac.TRAIN)]
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
        print(f'Cut off vocab to top {max_vocab_size} types')
        vocab_keys = vocab_keys[:max_vocab_size]

    open(joint_vocab_file, 'w').close()
    with open(joint_vocab_file, 'w') as fout:
        for idx, key in enumerate(vocab_keys):
            count = joint_vocab.get(key, 0)
            fout.write(f'{key} {idx} {count}\n')

    joint_vocab = {}
    for idx, key in enumerate(vocab_keys):
        joint_vocab[key] = idx

    # get logit mask for each language
    for lang in langs:
        # 0 means masked out, 1 means kept
        mask = np.zeros(len(joint_vocab), dtype=np.uint8)
        mask[ac.UNK_ID] = 1
        mask[ac.EOS_ID] = 1
        for key in sub_vocabs[lang]:
            mask[joint_vocab.get(key, ac.UNK_ID)] = 1

        mask_file = mask_files[lang]
        np.save(mask_file, mask, allow_pickle=True)

    # save all training data as npy files
    for pair in pairs:
        for mode in [ac.TRAIN, ac.DEV]:
            # read in parallel to make sure we remove empty lines
            src_data = []
            tgt_data = []
            src_lang, tgt_lang = pair.split('2')
            src_infile = bpe_files[(pair, src_lang, mode)]
            tgt_infile = bpe_files[(pair, tgt_lang, mode)]
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
            src_npy_file = npy_files[(pair, src_lang, mode)]
            tgt_npy_file = npy_files[(pair, tgt_lang, mode)]
            np.save(src_npy_file, src_data, allow_pickle=True)
            np.save(tgt_npy_file, tgt_data, allow_pickle=True)
