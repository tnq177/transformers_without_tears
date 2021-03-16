import re
import logging
from datetime import timedelta
from subprocess import Popen, PIPE
import torch


def get_logger(logfile):
    """Global logger for every logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(filename)s:%(lineno)s: %(message)s', '%m/%d %H:%M:%S')
    if not logger.handlers:
        debug_handler = logging.FileHandler(logfile)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger


def format_seconds(seconds):
    return str(timedelta(seconds=seconds))


def init_vocab(vocab_file):
    vocab = {}
    ivocab = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            temp = line.strip().split()
            if temp:
                vocab[temp[0]] = int(temp[1])
                ivocab[int(temp[1])] = temp[0]

    return vocab, ivocab


def get_positional_encoding(dim, sentence_length):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    div_term = -(torch.arange(end=float(dim), device=device) // 2) * 2.0 / dim
    div_term = torch.pow(10000.0, div_term).reshape(1, -1)
    pos = torch.arange(end=float(sentence_length), device=device).reshape(-1, 1)
    encoded_vec = torch.matmul(pos, div_term)
    encoded_vec[:, 0::2] = torch.sin(encoded_vec[:, 0::2])
    encoded_vec[:, 1::2] = torch.cos(encoded_vec[:, 1::2])

    return encoded_vec.reshape([sentence_length, dim])


def remove_bpe(infile, outfile=None):
    if not outfile:
        outfile = infile + '.nobpe'

    open(outfile, 'w').close()
    Popen("sed -r 's/(@@ )|(@@ ?$)//g' < {} > {}".format(infile, outfile), shell=True, stdout=PIPE).communicate()


def calc_bleu(bleu_script, trans_file, ref_file):
    # compute BLEU
    multibleu_cmd = ['perl', bleu_script, ref_file, '<', trans_file]
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
