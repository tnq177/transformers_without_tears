# Special vocabulary symbols - we always put them at the start.
_PAD = u'_PAD_'
_BOS = u'_BOS_'
_EOS = u'_EOS_'
_UNK = u'_UNK_'
_START_VOCAB = [_PAD, _BOS, _EOS, _UNK]

PAD_ID = 0
BOS_ID = 1
# It's crucial that EOS_ID != 0 (see beam_search decoder)
EOS_ID = 2
UNK_ID = 3

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

# no warmup
NO_WU = 0
# same as in https://arxiv.org/pdf/1706.03762.pdf
ORG_WU = 1
# warmup like ORG_WU but stays there and decay as NO_WU
UPFLAT_WU = 2

# options for what to look at while adjusting learning rate
DEV_BLEU = 0
DEV_PPL = 1 # ordinary perplexity
DEV_SMPPL = 2 # perplexity after label smoothing

# decoding methods
BEAM_SEARCH = 0
SAMPLING = 1

SEED = 147
