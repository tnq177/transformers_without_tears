# Ace: An implementation of Transformer in Pytorch
Original code by [Toan Q. Nguyen](http://tnq177.github.io), University of Notre Dame. This fork maintained by Notre Dame's [NLP group](https://nlp.nd.edu/).

This is the *re*-implementation of the paper [Transformers without Tears: Improving the Normalization of Self-Attention](https://arxiv.org/pdf/1910.05895.pdf).

While the code was initially developed to experiment with multilingual NMT, all experiments mentioned in the paper and also in this guide are meant for bilingual only. Regarding the multilingual parts of the code, I followed [XLM](https://github.com/facebookresearch/XLM) and added the following changes:

* language embedding: each language has a an embedding vector which is summed to input word embeddings, similar to positional embedding
* oversampling data before BPE: First training sentences are grouped by languages, then a heuristic multinomial distribution is calculated based on the size of each language. We sample sentences from each language according to this distribution so that rarer languages are better represented and won't be broken into very short BPE segments. See [their paper](https://arxiv.org/abs/1901.07291) for more information. My own implementation is in `preprocessing.py`

If we train for bilingual only, adding language embedding and oversampling data won't make any difference (according to my early experiments). I, however, keep them in the code since they might be useful later. (The language embedding is always used, but whether to do the oversampling is controlled by a parameter. See below.)

This code has been tested with only Python 3.6 and PyTorch 1.4. Pretrained models (retrained, not the ones from the paper): [ted gl2en with warmup](https://drive.google.com/file/d/1yhzSLtAHTOjVTFPdpydrBRlm8lhx2jm8/view?usp=sharing), [iwslt15 en-vi with warmup](https://drive.google.com/file/d/1E4suCr-UDlMtjeNCdTrilYU8Szw56m2X/view?usp=sharing) and [without warmup](https://drive.google.com/file/d/1zgnOr1PmHEdt6_0q2Ebt07eOcfnJU-EX/view)


## Input and Preprocessing

Prior to using this codebase, you should do some initial preprocessing, such as:

* tokenize data
* filter out sentences longer than 200-250

Transformer is known to not generalize well to sentences longer than what it's seen (see [this](https://arxiv.org/abs/1804.00247)), so we need long sentences during training. We don't have to worry about OOM because we always batch by number of tokens. Even a really long sentence of 250 words won't be have more than 2048/4096 BPE tokens.

To use this codebase, create a `data_raw` directory. Then, for each language pair of `src_lang` and `tgt_lang`, create a folder of name `src_lang2tgt_lang` which has the following files:

    train.src_lang   train.tgt_lang
    dev.src_lang     dev.tgt_lang
    test.src_lang    test.tgt_lang

Additionally, create an empty `data_processed` directory.

After that, install [fastBPE](https://github.com/glample/fastBPE). Then run:

`python3 preprocessing.py --raw-data-dir data_raw --processed-data-dir data_processed --num-ops number_of_bpe_ops --pairs src_lang2tgt_lang --fast path_to_fastbpe_binary"`

This will:

* sample sentences from `train.{src_lang, tgt_lang}` to a joint text file
* learn bpe from that
* bpe-encode the rest of files
* create vocabularies
* convert data into ids and save as `.npy` files

Additional preprocessing options:

* `--joint False`: run bpe separately for each language (it defaults to joint)
* `--oversampling False`: don't oversample data, just use original data files (defaults to True)
* `--alpha {value}`: determines amount of oversampling
* `--max-vocab-size`

Example: if we're training an English-Vietnamese model (`en2vi`) using 8000 BPE operations, then the resultant directory looks like this:

```
data
├── en2vi
│   ├── dev.en.bpe
│   ├── dev.en.npy
│   ├── dev.vi.bpe
│   ├── dev.vi.npy
│   ├── test.en.bpe
│   ├── test.vi.bpe
│   ├── train.en.bpe
│   ├── train.en.npy
│   ├── train.vi.bpe
│   └── train.vi.npy
├── joint_all.txt
├── joint.bpe
├── lang.vocab
├── mask.en.npy
├── mask.vi.npy
├── subjoint_en.txt
├── subjoint_en.txt.8000
├── subjoint_en.txt.8000.vocab
├── subjoint_vi.txt
├── subjoint_vi.txt.8000
├── subjoint_vi.txt.8000.vocab
└── vocab.joint
```

These files are just used for computing bpe, and are not needed by the main code:

* `joint_all.txt`: all of the text data which was sampled for all languages
* `subjoint_{lang}.txt`: all of the text data which was sampled for `{lang}`
* `joint.bpe`: the bpe codes file
* `subjoint_{lang}.txt.{num_ops}`: the subjoint files after applying bpe
* `subjoint_{lang}.txt.{num_ops}.vocab`: the bpe vocabulary extracted for `{lang}`

These files are used by the main code:

* `*.bpe`: bpe-encoded versions of the raw data files
* `*.npy`: same data but stored as .npy files
* `mask.{lang}.npy`: vocabulary mask for each language
* `vocab.joint`: bpe vocabulary for all languages (this file is still generated even if you don't use joint bpe)
* `lang.vocab`: each language, paired with a numerical index, for computing language embeddings


## Usage

There are three modes:

* `train`: just train a model
* `train_and_translate`: train the model, then use it to translate the test data
* `translate`: just translate the test data, using a saved model checkpoint

To use the `train` mode:

* write a new configuration function in `configurations.py`
* run `python3 main.py --mode train --raw-data-dir ./data_raw --processed-data-dir ./data_processed --dump-dir ./dump --pairs src_lang2tgt_lang --config config_name`

Note that I separate the two configs:

* hyperparameters/training options: in `configurations.py`
* what pairs are we training on, are we training or translating...: just see `main.py`

Training is logged in `dump/DEBUG.log`. During training, the model is validated on the dev set, and the best checkpoint is saved to `dump/model-SCORE.pth` (also `dump/src_lang2tgt_lang-SCORE.pth`, they are the same). All best/beam translations, final training stats (train/dev perplexities)... are stored in `dump` as well.

To use `train_and_translate` mode, add some more parameters:

`python3 main.py --mode train_and_translate --raw-data-dir ./data_raw --processed-data-dir ./data_processed --dump-dir ./dump --translate-dir ./translate --translate-test True --pairs src_lang2tgt_lang --config config_name`

This will translate the test data and store the output in `./translate`. When translating `test.en.bpe` from language pair `en2vi` it will write the translations to `./translate/test.en2vi.bpe.*`.

To translate using a checkpoint, run:

`python3 main.py --mode translate --raw-data-dir ./data_raw --processed-data-dir ./data_processed --dump-dir ./dump --translate-dir ./translate --files-langs data_proc/src_lang2tgt_lang/temp,temp,src_lang2tgt_lang --config src_lang2tgt_lang --pairs src_lang2tgt_lang --model-file dump/src_lang2tgt_lang-SCORE.pth`

`--files-langs` lets you list specific files to translate. Format is `{path_to_file},{output_name},{src_lang}2{tgt_lang}`. It will write the translations to `./translate/{output_name}.*`.

(You can also use `--translate-test` in `translate` mode, and `--files-langs` in `train_and_translate` mode. You can even use both at once.)

See `main.py` for additional options.

## Options
Many options in `configurations.py` are pretty important:

* ``use_bias``: if set to False, all linear layer won't use bias. Default to True which uses bias.
* ``fix_norm``: fix the word embedding norm to 1 by dividing each word embedding vector by its l2 norm ([Improving Lexical Choice in Neural Machine Translation](https://aclweb.org/anthology/N18-1031))
* ``scnorm``: the ScaleNorm in our paper. This replaces Layer Normalization with a scaled l2-normalization layer. It works by first normalizing input to norm 1 (divide vector by its l2 norm) then scale up by a single, learnable parameter. See ``ScaleNorm`` in ``layers.py``
* ``mask_logit``: if set to True, for each target language, we set the logits of types that are not in its vocabulary to -inf (so after softmax, those probs become 0). The idea is, say src and tgt each has 8000 types in their vocabs, but only 1000 are shared, then we should not predict the other 7000 types in the source.
* ``pre_act``: if True, do PreNorm (normalization->sublayer->residual-add), else do PostNorm (sublayer->residual-add->normalization). See the paper for more discussion and related works.
* ``clip_grad``: gradient clipping value. I find 1.0 works well and stabilizes training as well.
* ``warmup_steps``: number of warmup steps if we do warmup
* ``lr_scheduler``: if `ORG_WU` (see `all_constants.py`), we follow the warmup-cooldown schedule in the [original paper](https://arxiv.org/abs/1706.03762). If ``NO_WU``, we use a constant learning rate ``lr`` which is then decayed whenever development BLEU has not improved over ``patience`` evaluations. If ``UPFLAT_WU`` then we do warmup, but then stay at the peak learning rate and decay like ``NO_WU``.
* ``lr_scale``: multiply learning rate by this value
* ``lr_decay``: decay factor (new_lr <-- old_lr * lr_decay)
* ``stop_lr``: stop training when learning rate reaches this value
* ``eval_metric``: evaluation metric to use when deciding whether to decay learning rate. Default is ``DEV_BLEU`` but you can also use ``DEV_PPL`` (ordinary dev perplexity) or ``DEV_SMPPL`` (dev perplexity after label smoothing)
* ``label_smoothing``: default to 0.1 like in original paper
* ``batch_size``: number of src+tgt tokens in a batch
* ``epoch_size``: number of iterations we consider one epoch
* ``max_epochs``: maximum number of epochs we train for
* ``dropout``: sublayer output's dropout rate
* ``{att,ff,word}_dropout``: dropout rate for attention layer, feedforward and word-dropout. For word-dropout, we replace with UNK instead of zero-ing embeddings. I find word-dropout useful for training low-resource, bilingual model.

Configurations for decoding:

* ``decode_method``: either beam search or ancestral sampling. (note that, if you are doing training, whatever you choose here will be used when evaluating on the dev data. so you should probably make sure to use beam search mode if you are training)
* ``decode_batch_size``: batch size to use during decoding (the default works well for beam search; you might want to lower the batch size if doing sampling and taking many samples)
* ``beam_size``: beam size (or number of samples to generate, if sampling)
* ``beam_alpha``: length normalization using [Wu et al.'s magic formula](https://arxiv.org/abs/1609.08144) (to turn off length normalization, set this to `-1`; it is also ignored during sampling)
* ``use_rel_max_len, rel_max_len, abs_max_len``: decoding needs a max length; determines whether this should be relative (based on length of src sentence) or absolute. defaults to relative length 50 (as used in [Attention Is All You need](https://arxiv.org/abs/1706.03762)), meaning src length + 50 tokens
* ``allow_empty``: whether to allow the empty string as a translation when doing beam search decoding (this parameter is ignored during sampling)


## Some other notes
Because this is my *re*-implementation from memory, there are many pieces of information I forget. I just want to clarify the followings:

* The IWSLT English-Vietnamese dataset is from [paper](http://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf), [data](https://nlp.stanford.edu/projects/nmt/). The other IWSLT datasets are from [paper](https://www.aclweb.org/anthology/N18-2084/), [data](https://github.com/neulab/word-embeddings-for-nmt). I don't remember what is the length limit I use to filter those datasets, but must be approx. 200-250.
* This code doesn't implement the fixed `g=sqrt(d)` experiments in table 7. One can try those experiments by simply edit the `ScaleNorm` class to take in a `trainable` bool param which determines if `g` should be learned or fixed. Then for all normalization layers, set that to False (so we always use `g=sqrt(d)`), except for the final output from the decoder because we want it to scale up to widen the logit range (and sharpen the softmax).
* In the paper, we use early stopping (stop training if dev BLEU has not improved over 20-50 epochs). This code doesn't do that since all kind of early stopping heuristics can sometimes hurt performance. I suggest to just train until learning rate gets too small or max_epochs is reached.
* The original implementation shuffles the whole training dataset every epoch, then re-generates batches. After reading [fairseq](https://arxiv.org/pdf/1904.01038.pdf), I change it to generating batches at first, then reuse them (but still shuffle their order).

If there are any questions, feel free to send me an email (email address in the paper).


## References
Parts of code/scripts are inspired/borrowed from:

* [fairseq](https://github.com/pytorch/fairseq)
* [blocks](https://github.com/mila-iqia/blocks)
* [tensor2tensor](https://github.com/tensorflow/tensor2tensor)
* [nematus](https://github.com/EdinburghNLP/nematus/)
* [moses](https://github.com/moses-smt/mosesdecoder)
* [xlm](https://github.com/facebookresearch/XLM)
* [witwicky](https://github.com/tnq177/witwicky)
* [annotated transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)


![alt text](ace.jpg "Portgas D. Ace")
The art is from [here](https://coverfiles.alphacoders.com/527/52779.jpg)
