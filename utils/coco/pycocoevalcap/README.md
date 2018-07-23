Microsoft COCO Caption Evaluation
===================

Evaluation codes for MS COCO caption generation.

## Description ##
This repository provides Python 3 support for the caption evaluation metrics used for the MS COCO dataset.

The code is derived from the original repository that supports Python 2.7: https://github.com/tylin/coco-caption.  
Caption evaluation depends on the COCO API that natively supports Python 3 (see Requirements).

## Requirements ##
- Java 1.8.0
- Python 3 (tested on Python 3.6)
- pycocotools (COCO Python API): https://github.com/cocodataset/cocoapi

## Files ##
./
- evals.py: The file includes COCOEavlCap class that can be used to evaluate results on COCO.
- tokenizer: Python wrapper of Stanford CoreNLP PTBTokenizer
- bleu: Bleu evalutation codes
- meteor: Meteor evaluation codes
- rouge: Rouge-L evaluation codes
- cider: CIDEr evaluation codes

## References ##

- [Microsoft COCO Captions: Data Collection and Evaluation Server](http://arxiv.org/abs/1504.00325)
- PTBTokenizer: We use the [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) which is included in [Stanford CoreNLP 3.4.1](http://nlp.stanford.edu/software/corenlp.shtml).
- BLEU: [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)
- Meteor: [Project page](http://www.cs.cmu.edu/~alavie/METEOR/) with related publications. We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor). Changes have been made to the source code to properly aggreate the statistics for the entire corpus.
- Rouge-L: [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf)
- CIDEr: [CIDEr: Consensus-based Image Description Evaluation] (http://arxiv.org/pdf/1411.5726.pdf)

## Developers ##
- Xinlei Chen (CMU)
- Hao Fang (University of Washington)
- Tsung-Yi Lin (Cornell)
- Ramakrishna Vedantam (Virgina Tech)

## Acknowledgement ##
- David Chiang (University of Norte Dame)
- Michael Denkowski (CMU)
- Alexander Rush (Harvard University)
