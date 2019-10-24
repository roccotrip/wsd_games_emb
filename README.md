# Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation
This repository contains:
* the code of the model proposed in [Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation (EMNLP 2019)](http://wwwusers.di.uniroma1.it/~navigli/pubs/EMNLP_2019_TripodiNavigli.pdf)
* the code for generating the input vectors
* the code for the [evaluation](https://www.aclweb.org/anthology/E17-1010/)

Tested on Python 3.6.2

## To generate the input vectors ([vector_factory.py](https://github.com/roccotrip/wsd_games_emb/blob/master/vector_factory.py))
* download the [LMMS model(s)](https://github.com/danlou/LMMS#download-sense-embeddings) into data/sensevectors/
* run ```> python vector_factory.py```

## To evaluate the model ([WSDG.py](https://github.com/roccotrip/wsd_games_emb/blob/master/WSDG.py))
* run ```> python WSDG.py```

# Citation
```
@inproceedings{tripodi-2019-game,
    title = "Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation",
    author = "Tripodi, Rocco  and Navigli, Roberto",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing ({EMNLP})",
    month = nov,
    year = "2019",
    address = "Hong Kong",
    publisher = "Association for Computational Linguistics"
}
```
