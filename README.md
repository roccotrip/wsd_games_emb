# Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation
This repository contains:
* the code of the model proposed in [Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation (EMNLP 2019)](http://wwwusers.di.uniroma1.it/~navigli/pubs/EMNLP_2019_TripodiNavigli.pdf)
* the code for generating the input vectors
* the code for the [evaluation](https://www.aclweb.org/anthology/E17-1010/)

Tested on Python 3.6.2

## To generate the input vectors ([vector_factory.py](https://github.com/roccotrip/wsd_games_emb/blob/master/vector_factory.py))
* download the [LMMS model(s)](https://github.com/danlou/LMMS#download-sense-embeddings) into data/sensevectors/
* alternatively you can these [sense vectors](https://drive.google.com/file/d/1zGjs2amrfofPAkS89XCLwkE1nuV6b--F/view?usp=sharing) trained using BERT (large) avareging the word representations of the senses tagged in SemCor.
* run ```> python vector_factory.py```

## To evaluate the model ([WSDG.py](https://github.com/roccotrip/wsd_games_emb/blob/master/WSDG.py))
* run ```> python WSDG.py```

# Citation
Tripodi, Rocco and Navigli, Roberto, Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019. (to appear)

```
@inproceedings{tripodi-navigli-2019-game,
    title = "Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation",
    author = "Tripodi, Rocco  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1009",
    doi = "10.18653/v1/D19-1009",
    pages = "88--99",
    abstract = "Game-theoretic models, thanks to their intrinsic ability to exploit contextual information, have shown to be particularly suited for the Word Sense Disambiguation task. They represent ambiguous words as the players of a non cooperative game and their senses as the strategies that the players can select in order to play the games. The interaction among the players is modeled with a weighted graph and the payoff as an embedding similarity function, that the players try to maximize. The impact of the word and sense embedding representations in the framework has been tested and analyzed extensively: experiments on standard benchmarks show state-of-art performances and different tests hint at the usefulness of using disambiguation to obtain contextualized word representations.",
}
```
