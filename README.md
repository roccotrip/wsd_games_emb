# Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation
This repository contains:
* the code of the model proposed in [Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation (EMNLP 2019)](http://wwwusers.di.uniroma1.it/~navigli/pubs/EMNLP_2019_TripodiNavigli.pdf)
* the code for generating the input vectors
* the code for the [evaluation](https://www.aclweb.org/anthology/E17-1010/)

Tested on Python 3.6.2

## To generate the input vectors ([vector_factory.py]())
* download the [LMMS model(s)](https://github.com/danlou/LMMS#download-sense-embeddings) into data/sensevectors/
* run ```> python vector_factory.py```

## To evaluate the model ([WSDG.py]())
* run ```> python WSDG.py```
