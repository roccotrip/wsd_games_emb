from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import platform
import xml.etree.ElementTree as ET
from sklearn.decomposition import TruncatedSVD
import os
import numpy as np
import pickle
import logging
import re
import argparse
import collections
from tqdm import tqdm
#import seaborn
#import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from os import listdir
import spacy
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet as wn
stopset = set([w for w in stopwords.words('english') if w not in []] + [p for p in string.punctuation])
stops = list(stopset)+['[SEP]','[CLS]','[UNK]']
stops = [s for s in stops if s not in ['down','can','s','ll','near','d']]

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_transformers import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def writeMat(Mat,path):
    with open(path+'.tsv','w') as f:
        w = csv.writer(f, delimiter='\t')
        for row in Mat:
            w.writerow(row)

def entropy(p):
    #p = p + abs(min(p.detach())) + .00001
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

def getSpace(words):
    Space = []
    Keys = []
    for word in words:
        synsets = word.synsets
        for synset in synsets:
            if synset not in Space:
                Space.append(synset)
                lemmas = synset.lemmas()
                lkeys = [lemma.key() for lemma in lemmas]
                Keys.append([k for k in lkeys if k.startswith(word.lemma + '%')])
        # keys = word.keys
        # for key in keys:
        #     if key not in Keys and key.startswith(word.lemma+'%'):
        #         Keys.append(key)
    return [Space, Keys]

def getSpaceWiC(words):
    Space = []
    Keys = []
    for word in words:
        synsets = word.synsets
        for synset in synsets:
            if synset not in Space:
                Space.append(synset)
                Keys.append([l.key() for l in synset.lemmas()])
    return [Space, Keys]

def get_text(ds,fileid):
    path = 'datasets/' + ds
    dataFilePath = path + '/data.xml'
    Texts = []
    tree = ET.parse(dataFilePath)
    for s in tree.iter('text'):
        Texts.append(s)
    for text in Texts:
        if text.items()[0][1] in fileid:
            return text

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_model(word_embedding):
    mdl = 'bert-'
    if '-l-' in word_embedding:
        mdl = mdl+'large-'
    else:
        mdl = mdl+'base-'
    if '-u-' in word_embedding:
        mdl = mdl + 'uncased'
    else:
        mdl = mdl + 'cased'
    return mdl

# BERT FUNCTION - adapted form

class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


## -------- WiC aux ------------ ##

class Word0(object):
    idd = ''
    lemma = ''
    pos = ''
    form = ''
    synsets = ''


    # The class "constructor" - It's actually an initializer
    def __init__(self, wid, lemma, pos, form, synsets,vector):
        self.idd = wid
        self.lemma = lemma
        self.pos = pos
        self.form = form
        self.synsets = synsets
        self.vector = vector

# maps spacy pos tags to wordnet pos tags
def map_pos_(pos):
    if pos.startswith('N') or pos.startswith('PROPN'):
        return 'n'
    if pos.startswith('V'):
        return 'v'
    if pos.startswith('ADJ'):
        return 'a'
    if pos.startswith('ADV'):
        return 'r'
    return ''

# get the synsets of a lemma-pos
def getSynsets(lemma,pos):
    synsets = wn.synsets(lemma,pos)
    if synsets == [] and '_' in lemma:
        synsets = wn.synsets(lemma)
    if synsets == [] and '-' in lemma:
        synsets = wn.synsets(lemma.replace('-', ' '))
    return synsets

def getWordsWic(text,tidx,p):
    docId = 0
    sentId = 0
    wordId = 0
    Words = []
    idx=0
    for sent in text.sents:
        for w in sent:
            if w.text.lower() not in stopset or w.text.lower() in ['down','can']:
                lemma = w.lemma_
                if idx == tidx:
                    pos = map_pos_(p)
                else:
                    pos = map_pos_(w.pos_)
                form = w.text
                vector = w.vector
                if pos in ['n', 'v', 'a', 'r']:
                    synsets = getSynsets(lemma, pos)
                    wid = 'd' + str(docId) + '.' + 's' + str(sentId) + '.' + 't' + str(wordId)
                    word = Word0(wid, lemma, pos, form, synsets,vector)
                    Words.append(word)
                    wordId += 1
            idx+=1
        sentId += 1
    return Words

def instring(w,words):
    for ww in words:
        if w in ww:
            return True
    return False

def get_bert_WiC(T,mdl,words_,target):
    model_class = BertModel
    tokenizer_class = BertTokenizer
    pretrained_weights = mdl
    Mat = []
    words_ = [w.form.lower().strip('-').strip('"') for w in words_]
    with open('bert_text.txt','w') as f:
        for t in T:
            f.write(t +'\n')
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_file", default='bert_text.txt', type=str, required=False)
    parser.add_argument("--output_file", default='output_bert.txt', type=str, required=False)
    parser.add_argument("--bert_model", default=mdl, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    # if 'uncased' in mdl:
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--mode", default='client', type=str)
    parser.add_argument("--output_hidden_states", default='True', type=bool)
    parser.add_argument("--port", default=51940, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=250, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    args = parser.parse_args()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))
    layer_indexes = [int(x) for x in args.layers.split(",")]
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    examples = read_examples(args.input_file)
    features = convert_examples_to_features(examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
    for input_ids, input_mask, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        all_hidden_states, all_attentions = model(input_ids, token_type_ids=None, attention_mask=input_mask)[-2:]
        for b, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            unique_id = int(feature.unique_id)
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_out_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = all_hidden_states[int(layer_index)].detach().cpu().numpy()
                    layer_output = layer_output[b]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [round(x.item(), 6) for x in layer_output[i]]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = token
                out_features["layers"] = all_layers
                all_out_features.append(out_features)
            ww = words_
            ww_bert = [out_features['token'] for out_features in all_out_features]
            Idx = []
            #print(ww)
            #w_forms = [wf.form for wf in words_]
            if len(ww) > 0:
                idx_bert = 0
                for i, www in enumerate(ww):
                    if (www.lower() in stops and ww_bert[idx_bert+1].startswith('#')==False and www.lower() != target.lower()):
                        if ww_bert[idx_bert-1].startswith('#') == False:
                            idx_bert += 1
                        continue
                    for w in ww_bert[idx_bert:]:
                        if w in ['[SEP]', '[CLS]','[UNK]'] or w in stops:
                            idx_bert += 1
                            continue
                        if w == www:
                            Idx.append([idx_bert])
                            idx_bert += 1
                            break
                        if ' ' in www:
                            if www.split(' ')[0] == w:
                                Idx.append([idx_bert])
                                idx_bert += 1
                                for w_ in www.split(' ')[1:]:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                while ww_bert[idx_bert].startswith('##') and ww_bert[idx_bert].strip('#') in www:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                break
                        if '\'' in www:
                            if www.split('\'')[0] == w:
                                Idx.append([idx_bert])
                                idx_bert += 1
                                break
                        if '-' in www:
                            if www.split('-')[0] == w:
                                Idx.append([idx_bert])
                                idx_bert += 1
                                for w_ in ['-'] + www.split('-')[1:]:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                while ww_bert[idx_bert].startswith('##') and ww_bert[idx_bert].strip('#') in www:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                break
                        if '/' in www:
                            if www.split('/')[0] == w:
                                Idx.append([idx_bert])
                                idx_bert += 1
                                for w_ in ['/'] + www.split('/')[1:]:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                while ww_bert[idx_bert].startswith('##') and ww_bert[idx_bert].strip('#') in www:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                break
                        if len(www)>0:
                            if '.' == www[-1]:
                                if www.split('.')[0] == w:
                                    Idx.append([idx_bert])
                                    idx_bert += 1
                                    www_list = [x for x in www.split('.')[1:] if x != '']
                                    for w_ in www_list + ['.']:
                                        Idx[-1].append(idx_bert)
                                        idx_bert += 1
                                    while ww_bert[idx_bert].startswith('##') and ww_bert[idx_bert].strip('#') in www:
                                        Idx[-1].append(idx_bert)
                                        idx_bert += 1
                                    break

                        if www.startswith(w) and ww_bert[idx_bert + 1].startswith('##') and www not in ww_bert[idx_bert:]:
                            Idx.append([idx_bert])
                            idx_bert += 1
                            continue
                        if len(Idx) > 0:
                            if w.startswith('##') and Idx[-1][-1] == idx_bert - 1:
                                Idx[-1].append(idx_bert)
                                idx_bert += 1
                                if ww_bert[idx_bert].startswith('##'):
                                    continue
                                else:
                                    break
                        idx_bert += 1
                print(ww_bert)
                print(ww)
                print(Idx)
                gloss_conc = []
                for idx in Idx:
                    vec1 = torch.mean(torch.tensor([all_out_features[iidx]['layers'][0]['values'] for iidx in idx]), 0)
                    vec2 = torch.mean(torch.tensor([all_out_features[iidx]['layers'][1]['values'] for iidx in idx]), 0)
                    vec3 = torch.mean(torch.tensor([all_out_features[iidx]['layers'][2]['values'] for iidx in idx]), 0)
                    vec4 = torch.mean(torch.tensor([all_out_features[iidx]['layers'][3]['values'] for iidx in idx]), 0)
                    vec_sum = vec1+vec2+vec3+vec4
                    gloss_conc.append(vec_sum.tolist())
    return gloss_conc


def getZWiC(Keys, dims,VS):
    Z=[]
    for keys in Keys:
        for k in keys:
            if k in VS.labels:
                Z.append(VS.get_vec(k).tolist())
                break
            Vec = np.random.rand(dims).tolist()
            print('Keys not found:', keys)
            Z.append(Vec)
    assert len(Z) == len(Keys)
    Z = np.array(Z)
    Z = remove_pc(Z, npc=1)
    Z = cosine_similarity(Z)
    Z = np.asmatrix(Z)
    #Z[Z < 0] = 0
    return Z

def dynamics(X,A,Z):
    iter = 1
    X_init = X
    while True:
        P = np.zeros(X.shape)
        C = 0
        for i in range(len(X)):
            nn = np.nonzero(A[i])[1].tolist()
            x_i_idxs = X[i].nonzero()[0]
            x_i = X[i,x_i_idxs]
            if x_i.nonzero()[0].shape[0] > 1:
                for j in nn:
                    x_j_idxs = X[j].nonzero()[0]
                    x_j = X[j, x_j_idxs]
                    payoff = A[i,j] * Z[np.ix_(x_i_idxs,x_j_idxs)]
                    P[i,x_i_idxs] = P[i,x_i_idxs] + np.multiply(x_i, np.dot(payoff,x_j.transpose()))
            else:
                P[i] = x_i
        shift = max(-P.min()+1e-20,C)
        P = P+shift
        P = np.multiply(P,X)
        Xnew = P / np.sum(P, axis=1)[:, None]
        diff = np.linalg.norm(X-Xnew)
        X = Xnew
        if iter >= 100 or diff < 1e-05:
            return X
        iter += 1

def get_graph_WiC(A):
    A = cosine_similarity(A)
    A = np.asmatrix(A)
    np.fill_diagonal(A, 0)
    A[A < .1] = 0
    return A

def get_softmax(X):
    softX = []
    for row in X.tolist():
        rr = np.array(row)
        r = rr[rr > 0]
        X2 = np.exp(r) / np.sum(np.exp(r))
        rr[rr > 0] = X2
        softX.append(rr.tolist())
    return softX