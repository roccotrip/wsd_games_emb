from word_vectors import *
from sense_vectors import *
from strategy_space import *
from bert_attention import *

class wsd_games_vectors(object):
    def __init__(self,
                 datasets=['semeval2013','senseval2', 'senseval3', 'semeval2007', 'semeval2015'],
                 #word_embeddings=['bert-l-u-4','bert-b-c-4'],
                 word_embeddings=['bert-l-c-4'],
                 sense_embeddings=['LMMS_1024'],
                 initializations = ['semcor'],
                 use_softmax = True,
                 use_laplacian = True,
                 use_bert_attention = True,
                 use_svd = True,
                 save_matrices = True):
        self.datasets = datasets
        self.word_embeddings = word_embeddings
        self.sense_embeddings = sense_embeddings
        self.initializations = initializations
        self.use_softmax = use_softmax
        self.use_laplacian = use_laplacian
        self.use_bert_attention = use_bert_attention
        self.use_svd = use_svd
        self.save_matrices = save_matrices

    def get_word_vectors(self):
        for ds in self.datasets:
            texts = [f for f in listdir('datasets/'+ds) if f.startswith('Words')]
            for fileid in sorted(texts):
                with open('datasets/' + ds + '/' + fileid) as f:
                    csvfile = csv.reader(f, delimiter=';')
                    words_data = [w for w in csvfile]
                    words = [w[1] for w in words_data]
                for word_embedding in self.word_embeddings:
                    if word_embedding.startswith('bert'):
                        get_bert_vectors(words, ds, fileid, word_embedding, self.use_svd,self.save_matrices)

    def get_strategy_space(self):
        for ds in self.datasets:
            texts = [f for f in listdir('datasets/'+ds) if f.startswith('Words')]
            for fileid in sorted(texts):
                with open('datasets/' + ds + '/' + fileid) as f:
                    csvfile = csv.reader(f, delimiter=';')
                    words_data = [w for w in csvfile]
                for initialization in self.initializations:
                    get_strat_space(ds, fileid, words_data, initialization,self.use_softmax,self.save_matrices)


    def get_sense_vectors(self):
        for ds in self.datasets:
            texts = [f for f in listdir('datasets/'+ds) if f.startswith('Words')]
            for fileid in sorted(texts):
                with open('datasets/' + ds + '/' + fileid) as f:
                    csvfile = csv.reader(f, delimiter=';')
                    words_data = [w for w in csvfile]
                for sense_embedding in self.sense_embeddings:
                    if sense_embedding.startswith('LMMS'):
                        get_lmms_vectors(ds, fileid, words_data, sense_embedding, self.initializations[0], self.use_svd, self.save_matrices)

    def get_attention_vectors(self):
        for ds in self.datasets:
            texts = [f for f in listdir('datasets/'+ds) if f.startswith('Words')]
            for fileid in sorted(texts):
                with open('datasets/' + ds + '/' + fileid) as f:
                    csvfile = csv.reader(f, delimiter=';')
                    words_data = [w for w in csvfile]
                    words = [w[1] for w in words_data]
                if self.use_bert_attention == True:
                    for word_embedding in self.word_embeddings:
                        Tokens, Indices = get_sentences(words, ds, fileid)
                        mdl = get_model(word_embedding)
                        get_bert_att(Tokens, Indices, ds, mdl, fileid, words,self.save_matrices)


if __name__ == '__main__':
    M = wsd_games_vectors()
    M.get_word_vectors()
    M.get_attention_vectors()
    M.get_strategy_space()
    M.get_sense_vectors()


