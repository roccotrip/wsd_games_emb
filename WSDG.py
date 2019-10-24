# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from aux import *

class WSDG(object):
    def __init__(self, datasets=['semeval2013','senseval2', 'senseval3', 'semeval2007', 'semeval2015'],
                 word_embeddings=['bert-l-u-4'],
                 sense_embeddings=['LMMS_2048'],
                 initializations = ['semcor'],
                 use_softmax = True,
                 use_laplacian = True,
                 use_bert_attention = True,
                 use_svd_senses = True,
                 use_svd_words=True,
                 max_iter=100,
                 tol=.0001):
        self.datasets = datasets
        self.word_embeddings = word_embeddings
        self.sense_embeddings = sense_embeddings
        self.initializations = initializations
        self.use_softmax = use_softmax
        self.use_laplacian = use_laplacian
        self.use_bert_attention = use_bert_attention
        self.use_svd_senses = use_svd_senses
        self.use_svd_words = use_svd_words
        self.max_iter = max_iter
        self.tol = tol

    def get_gold_standard(self,ds,fileid):
        path = 'datasets/' + ds
        with open(path + '/Y-' + fileid + '.csv', 'r') as f:
            w = csv.reader(f, delimiter=';')
            self.Y = [[int(xx) for xx in x] for x in w]

    def get_X(self, distribution):
        path = 'vectors/' + self.ds + '/strat_space/' + distribution
        with open(path + '/X-' + self.fileid + '.tsv', 'r') as f:
            w = csv.reader(f, delimiter='\t')
            X = np.array([[float(xx) for xx in x] for x in w])
        #if self.use_softmax == True:
        #    X = np.array(get_softmax(X))
        self.X = X

    def get_Z(self, sense_embedding):
        #base_data = '/Volumes/T5/python/wsd_games/datasets/'
        #path = base_data + 'all_words_synonyms/senses/' + self.ds
        path = 'vectors/' + self.ds + '/sense/' + sense_embedding
        if self.use_svd_senses == True:
            fpath = path + '/Z-' + self.fileid + '-svd.tsv'
        else:
            fpath = path + '/Z-' + self.fileid + '.tsv'
        with open(fpath, 'r') as f:
            w = csv.reader(f, delimiter='\t')
            Z = [[float(xx) for xx in x] for x in w]
        if len(Z[0]) != len(Z):
            Z = cosine_similarity(Z)
        else:
            Z = np.array(Z)
        Z[Z < 0] = 0
        self.Z = Z

    def get_A(self, word_embedding):
        mdl = get_model(word_embedding)
        layer = word_embedding.split('-')[-1]
        path = 'vectors/' + self.ds + '/word/' + mdl + '-' + layer
        if self.use_svd_words == True:
            fpath = path + '/A-' + self.fileid + '-svd.tsv'
        else:
            fpath = path + '/A-' + self.fileid + '.tsv'
        with open(fpath, 'r') as f:
            w = csv.reader(f, delimiter='\t')
            A = [x for x in w]
        A = cosine_similarity(A)
        A = np.asmatrix(A)
        np.fill_diagonal(A, 0)
        A[A < .1] = 0
        if self.use_bert_attention == True:
            path = 'vectors/' + self.ds + '/attention/' + mdl
            with open(path + '/Att-' + self.fileid + '.tsv', 'r') as f:
                w = csv.reader(f, delimiter='\t')
                Att = [[float(xx) for xx in x] for x in w]
                A = A + np.asmatrix(Att)
        if self.use_laplacian == True:
            Ssum = A.sum(axis=1)
            Ssum = np.power(Ssum, -0.5)
            Ssum = Ssum.transpose().tolist()[0]
            S12 = np.diag(Ssum)
            S12[~ np.isfinite(S12)] = 0
            A = (S12 * A) * S12
        self.A = A
        self.n_words = A.shape[0]

    def dynamics(self):
        iter = 1
        self.X_init = self.X
        while True:
            P = np.zeros(self.X.shape)
            C = 0
            for i in range(self.n_words):
                nn = np.nonzero(self.A[i])[1].tolist()
                x_i_idxs = self.X[i].nonzero()[0]
                x_i = self.X[i,x_i_idxs]
                if x_i.nonzero()[0].shape[0] > 1:
                    for j in nn:
                        x_j_idxs = self.X[j].nonzero()[0]
                        x_j = self.X[j, x_j_idxs]
                        Z = self.A[i,j] * self.Z[np.ix_(x_i_idxs,x_j_idxs)]
                        P[i,x_i_idxs] = P[i,x_i_idxs] + np.multiply(x_i, np.dot(Z,x_j.transpose()))
                else:
                    P[i] = x_i
            shift = max(-P.min()+1e-20,C)
            P = P+shift
            P = np.multiply(P,self.X)
            Xnew = P / np.sum(P, axis=1)[:, None]
            diff = np.linalg.norm(self.X-Xnew)
            self.X = Xnew
            if iter >= self.max_iter or diff < self.tol:
                self.iterations = iter
                break
            iter += 1

    def eval(self):
        correct = 0
        unique = 0
        given = 0
        n_instances = 0
        for i,y in enumerate(self.Y):
            if y[0] != -1:
                n_instances+=1
                prediceted_idxs = self.X[i].nonzero()[0]
                prediceted_probs = self.X[i][prediceted_idxs]
                senses = self.X_init[i].nonzero()[0]
                if len(senses) == 1:
                    given+=1
                    unique+=1
                    correct+=1
                else:
                    max_prob = max(prediceted_probs)
                    if max_prob>1/len(senses):
                        given+=1
                        given_idx = np.where(self.X[i] == max_prob)[0]
                        if given_idx+1 in y:
                            correct+=1
        accuracy = (correct/given)*100
        recall = (correct/n_instances)*100
        F1 = 2 * ((accuracy * recall) / (accuracy + recall))
        self.Results[self.ds][self.fileid]['n_instances'] = n_instances
        self.Results[self.ds][self.fileid]['given'] = given
        self.Results[self.ds][self.fileid]['unique'] = unique
        self.Results[self.ds][self.fileid]['correct'] = correct
        self.Results[self.ds][self.fileid]['accuracy'] = accuracy
        self.Results[self.ds][self.fileid]['recall'] = recall
        print(self.ds, self.fileid, F1)

    def eval_ds(self):
        n_instances = 0
        given = 0
        correct = 0
        for k,v in self.Results[self.ds].items():
            n_instances+=v['n_instances']
            given+=v['given']
            correct+=v['correct']
        accuracy = (correct / given) * 100
        recall = (correct / n_instances) * 100
        F1 = 2 * ((accuracy * recall) / (accuracy + recall))
        self.Results[self.ds]['accuracy'] = accuracy
        self.Results[self.ds]['recall'] = recall
        self.Results[self.ds]['F1'] = F1
        print(self.ds, F1,'\n')

    def wsdg(self):
        self.Results = {}
        for word_embedding in self.word_embeddings:
            for sense_embeding in self.sense_embeddings:
                for initialization in self.initializations:
                    print('WORD EMBEDDING: ' + word_embedding)
                    print('SENSE EMBEDDING: ' + sense_embeding)
                    print('INITIALIZATION: ' + initialization)
                    print('USE BERT ATTENTION: ' + str(self.use_bert_attention))
                    print('USE SVD SENSES: ' + str(self.use_svd_senses))
                    print('USE SVD WORDS: ' + str(self.use_svd_words))
                    print('USE SOFTMAX: ' + str(self.use_softmax))
                    print('USE LAPLACIAN: ' + str(self.use_laplacian))
                    for ds in self.datasets:
                        self.Results[ds] = {}
                        texts = [f for f in listdir('datasets/' + ds) if f.startswith('Words')]
                        for fileid in sorted(texts):
                            fileid = fileid.split('.')[0]
                            fileid = fileid.split('-')[-1]
                            self.ds = ds
                            self.fileid = fileid
                            self.Results[ds][fileid] = {}
                            self.get_A(word_embedding)
                            self.get_X(initialization)
                            self.get_Z(sense_embeding)
                            self.get_gold_standard(ds,fileid)
                            assert len(self.A) == len(self.X)
                            assert len(self.Z) == len(self.X[0])
                            self.dynamics()
                            self.eval()
                        self.eval_ds()

if __name__ == '__main__':
    WSDG = WSDG()
    WSDG.wsdg()
