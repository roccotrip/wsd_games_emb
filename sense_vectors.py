from aux import *
from vectorspace import *
from dataset_reader import *


def get_sense_vectors():
    fname = '../LMMS_mio/data/sense_vectors/SemCor_bert_large_cased.pkl'
    if os.path.isfile(fname) == True:
        with open(fname, 'rb') as f:
            sense_vectors = pickle.load(f)
    return sense_vectors

def get_lmms_vectors(ds,fileid,words_data,sense_embedding, distribution, use_svd, save_matrices):
    fileid = fileid.split('.')[0]
    fileid = fileid.split('-')[1]
    print(ds,fileid)
    if '1024' in sense_embedding:
        VS = get_sense_vectors()
        dims = 1024
    if '2048' in sense_embedding:
        VS = SensesVSM('data/sense_vectors/lmms_2048.bert-large-cased.npz')
        dims = VS.ndims
    if '2348' in sense_embedding:
        VS = SensesVSM('data/sense_vectors/lmms_2348.bert-large-cased.fasttext-commoncrawl.npz')
        dims = VS.ndims
    text = get_text(ds,fileid)
    with open('datasets/' + ds + '/key.txt') as f:
        keyS = f.readlines()
    Words = getAllWords(fileid, text, keyS)
    ww = [(w[1], w[2]) for w in words_data]
    new_words = []
    x = 0
    for word in Words:
        if (word.lemma, word.pos) == ww[x]:
            new_words.append(word)
            x += 1
    assert len(ww) == x
    words = new_words
    #print('N Words:', len(words))
    [space, keys] = getSpace(words)
    with open('datasets/' + ds + '/Keys-' + fileid + '.csv') as f:
        csvfile = csv.reader(f, delimiter=';')
        Keys = [k for k in csvfile]
    with open('vectors/' + ds + '/strat_space/' + distribution + '/X-' + fileid + '.tsv') as f:
        csvfile = csv.reader(f, delimiter='\t')
        X = np.array([[float(kk) for kk in k] for k in csvfile])
    Z = []
    assert len(Keys) == len(space) == len(X[0])
    assert len(words) == len(X)
    I = []
    for i, word_ in enumerate(words):
        idx_keys = np.where(X[i] > 0)[0].tolist()
        idx_keys = [idx_ for idx_ in idx_keys if idx_ not in I]
        I += idx_keys
    for keys_ in keys:
        counts = []
        Vec = []
        for key in keys_:
            try:
                counts.append((key, wn.lemma_from_key(key).count()))
            except:
                None
        counts = sorted(counts, key=lambda element: (element[1], element[0]),reverse=True)
        for key,c in counts:
            if '2048' in sense_embedding or '2348' in sense_embedding:
                if key in VS.labels:
                    Vec = [VS.get_vec(key)]
                break
            if '1024' in sense_embedding:
                if key in VS:
                    vecs =  VS.get(key)
                    Vec = np.mean(vecs,axis=0)
                    break
        if Vec == []:
            Vec = np.random.rand(dims)
        if '2048' in sense_embedding or '2348' in sense_embedding:
            Vec = np.mean(np.array(Vec),axis=0)
        Z.append(Vec.tolist())
    assert len(Z) == len(Keys)
    if save_matrices == True:
        path = 'vectors/' + ds + '/sense/' + sense_embedding
        create_directory(path)
        writeMat(Z, path + '/Z-' + fileid)
    if use_svd == True:
        Z = np.array(Z)
        Z = remove_pc(Z, npc=1)
        if save_matrices == True:
            writeMat(Z, path + '/Z-' + fileid + '-svd')
    if save_matrices == False:
        return Z
