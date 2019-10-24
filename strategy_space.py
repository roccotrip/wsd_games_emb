from aux import *
from dataset_reader import *

def get_strat_space(ds, fileid, words_data, dist,use_softmax,save_matrices):
    fileid = fileid.split('.')[0]
    fileid = fileid.split('-')[1]
    print(ds, fileid)
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
    [space, keys] = getSpace(words)
    l = len(space)
    Sspace = []
    Indices = []
    tidxw = 0
    if dist == 'semcor':
        with open('data/SemCor/SC2.pk', 'rb') as f:
            SC = pickle.load(f)
    for word in words:
        row = [0] * l
        synsets = word.synsets
        sense_keys = word.sense_keys
        indices = [space.index(synset) for synset in synsets]
        # indices2 = [space.index(synset) + 1 for synset in synsets]
        Indices.append(indices)
        if dist == 'semcor':
            senseCount = [sum([SC[l.key()] for l in synset.lemmas() if l.key().startswith(word.lemma) and l.key().endswith('::')]) + 1 for synset in synsets]
            #senseCount = [sum([SC[k] for k in key]) + 1 for key in sense_keys]
            for ii in range(len(indices)):
                i = indices[ii]
                row[i] = senseCount[ii] / float(sum(senseCount))
        if use_softmax == True:
            rr = np.array(row)
            r = rr[rr > 0]
            X2 = np.exp(r) / np.sum(np.exp(r))
            rr[rr > 0] = X2
            row = rr.tolist()
        if dist == 'uniform':
            for ii in range(len(indices)):
                i = indices[ii]
                row[i] = 1 / float(len(indices))
        Sspace.append(row)
        tidxw += 1
    if save_matrices == True:
        path = 'vectors/' + ds + '/strat_space/' + dist
        create_directory(path)
        writeMat(Sspace, path + '/X-' + fileid)
    else:
        return Sspace
