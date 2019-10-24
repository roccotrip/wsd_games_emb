from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))
lc = 'datasets/'
numbers = ['zero','one','two','three','four','five','six','seven','eighth','nine','ten','eleven','twelve','hundred','hundreds','thousand','thousands','i','ii','iii']

def map_pos_(pos):
    if pos.startswith('N'):
        return 'n'
    if pos.startswith('V'):
        return 'v'
    if pos.startswith('ADJ'):
        return 'a'
    if pos.startswith('ADV'):
        return 'r'
    return ''

def getAllWords(fileid,text,keyS):
    words = []
    ks = [key[:-1] for key in keyS if key.startswith(fileid)]
    kidx = 0
    for item in text.iter():
        if item.tag in ['wf','instance']:
            pos = item.get('pos')
            pos = map_pos_(pos)
            if pos == 's':
                pos = 'a'
            if pos not in ['']:
                lemma = item.get('lemma')
                if (lemma in stopset and item.tag == 'wf') or (lemma.isdecimal() and item.tag == 'wf') == True or (lemma in numbers and item.tag == 'wf') == True:
                    continue
                form = item.text
                wid = item.get('id')
                synsets = wn.synsets(lemma, pos)
                len_synsets = len(synsets)
                lemmas = [synset.lemmas() for synset in synsets]
                sense_keys = [[l.key() for l in lemma_] for lemma_ in lemmas]
                sense_keys = [[k for k in sense_key if (k.startswith(lemma+'%') and k.endswith('::')) or (k.startswith(lemma) and wn.lemma_from_key(k).synset().pos() == 's')] for sense_key in sense_keys]
                synsets = [synsets[x] for x in range(len(synsets)) if len(sense_keys[x]) > 0]
                sense_keys = [x for x in sense_keys if x != []]
                if len_synsets > len(synsets):
                    print('synset/s removed')
                if len(synsets) == 0 and wid is not None:
                    print('no sysets for an instance')
                    break
                if len(synsets) > 0:
                    keys = [[l.key() for l in synset.lemmas()] for synset in synsets]
                    keys = [k for sublist in keys for k in sublist]
                    senses = 0
                    k = 0
                    if item.tag == 'instance':
                        key = ks[kidx].split(' ')
                        kidx += 1
                        wid = key[0]
                        senses = key[1:]
                        senses = [sense for sense in senses if sense != '!!' and sense.startswith('lemma=') == False]
                        k = [wn.lemma_from_key(sense).synset() for sense in senses]
                        assert set(k).intersection(set(synsets)) != set()
                    word = Word(wid,lemma,pos,form,senses,synsets,k,keys, sense_keys)
                    words.append(word)
    return words

class Word(object):
    idd = ''
    lemma = ''
    pos = ''
    form = ''
    synsets = ''
    sent = 0
    key = ''

    def __init__(self, wid,lemma,pos,form,senses,synsets,k,keys,sense_keys):
        self.idd = wid
        self.lemma = lemma
        self.pos = pos
        self.form = form
        self.senses = senses
        self.synsets = synsets
        self.key = k
        self.keys = keys
        self.sense_keys = sense_keys
        if k != 0:
            assert set(keys).intersection(set(senses)) != set()
            assert set([x for sub in sense_keys for x in sub]).intersection(set(senses)) != set()

def getSynsets(lemma,pos):
    synsets = wn.synsets(lemma,pos)
    if synsets == [] and '_' in lemma:
        synsets = wn.synsets(lemma)
    if synsets == [] and '-' in lemma:
        synsets = wn.synsets(lemma.replace('-', ' '))
    return synsets
