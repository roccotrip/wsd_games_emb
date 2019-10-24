from collections import Counter
import pickle

with open('/Users/rocco/Desktop/WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt') as f:
    rows = f.readlines()
    keys = [line.split(' ')[1][:-1] for line in rows]
dist = Counter(keys)

with open('/Volumes/T5/python/WSD_EMB/data/SemCor/SC+OMSTI.pk', 'wb') as f:
    pickle.dump(dist,f)

with open('/Users/rocco/Desktop/WSD_Training_Corpora/SemCor/semcor.gold.key.txt') as f:
    rows = f.readlines()
    keys = [line.split(' ')[1][:-1] for line in rows]
dist = Counter(keys)

with open('/Volumes/T5/python/WSD_EMB/data/SemCor/SC2.pk', 'wb') as f:
    pickle.dump(dist,f)