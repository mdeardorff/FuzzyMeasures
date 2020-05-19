from choquet_integral import *
from emd_clustertend import vat, ivat
import csv

keys = []
for g in range(1,2**7):
    key = []
    for i,val in enumerate(format(g,'b')[::-1]):
        if val == '1':
            key.append(i+1)
    keys.append(np.asarray(key))

learner_fm = {}

with open('FMs/fmvars_shared_0_0.txt', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i,row in enumerate(reader):
        learner_fm[str(keys[i])] = float(row[0])

learner_ch = ChoquetIntegral()
learner_ch.N = 7
learner_ch.M = 7
learner_ch.type = 'quad'
learner_ch.fm = learner_fm
learner_diffs = learner_ch.generate_walk_diffs()
print("done with diffs")

vat(learner_diffs,euclidean=True)