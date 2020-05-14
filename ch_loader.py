import choquet_integral as ci
import csv
import numpy as np
import xai_indices as xai


data = []
inputs = []
labels = []
with open("bad_ascending_ch.csv", encoding='utf-8-sig') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        r = []
        l = []
        for j,col in enumerate(row):
            if(j < 4):
                r.append(float(row[j]))
            else:
                l.append(float(row[j]))

        inputs.append(r)
        labels.append(l)

inputs = np.asarray(inputs).transpose()
print(inputs)
print(labels)
labels = np.asarray(labels)

ch = ci.ChoquetIntegral()
ch.train_chi(inputs,labels)

(p,walks) = xai.percentage_walks(inputs)
print(xai.walk_centric_shapley(ch.fm,inputs))
print(ch.fm)
print("Coverage:", p)
print("Walks:", walks)
print(ch.chi_quad([1,.1,.2]))
#print(xai.walk_visitation(inputs))
#print(xai.percentage_walks(inputs))
#print(xai.harden_variable_visitation(inputs))


