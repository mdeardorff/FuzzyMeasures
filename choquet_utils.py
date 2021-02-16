import pyemd
from choquet_integral import *
import random
import numpy as np
import xai_indices as xai
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits import mplot3d
from PIL import ImageDraw, Image, ImageColor, ImageFont
import itertools
import cv2
import time
import copy
import pandas
import pyemd

def random_node_fm(n):
    ch = ChoquetIntegral()
    ch.type='quad'
    ch.fm = {}
    ch.fm[str(np.arange(1,n+1))] = 1
    ch.fm['[]'] = 0
    ch.N = n
    ch.M = n
    numkeys = len(ch.get_keys_index())
    keys = list(ch.get_keys_index().keys())
    keys.append('[]')
    s = [0] * (numkeys + 1)
    s[-1] = 1
    s[-2] = 1
    done = False
    while not done:
        randindex = random.randrange(0,numkeys)
        if s[randindex] == 0:
            s[randindex] = 1
            if keys[randindex] != '[]':
                compare_key = [int(s) for s in keys[randindex][1:-1].split() if s.isdigit()]
            else:
                compare_key = []
                
            maxi = 0
            max_index = 0
            mini = 1
            min_index = 0
            for i,key in enumerate(keys):
                if s[i] == 1 and i != randindex:
                    if key != '[]':
                        stripped = [int(s) for s in key[1:-1].split() if s.isdigit()]
                    else:
                        stripped = []
                        
                    if is_subset(stripped,compare_key) and s[i] == 1 and ch.fm[key] >= maxi:
                        maxi = ch.fm[key]
                        max_index = i
                    if is_subset(compare_key,stripped) and s[i] == 1 and ch.fm[key] <= mini:
                        mini = ch.fm[key]
                        min_index = i

            rb = ch.fm[keys[max_index]]
            ru = ch.fm[keys[min_index]]
            g = random.uniform(rb,ru)
            ch.fm[keys[randindex]] = g


        else:
            pass

        if min(s) == 1:
            done = True

    return ch

def is_subset(a,b):
    if len(a) == 0:
        return True
    if len(b) == 0 and len(a) != 0:
        return False

    for val in a:
        if val not in b:
            return False

    else:
        return True

    

def gen_unit_distance_matrix(n):
    distance_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = np.abs(i-j)
    return distance_matrix 


def gen_datapoints(m,n):
    points = []
    for i in range(n):
        point = []
        for j in range(m):
            point.append(random.random())
        points.append(point)
    return np.asarray(points)


def sample_with_noise(ch,data,mean,var):
    labels = np.zeros(data.shape[0])
    for i,point in enumerate(data):
        labels[i] = max(min(ch.chi_quad(point) + random.gauss(mean,var),1),0)

    return labels


def percentage_walks_observed(walks):
    seen = 0
    total = 0
    for key in walks.keys():
        if walks[key] > 0:
            seen = seen + 1
        total = total+1

    return seen/total

def average_fms(fms):
    res_fm = {}
    for key in fms[0].keys():
        val = 0
        for fm in fms:
            val = val + float(fm[key])
        val = val / len(fms)
        res_fm[key] = val
    return res_fm