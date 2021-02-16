import sys
sys.path.append('./atlas_scorer/')
import os
from os import path
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from atlas_scorer.score import Scorer
import argparse
from matplotlib.font_manager import FontProperties
import json
import colorsys
import random
import math
import datetime

class blank_roc:
    pf = []
    pd = []
    far = []

    def __init__(self):
        self.pf = []
        self.pf.append(0)
        self.pf.append(1)

        self.pd = []
        self.pd.append(0)
        self.pd.append(0)

        self.far = []
        self.far.append(0)
        self.far.append(1)

    def plot_roc(self, title, label):
        return

    def plot_far(self, title, label):
        return

def generate_roc(test_dir, n_exp, truth_dir, out_dir, roc_title, run_names, score_mode, score_target):
    dfs = []
    fig, ax = plt.subplots()
    plt_handles = []

    num_colors = len(run_names)
    colors = []
    for i in range(0, len(run_names)):
        hue = float(i) * (360.0 / float(num_colors)) 
        saturation = 90 + random.random() * 10
        value = 50 + random.random() * 10

        hue = hue / 360.0
        saturation = saturation / 100.0
        value = value / 100.0

        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))

    for i in range(0, len(run_names)):

        resampled_x_vals = []
        resampled_y_vals = []

        for exp in range(0, n_exp):
            decl_json_file_path = os.path.join(test_dir + "/test_" + str(exp) + "/decl_files", f'{run_names[i]}.json')
            decl_files  = [decl_json_file_path]
            truth_json_file_path = os.path.join(truth_dir, f'{run_names[i]}.truth.json')
            truth_files = [truth_json_file_path]

            missing_file = False
            
            if not path.exists(decl_json_file_path):
                print("Decl json file doesn't exist. Skipping." + decl_json_file_path)
                missing_file = True
      
            if not path.exists(truth_json_file_path):
                print("Truth json file doesn't exist. Skipping." + truth_json_file_path)
                missing_file = True

            if not missing_file:
                n_score = n_exp * len(run_names) + 1
                cur_score = i*n_exp + exp + 1

                if score_mode == 0 or score_mode == 1:
                    # roc, _ = scorer.aggregate(alg='detection')
                    print("Needs to be fixed.")
                    exit()
                elif score_mode == 2:
                    # Check for empty declaration file. 
                    decl_json = json.load(open(decl_json_file_path, 'r'))
                    empty_decl = False
                    if len(decl_json['frameDeclarations']) == 0:
                        empty_decl = True

                    truth_json = json.load(open(truth_json_file_path, 'r'))
                    target_present = False
                    for frame in truth_json['frameAnnotations'].keys():
                        for decl in truth_json['frameAnnotations'][frame]['annotations']: 
                            try:
                                targetName = decl['metadata']['target']
                            except:
                                targetName = decl['class']

                            if targetName == score_target:
                                target_present = True

                    if target_present and not empty_decl:
                        print("\nInitializing scorer..." + str(cur_score) + '/' + str(n_score)+ ' | ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        scorer = Scorer()
                        print("Loading truth and decl files..." + str(cur_score) + '/' + str(n_score)+ ' | ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        scorer.load(truth_files, decl_files)
                        print("Linking..." + str(cur_score) + '/' + str(n_score)+ ' | ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        scorer.link()
                        print("Aggregating..." + str(cur_score) + '/' + str(n_score)+ ' | ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        print(score_target)
                        # roc, _, _ = scorer.aggregate(alg='detection', score_class=score_target)
                        score_results = list(scorer.aggregate(alg='detection', score_class=score_target))
                        if len(score_results) == 0:
                            print("Scoring error. Empty results returned.")
                            cur_label = run_names[i] + '_[Scoring Error]'
                            roc = blank_roc()
                        else:
                            roc = score_results[0]
                        cur_label = run_names[i]
                    elif not target_present and not empty_decl:
                        print("Score target " + score_target + " not found in truth json.")
                        cur_label = run_names[i] + '_[Not in Truth]'
                        roc = blank_roc()
                    elif target_present and empty_decl:
                        print("Empty declaration file found.")
                        cur_label = run_names[i] + '_[Empty Decl]'
                        roc = blank_roc()
                    else:
                        print("Empty declaration file found and score target " + score_target + " not found in truth json.")
                        cur_label = run_names[i] + '_[Not in Truth and Empty Decl]'
                        roc = blank_roc()
                else:
                    print("Error: Unrecognized socring mode.")
                    exit()
                
                print("Plotting..." + str(cur_score) + '/' + str(n_score)+ ' | ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                if score_mode == 0:
                    roc.plot_roc(title=roc_title, label=cur_label+'_exp_'+str(exp))
                    df = pd.DataFrame(index=np.arange(len(roc.pf)), columns=['pd', 'pf'])
                    df['pf'] = roc.pf
                elif score_mode == 1 or score_mode == 2:
                    roc.plot_far(title=roc_title, label=cur_label+'_exp_'+str(exp))
                    df = pd.DataFrame(index=np.arange(len(roc.pf)), columns=['pd', 'far'])
                    df['far'] = roc.far
                
                df['pd'] = roc.pd
                dfs.append(df)

                # Reset x-axis lims since default of ~0 doesn't work well with log scale
                if score_mode == 0:
                    x_vals = list(df['pf'])
                    y_vals = list(df['pd'])
                elif score_mode == 1 or score_mode == 2:
                    x_vals = list(df['far'])
                    y_vals = list(df['pd'])

                # Add final value. 
                x_vals.append(1)
                y_vals.append(y_vals[-1])
                    
                # Uniformly resample curves.
                x_vals, y_vals = uniform_resample(x_vals, y_vals)
                resampled_x_vals.append(x_vals)
                resampled_y_vals.append(y_vals)

        # Plot average curve, fill between min and max. 
        min_vals, max_vals, avg_vals = get_min_max_avg_curves(resampled_x_vals, resampled_y_vals)
        pd_val = "%.2f" % get_pd_at_far(resampled_x_vals[0], avg_vals, 0.025)
        h1, = ax.plot(resampled_x_vals[0], avg_vals, label = cur_label + ', pd@0.025=' + pd_val, linewidth=3, color = colors[i])
        plt_handles.append(h1)
        ax.fill_between(resampled_x_vals[0], min_vals, max_vals, alpha=0.333, color = colors[i])
        ax.plot(resampled_x_vals[0], min_vals, '--', label = cur_label, linewidth=1, color = colors[i])
        ax.plot(resampled_x_vals[0], max_vals, '--', label = cur_label, linewidth=1, color = colors[i])
    
    # Add axis labels and titles. 
    if score_mode == 1 or score_mode == 2:
        ax.set_xlabel('FAR')
    else:
        ax.set_xlabel('PF')
    ax.set_ylabel('PD')
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 1)
    ax.set_title(roc_title)
    # ax.set_facecolor((0.15, 0.15, 0.15))
    ax.grid()
    fig.legend(handles=plt_handles, loc='center right')
    fig.show()
    out_filename = out_dir + '/' + roc_title
    fig.savefig(out_filename)
    print("Figure saved to " + out_filename)
    plt.close('all')

def get_pd_at_far(x_vals, y_vals, far):
    ind1 = 1
    while(x_vals[ind1] < far):
        ind1 = ind1 + 1
    ind0 = ind1 - 1
    pd_val = linear_interpolation(x_vals[ind0], x_vals[ind1], y_vals[ind0], y_vals[ind1], far)

    return pd_val

# Computes min, max and average of multiple curves. 
def get_min_max_avg_curves(resampled_x_vals, resampled_y_vals):
    min_vals = []
    max_vals = []
    avg_vals = [] 
    for x in range(0, len(resampled_x_vals[0])):
        min_val = float("inf")
        max_val = float("-inf")
        avg_val = 0.0
        for a in range(0, len(resampled_y_vals)):
            if resampled_y_vals[a][x] < min_val:
                min_val = resampled_y_vals[a][x]
            if resampled_y_vals[a][x] > max_val:
                max_val = resampled_y_vals[a][x]
            avg_val = avg_val + resampled_y_vals[a][x]
        avg_val = avg_val / float(len(resampled_y_vals))
        min_vals.append(min_val)
        max_vals.append(max_val)
        avg_vals.append(avg_val)

    return min_vals, max_vals, avg_vals


def linear_interpolation(xa,xb,ya,yb,xc):
    if xa == xb:
        return float(ya)
        
    xa = float(xa)
    xb = float(xb)
    ya = float(ya)
    yb = float(yb)
    xc = float(xc)

    m = (ya - yb) / (xa - xb)
    yc = (xc - xb) * m + yb

    return yc

# Resamples a curve using linear interpolation. 
def uniform_resample(x_vals, y_vals):
    # Generate new x values
    n_samples = 1000
    new_x_vals = list(range(0, n_samples+1))
    for a in range(0, len(new_x_vals)):
        new_x_vals[a] = float(new_x_vals[a]) / float(n_samples)

    # Find indexes and interpolate. 
    new_y_vals = []
    for a in range(0, n_samples):
        ind_t1 = 0
        while ind_t1 < len(x_vals):
            if new_x_vals[a] < x_vals[ind_t1]:
                break
            ind_t1 = ind_t1 + 1
        ind_t0 = ind_t1 - 1
        if ind_t1 == 0:
            ind_t0 = 0
        new_y_vals.append(linear_interpolation(x_vals[ind_t0], x_vals[ind_t1], y_vals[ind_t0], y_vals[ind_t1], new_x_vals[a]))
    new_y_vals.append(y_vals[-1])

    return new_x_vals, new_y_vals


if __name__ == '__main__':
    scorer = Scorer()
    fig, ax = plt.subplots()

    a10_a10 = "./a10/afternoon10m.json"
    a10_a25 = "./a10/afternoon25m.json"
    # decl_files = [decl_json_file_path,decl_file2]
    a10_truth = "./a10/afternoon10m.truth.json"
    a25_truth = "./a25/afternoon25m.truth.json"
    # truth_files = [truth_json_file_path,truth_file2]

    a25_a25 = "./a25/afternoon25m.json"
    a25_a10 = "./a25/afternoon10m.json"

    a10_fused_a10context = "./fused_decls/a10fuseda10context.json"
    a10_fused_a25context = "./fused_decls/a10fuseda25context.json"
    a25_fused_a10context = "./fused_decls/a25fuseda10context.json"
    a25_fused_a25context = "./fused_decls/a25fuseda25context.json"

    s = Scorer()
    s.load([a10_truth], [a10_a10])
    roc = s.score()
    x_vals = roc.pf
    y_vals = roc.pd
    # x_vals = np.append(x_vals,1)
    # y_vals = np.append(y_vals,y_vals[-1])
    # x_vals, y_vals = uniform_resample(x_vals, y_vals)
    plt.plot(x_vals,y_vals,label="A10 on A10")
    
    s.reset()
    s.load([a25_truth], [a10_a25])
    roc = s.score()
    x_vals = roc.pf
    y_vals = roc.pd
    # x_vals = np.append(x_vals,1)
    # y_vals = np.append(y_vals,y_vals[-1])
    plt.plot(x_vals,y_vals,label="A10 on A25")

    # s.reset()
    # s.load([])

    s.reset()
    s.load([a10_truth],[a10_fused_a10context])
    roc = s.score()
    x_vals = roc.pf
    y_vals = roc.pd
    # x_vals = np.append(x_vals,1)
    # y_vals = np.append(y_vals,y_vals[-1])
    plt.plot(x_vals,y_vals)

    s.reset()
    s.load([a10_truth],[a10_fused_a25context])
    roc = s.score()
    x_vals = roc.pf
    y_vals = roc.pd
    plt.plot(x_vals,y_vals,label="Fused A10, wrong context")
 
    s.reset()
    s.load([a25_truth],[a25_a25])
    roc = s.score()
    x_vals = roc.pf
    y_vals = roc.pd
    plt.plot(x_vals,y_vals,label="A25 on A25")
    
    s.reset()
    s.load([a10_truth],[a25_a10])
    roc = s.score()
    x_vals = roc.pf
    y_vals = roc.pd
    plt.plot(x_vals,y_vals,label="A25 on A10")

    s.reset()
    s.load([a25_truth],[a25_fused_a10context])
    roc = s.score()
    x_vals = roc.pf
    y_vals = roc.pd
    plt.plot(x_vals,y_vals,label="Fused A25, wrong context")

    s.reset()
    s.load([a25_truth],[a25_fused_a25context])
    roc = s.score()
    x_vals = roc.pf
    y_vals = roc.pd
    plt.plot(x_vals,y_vals,label="Fused A25, right context")

    # s.reset()
    # s.load([a10_truth],[a10_fused_a10context])
    # roc = s.score()
    # x_vals = roc.far
    # y_vals = roc.pd
    # plt.plot(x_vals,y_vals, '--', label="Fused A10, right context")
    # plt.legend()
    # plt.show()
    # scorer.load(truth_files,decl_files)

    # score_results = list(scorer.aggregate(alg='detection',score_class="Explosive_Hazard"))
    # roc = score_results[0]

    # roc.plot_roc(title="Afternoon", label="test")
    # df = pd.DataFrame(index=np.arange(len(roc.pf)),columns=['pd','pf'])
    # df['pf'] = roc.pf
    # df['pd'] = roc.pd
    # x_vals = list(df['pf'])
    # y_vals = list(df['pd'])
    # x_vals.append(1)
    # y_vals.append(y_vals[-1])

    # # pd_val = "%.2f" % get_pd_at_far(resampled_x_vals[0], avg_vals, 0.025)
    # h1, = ax.plot(x_vals, y_vals, label = "hey" + ', pd@0.025=' + "bruh", linewidth=3)
    plt.legend()
    plt.show()
    