import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import time
import math

slice_num = 20

def interval_trap(a,b,c,d,num_slices,max_height=1):
    y_interval = max_height / num_slices
    scale = 1/max_height
    res = []
    for i in range(num_slices):
        left = scale * (i+1)*y_interval * (b - a) + a
        right = -(scale * (i+1)*y_interval * (d-c) - d)
        res.append(((left,right), (i+1)*y_interval))
        
    return res

def eval_interval(x, intervals):
    minh = 0
    res = 0
    for i,((l,r),h) in enumerate(intervals):
        if(x < l or x > r):
            return res
        res = h
    ((l,r,),maxh) = intervals[-1]
    return (maxh)

def plot_interval(intervals):
    ax = plt.gca()
    color = next(ax._get_lines.prop_cycler)['color']
    for i,((l,r),h) in enumerate(intervals):
        x = [l,r]
        y = [h, h]
        plt.plot(x,y, color=color)
    plt.show()


def plot_intervals(multiple):
    ax = plt.gca()
    count = 0
    for intervals in multiple:
        color = next(ax._get_lines.prop_cycler)['color']
        perturb = count * .001
        for i,((l,r),h) in enumerate(intervals):
            x = [l,r]
            y = [h + perturb, h + perturb]
            plt.plot(x,y, color=color)
            
        count = count+1
    plt.show()

    
def plot_intervals2(multiple, res):
    ax = plt.gca()
    count = 0
    for intervals in multiple:
        color = next(ax._get_lines.prop_cycler)['color']
        perturb = count * .005
        for i,((l,r),h) in enumerate(intervals):
            x = [l,r]
            y = [h + perturb, h + perturb]
            plt.plot(x,y, color=color)

        count = count+1
    color = next(ax._get_lines.prop_cycler)['color']
    for i,((l,r),h) in enumerate(res):
        x = [l,r]
        y = [h,h]
        plt.plot(x,y,color=color)
    plt.show()

def interval_medioid(interval):
    top, bottom = 0, 0
    for i,((l,r),h) in enumerate(interval):
        top = top + (h * l) + (h * r)
        bottom = bottom + (h * 2)
    if bottom > 0:
        return top / bottom
    else:
        return -1

    



def genkeys(vals): # ex: [3,1,2] => [(3), (3,1), (3,1,2)] => [4, 5, 7]
    res = []
    for i in range(len(vals)):
        key = 0
        for val in vals[0:i+1]:
            key = set_bit(key,val-1,True)
        res.append(key)
    return res

def makekey(vals):
    key = 0
    for val in vals:
        key = set_bit(key,val-1,True)
    return key

def makefm(num_inputs, vals):
    if len(vals) != (np.power(2,num_inputs)-1):
        print("you fucked up")
    fm = {}
    fm[0] = 0
    count = 0
    for i in range(num_inputs):
        combs = list(itertools.combinations(np.arange(1,num_inputs+1),i+1))
        for val in combs:
            fm[makekey(val)] = vals[count]
            count = count + 1
    return fm

def set_bit(v, index, x):
    mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
    v &= ~mask          # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask         # If x was True, set the bit indicated by the mask.
        return v    


def choquet_height(fm,meds,heights, height_augment):
    
    sorted_meds = np.argsort(meds)[::-1]+1
    if(np.sum(height_augment) != 0):
        #print("a rule didn't fire")
        sorted_rule_index = np.argsort(meds)[::-1]+1
        for i in range(len(sorted_rule_index)):
            sorted_rule_index[i] = sorted_rule_index[i] + height_augment[sorted_rule_index[i]-1]
        med_keys = genkeys(sorted_rule_index)
    else:
        med_keys = genkeys(sorted_meds)
    gs = []
    for i in range(len(med_keys)):
        gs.append(fm[med_keys[i]])
        
    if(len(gs) > 0 and gs[-1] < 1):
        for i in range(len(gs)):
            gs[i] = gs[i] / gs[-1]
    b = 0
    for i in range(len(gs)):
        if i > 0:
            b = b + heights[sorted_meds[i]-1] * (gs[i] - gs[i-1])
        else:
            b = heights[sorted_meds[i]-1] * gs[i]
            
    return b


def choquet_height2(fm,meds,heights):
    sorted_meds = np.argsort(heights)[::-1]+1
    med_keys = genkeys(sorted_meds)
    gs = []
    for i in range(len(med_keys)):
        gs.append(fm[med_keys[i]])
        
    b = 0
    for i in range(len(gs)):
        if i > 0:
            b = b + heights[sorted_meds[i]-1] * (gs[i] - gs[i-1])
        else:
            b = heights[sorted_meds[i]-1] * gs[i]
    return b

def choquet(fm, ints):
    result = []
    meds = []
    heights = []
    height_augment = []
    height_rule_offset = 0
    for i in range(len(ints)):
        if ints[i] != []:  
            meds.append(interval_medioid(ints[i]))
            (bounds,h) = ints[i][-1]
            heights.append(h)
            height_augment.append(height_rule_offset)
        else:
            height_rule_offset = height_rule_offset + 1
            height_augment.append(height_rule_offset)     #if a rule doesn't fire, our list indices won't match rule numbers
    b = choquet_height(fm,meds,heights,height_augment)
    scale = 1
    for i in range(len(ints)):
        if len(ints[i]) > 0:
            scale = 1 / len(ints[i])
            break
    for i in range(len(ints[0])):
        ls, rs = [], []
        height = (i+1) * scale #normalization happens here
        rule_offset = 0
        arg_augments = []
        for j,fs in enumerate(ints):
            if(fs != []):
                ((l,r),h) = fs[i]
                ls.append(l)
                rs.append(r)
                arg_augments.append(rule_offset)
            else:
                rule_offset = rule_offset + 1
                arg_augments.append(rule_offset)
        
        sorted_ls = np.argsort(ls)[::-1]+1
        sorted_rs = np.argsort(rs)[::-1]+1
        
        if(np.sum(arg_augments) != 0):    #if a rule doesn't fire
            sorted_l_rule_index = np.argsort(ls)[::-1]+1
            sorted_r_rule_index = np.argsort(rs)[::-1]+1
            for j in range(len(sorted_l_rule_index)):
                sorted_l_rule_index[j] = sorted_l_rule_index[j] + arg_augments[sorted_l_rule_index[j]-1] #modify indices by offsets
                sorted_r_rule_index[j] = sorted_r_rule_index[j] + arg_augments[sorted_r_rule_index[j]-1]
            l_keys = genkeys(sorted_l_rule_index) #then generate keys with correct rule indices
            r_keys = genkeys(sorted_r_rule_index)
        else:
            l_keys = genkeys(sorted_ls)
            r_keys = genkeys(sorted_rs)
        lgs = []
        rgs = []
        for j in range(len(l_keys)):
            lgs.append(fm[l_keys[j]])
            rgs.append(fm[r_keys[j]])

        if(len(lgs) > 0 and lgs[-1] < 1):
            for j in range(len(lgs)):
                lgs[j] = lgs[j] / lgs[-1]
                rgs[j] = rgs[j] / rgs[-1]
        newL = 0
        newR = 0
        for j in range(len(lgs)):
            if j > 0:
                newL = newL + ls[sorted_ls[j]-1] * (lgs[j] - lgs[j-1])
                newR = newR + rs[sorted_rs[j]-1] * (rgs[j] - rgs[j-1])
            else:
                newL = ls[sorted_ls[j]-1] * (lgs[j])
                newR = rs[sorted_rs[j]-1] * (rgs[j])
        result.append(((newL,newR),height*b))
     
    #plot_intervals(ints)
    #plot_intervals2(ints,result)
    return result


def choquet2(fm, ints):
    result = []
    meds = []
    heights = []
    for i in range(len(ints)):
        meds.append(interval_medioid(ints[i]))
        (bounds,h) = ints[i][-1]
        heights.append(h)
    b = choquet_height2(fm,meds,heights)
    
    sorted_hs = np.argsort(heights)[::-1]+1
    h_keys = genkeys(sorted_hs)
    hgs = []
    for j in range(len(h_keys)):
            hgs.append(fm[h_keys[j]])
        
    for i in range(len(ints[0])):
        scale = 1/len(ints[0])
        ls, rs = [], []
        height = (i+1) * scale #normalization happens here
        for fs in ints:
            ((l,r),h) = fs[i]
            ls.append(l)
            rs.append(r)
        sorted_ls = np.argsort(ls)[::-1]+1
        sorted_rs = np.argsort(rs)[::-1]+1
        l_keys = genkeys(sorted_ls)
        r_keys = genkeys(sorted_rs)
        lgs = []
        rgs = []
        for j in range(len(l_keys)):
            lgs.append(fm[l_keys[j]])
            rgs.append(fm[r_keys[j]])

        newL = 0
        newR = 0
        for j in range(len(lgs)):
            if j > 0:
                newL = newL + ls[sorted_hs[j]-1] * (hgs[j] - hgs[j-1])
                newR = newR + rs[sorted_hs[j]-1] * (hgs[j] - hgs[j-1])
            else:
                newL = ls[sorted_hs[j]-1] * (hgs[j])
                newR = rs[sorted_hs[j]-1] * (hgs[j])
        result.append(((newL,newR),height*b))
     
    
    plot_intervals2(ints,result)
    return result

class Rule:
    def __init__(self, antecedents, ant_names, consequent):
        if not isinstance(antecedents, list):
            print("Antecedents should be in a list, even if there's only one.")
            exit(-1)
        self.antecedents = antecedents
        self.ant_names = ant_names
        self.consequent = consequent
        ((l,r),h) = consequent[-1]
        self.max_fire = h
        self.base = None
        self.domain = None

    def singleton_min(self, props):
        if not isinstance(props, list):
            print("Propositions should be given as list.")
            exit(-1)
        for i in range(len(props)):
            if len(self.antecedents[i]) != len(props[i]):
                print("Proposition ", i, " has length different than antecedent")
                exit(-4) 
        return fu.fast_min(props, self.antecedents, self.consequent)

    def singleton_prod(self, props):
        if not isinstance(props, list):
            print("Propositions should be given as list.")

        return fu.fast_prod(props, self.antecedents, self.consequent)
    
    def fire_interval_rule(self,vals):
        return interval_fast_min(vals,self.antecedents,self.consequent)


def interval_fast_min(props, ants, cons):
    res = 1
    for i in range(len(props)):
        res = min(res,eval_interval(props[i],ants[i]))
    result = []
    for i,((l,r),h) in enumerate(cons):
        if h <= res:
            result.append(((l,r),h))
    return result
            
def fill(ints, desired_slices):
    if len(ints) == desired_slices:
        return ints
    elif len(ints) > 0:
        ((l1,r1),h1) = ints[0]
        ((l2,r2),h2) = ints[-1]
        l_scale = (l2 - l1) / desired_slices
        r_scale = (r2 - r1) / desired_slices
        h_scale = (h2) / desired_slices
        res = []
        for i in range(desired_slices):
            res.append(((l1 + (i * l_scale), r1 + (i * r_scale)), ((i+1) * h_scale)))
        return res
    else:
        return []


class RuleBase:
    def __init__(self):
        self.domains = {}
        self.inputs = {}
        self.rules = []
        self.rule_activations = []
    
    def add_input_domain(self, name, domain_range):
        if name in self.inputs.keys():
            print(name, " already exists as domain name.")
            exit(-2)
        self.inputs[name] = domain_range

    def add_output_domain(self, name, domain_range):
        if name in self.domains.keys():
            print(name, " already exists as domain name.")
            exit(-2)
        self.domains[name] = domain_range

    def add_rule(self, rule, domain):
        for ant in rule.ant_names:
            if ant not in self.inputs.keys():
                print("Adding rule with undefined input domain")
             
        if domain not in self.domains.keys():
            print("Rule added for domain that doesn't exist")
            exit(-2)
#         if len(rule.consequent) is not self.domains[domain]:
#             print("Consequent has incorrect dimensions")
#             exit(-3)
        rule.base = self
        rule.domain = domain
        self.rules.append(rule)
        return rule

    def fire_rules(self, vals, prod=False):
        results = {}
        activation_percentages = []
        i = 0
        for rule in self.rules:
            
            prop_list = []
            for key in rule.ant_names:
                if key not in vals.keys() or key not in self.inputs.keys():
                    print("Issue with values provided.")
                    exit(-4)
                else:
                    prop_list.append(vals[key])
                    #prop_list.append(fu.singleton(self.inputs[key], vals[key]))
            rule_activation = []        
            if prod:
                rule_activation = rule.fire_interval_rule(prop_list)
            else:    
                rule_activation = rule.fire_interval_rule(prop_list)
            if rule.domain not in results.keys():
                results[rule.domain] = [fill(rule_activation,slice_num)]
            else:
                results[rule.domain].append(fill(rule_activation,slice_num))
            
            if(len(rule_activation) > 0):
                med = interval_medioid(rule_activation)
                ((l,r),h) = rule_activation[-1]
                activation_percentages.append(((h/rule.max_fire),med))
        self.rule_activations = activation_percentages
        #return results
        return results
                



# ChoquetBase = RuleBase()
# ChoquetBase.add_input_domain("color",100)
# ChoquetBase.add_input_domain("size",100)
# ChoquetBase.add_input_domain("distance",100)
# ChoquetBase.add_input_domain("recency",100)
# ChoquetBase.add_input_domain("velocity",100)
# ChoquetBase.add_output_domain("confidence",100)


# recent = interval_trap(0,0,3,10, slice_num)
# notRecent = interval_trap(7,11,100,100,slice_num)

# similarColor = interval_trap(0,0,20,40,slice_num)
# similarSize = interval_trap(75,90,100,100,slice_num)
# closeDistance = interval_trap(0,0,20,50,slice_num)

# dissimilarColor = interval_trap(35,70,100,100,slice_num)
# dissimilarSize = interval_trap(0,0,70,80,slice_num)
# farDistance = interval_trap(30,50,100,100,slice_num)

# highConf = interval_trap(85,90,100,100,slice_num)
# higherConf = interval_trap(95,97,99,100,slice_num)
# lowConf = interval_trap(0,0,10,15,slice_num)
# lowishConf = interval_trap(15,20,25,30,slice_num)
# medConf = interval_trap(45,50,55,60,slice_num)
# r1 = ChoquetBase.add_rule(Rule([similarColor,similarSize, closeDistance], ["color","size", "distance"], higherConf), "confidence")
# r2 = ChoquetBase.add_rule(Rule([similarColor,recent], ["color","recency"], highConf), "confidence")
# r3 = ChoquetBase.add_rule(Rule([dissimilarColor, closeDistance], ["color","distance"], lowConf), "confidence")
# r4 = ChoquetBase.add_rule(Rule([notRecent], ["recency"], lowishConf), "confidence")
#r5 = ChoquetBase.add_rule(Rule([notRecent], ["recency"], lowConf), "confidence")
#fm_vals = [1/3,1/3,1/3,2/3,2/3,2/3,1]
#fm = makefm(3,fm_vals)

# r1 = ChoquetBase.add_rule(Rule([similarColor], ["color"], higherConf), "confidence")
# r2 = ChoquetBase.add_rule(Rule([similarSize], ["size"], medConf), "confidence")
# r3 = ChoquetBase.add_rule(Rule([dissimilarColor, closeDistance], ["color","distance"], lowConf), "confidence")
# r4 = ChoquetBase.add_rule(Rule([notRecent], ["recency"], lowishConf), "confidence")
# fm_vals = [1/4,1/4,1/4,1/4,1/2,1/2,1/2,1/2,1/2,1/2,3/4,3/4,3/4,3/4,1]
# pess_fm = [1/8,1/8,1/8,1/8,1/4,1/4,1/4,1/4,1/4,1/4,3/4,3/4,3/4,3/4,1]
# opt_fm = [2/4,2/4,2/4,2/4,5/8,5/8,5/8,5/8,5/8,5/8,3/4,3/4,3/4,3/4,1]
# fm = makefm(4,opt_fm)


#fm_vals = [1/5,1/5,1/5,1/5,1/5,2/5,2/5,2/5,2/5,2/5,2/5,2/5,2/5,2/5,2/5,3/5,3/5,3/5,3/5,3/5,3/5,3/5,3/5,3/5,3/5,4/5,4/5,4/5,4/5,4/5,1]
#fm = makefm(5,fm_vals)

# vals = {}
# vals["color"] = 5
# vals["distance"] = 10
# vals["size"] = 85
# vals["recency"] = 10

# results = ChoquetBase.fire_rules(vals)
# decision = choquet(fm, results["confidence"])
# plots = results["confidence"]
# plots.append(decision)
# pretty_plot(plots)

# print(plots)
