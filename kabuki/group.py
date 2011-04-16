from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

import kabuki

def convert_model_to_dictionary(model):
    """convert_model_to_dictionary(model)
    transform a set or list of nodes to a dictionary
    """
    d = {}
    for node in model:
        d[node.__name__] = node
    return d

def get_root_nodes(model):
    """
    get_root_nodes(model)
    get only the root nodes from the model
    """
    
    if type(model)==type({}):
        nodes = model.values()
    else:
        nodes = model
    
    root = [z for z in nodes if re.match('[0-9]',z.__name__[-1]) is None]
    
    if type(model) == type({}):
        return convert_model_to_dictionary(root)
    else:
        return root    

def get_subj_nodes(model, i_subj=None):
    """get_subj_nodes(model, i_subj=None):
    return the nodes of subj i_subj. if is_subj is None then return all subjects' node
    if i_subj is -1, return root nodes
    """ 
    pass

    if i_subj==-1:
        return get_root_nodes(model)
    else: 
        if type(model) == type({}):
            nodes = model.values()
        else:
            nodes  = model
        
        if i_subj is None:        
            subj = [z for z in nodes if re.match('[0-9]',z.__name__[-1]) is not None]
        else:
            s_subj = str(i_subj)
            subj = [z for z in nodes if z.__name__[-len(s_subj):] == s_subj]
        
        if type(model) == type({}):
            return convert_model_to_dictionary(subj)
        else:
            return subj