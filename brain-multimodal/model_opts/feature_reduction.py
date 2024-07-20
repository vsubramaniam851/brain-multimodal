import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import os, sys, time, pickle, argparse
sys.path.append('..')

import torch as torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA

def check_reduction_inputs(feature_maps = None, model_inputs = None):
    if feature_maps == None and model_inputs == None:
        raise ValueError('Neither feature_maps nor model_inputs are defined.')
        
    if model_inputs is not None and not isinstance(model_inputs, (DataLoader, torch.Tensor)):
        raise ValueError('model_inputs not supplied in recognizable format.')

def get_feature_map_filepaths(feature_map_names, output_dir):
    return {feature_map_name: os.path.join(output_dir, feature_map_name + '.npy')
                                for feature_map_name in feature_map_names}

#source: stackoverflow.com/questions/26774892
def recursive_delete_if_empty(path):
    if not os.path.isdir(path):
        return False
    
    recurse_list = [recursive_delete_if_empty(os.path.join(path, filename))
                    for filename in os.listdir(path)]
    
    if all(recurse_list):
        os.rmdir(path)
        return True
    if not all(recurse_list):
        return False

def delete_saved_output(output_filepaths, output_dir = None, remove_empty_output_dir = False):
    for file_path in output_filepaths:
        os.remove(output_filepaths[file_path])
    if output_dir is not None and remove_empty_output_dir:
        output_dir = output_dir.split('/')[0]
        recursive_delete_if_empty(output_dir)
        

def torch_corrcoef(m):
    #calculate the covariance matrix
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov_m = 1 / (x.size(1) - 1) * x.mm(x.t())
    
    #convert covariance to correlation
    d = torch.diag(cov_m)
    sigma = torch.pow(d, 0.5)
    cor_m = cov_m.div(sigma.expand_as(cov_m))
    cor_m = cor_m.div(sigma.expand_as(cor_m).t())
    cor_m = torch.clamp(cor_m, -1.0, 1.0)
    return cor_m


#### Sparse Random Projection -------------------------------------------------------------------

def get_feature_map_srps(feature_maps, n_projections = None, upsampling = True, eps=0.1, seed = 0,
                        save_outputs = False, output_dir = 'srp_arrays', 
                        delete_originals = True, delete_saved_outputs = False):
    
    if n_projections is None:
        if isinstance(feature_maps, np.ndarray):
            n_samples = feature_maps.shape[0]
        if isinstance(feature_maps, dict):
            n_samples = next(iter(feature_maps.values())).shape[0]
        n_projections = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
        
    srp = SparseRandomProjection(n_projections, random_state=seed)
    
    def get_srps(feature_map):
        if feature_map.shape[1] <= n_projections and not upsampling:
            srp_feature_map = feature_map
        if feature_map.shape[1] >= n_projections or upsampling:
            srp_feature_map = srp.fit_transform(feature_map)
            
        return srp_feature_map
        
    if isinstance(feature_maps, np.ndarray) and save_outputs:
        raise ValueError('Please provide a dictionary of the form {feature_map_name: feature_map}' + 
                                 'in order to save_outputs.')
        
    if isinstance(feature_maps, np.ndarray) and not save_outputs:
        return srp.fit_transform(feature_maps)
         
    if isinstance(feature_maps, dict) and not save_outputs:
        srp_feature_maps = {}
        for feature_map_name, _ in tqdm(feature_maps.items(), desc = 'SRP Extraction (Layer)'):
            srp_feature_maps[feature_map_name] = get_srps(feature_maps[feature_map_name])
            
        if delete_originals:
            feature_maps.pop(feature_map_name)
            
        return srp_feature_maps
    
    if isinstance(feature_maps, dict) and save_outputs:
        output_dir = os.path.join(output_dir, '_'.join(['projections', str(n_projections), 'seed', str(seed)]))
        output_filepaths = get_feature_map_filepaths(feature_maps, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        srp_feature_maps = {}
        for feature_map_name in tqdm(list(feature_maps), desc = 'SRP Extraction (Layer)'):
            output_filepath = output_filepaths[feature_map_name]
            if not os.path.exists(output_filepath):
                srp_feature_maps[feature_map_name] = get_srps(feature_maps[feature_map_name])
                #np.save(output_filepath, srp_feature_maps[feature_map_name])
            if os.path.exists(output_filepath):
                srp_feature_maps[feature_map_name] = np.load(output_filepath, allow_pickle=True)
                
            if delete_originals:
                feature_maps.pop(feature_map_name)
                
        # if delete_saved_outputs:
        #     delete_saved_output(output_filepaths, output_dir, remove_empty_output_dir = True)
                
        return srp_feature_maps      
    
def srp_extraction(model_string, model = None, inputs = None, feature_maps = None, 
                   n_projections = None, upsampling = True, eps=0.1, seed = 0, 
                   output_dir='srp_arrays', delete_saved_outputs = True,
                   delete_original_feature_maps = False, verbose = False):
    
    check_reduction_inputs(feature_maps, inputs)
    output_dir_stem = os.path.join(output_dir, model_string)
        
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    
    if n_projections is None:
        if feature_maps is None:
            if isinstance(inputs, torch.Tensor):
                n_samples = len(inputs)
            if isinstance(inputs, DataLoader):
                n_samples = len(inputs.dataset)
        if feature_maps is not None:
            n_samples = next(iter(feature_maps.values())).shape[0]
        n_projections = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
        
    if verbose:
        print('Computing {} SRPs for {}; using {} for feature extraction...'
              .format(n_projections, model_string, device_name))
        
    output_dir_ext = '_'.join(['projections', str(n_projections), 'seed', str(seed)])
    output_dir = os.path.join(output_dir_stem, output_dir_ext)
            
    srp_args = {'feature_maps': feature_maps, 'n_projections': n_projections,
                'upsampling': upsampling, 'eps': eps, 'seed': seed,
                'save_outputs': True, 'output_dir': output_dir_stem,
                'delete_saved_outputs': delete_saved_outputs,
                'delete_originals': delete_original_feature_maps}
            
    return get_feature_map_srps(**srp_args)