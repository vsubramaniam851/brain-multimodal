import sys
import os

import numpy as np 
import pandas as pd 
import random
import cv2
import json
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from ridge import TorchRidge
from sklearn.model_selection import KFold
from torch.utils import data
from joblib import Parallel, delayed
from lavis.models import load_model_and_preprocess
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from torchmetrics.functional import pearson_corrcoef

from vil_embeds.vil_dataset import ViLDataset
from vil_embeds.vil_feature_extraction import *
sys.path.append('model_opts')
from feature_reduction import *
from mapping_methods import *
from make_plots import *
from args import *

if torch.cuda.is_available:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class ViLRegression(object):
    def __init__(self, args):
        self.args = args
    
    def regression_pipeline(self):
        trials_str = ' '.join(self.args.trial_list)
        if self.args.randomized:
            self.randomized_str = '-randomized'
        else:
            self.randomized_str = ''
        print(f'Vision-and-Language Regression on {self.args.subject}, {trials_str}, {self.args.model_name}{self.randomized_str} {self.args.model_output}')

        self.response_data, self.stimulus_data, self.electrode_metadata = self.get_dataframes(self.args)
        self.feature_maps = self.get_embeddings(self.args, self.stimulus_data)
        self.results = self.run_regression(self.args, self.response_data, self.feature_maps)

        trials_str = ''.join(self.args.trial_list)
        if not os.path.exists(self.args.out_dir):
            os.mkdir(self.args.out_dir)
        if not os.path.exists(os.path.join(self.args.out_dir, self.args.subject)):
            os.makedirs(os.path.join(self.args.out_dir, self.args.subject))

        self.results['srpr'].to_parquet(os.path.join(self.args.out_dir, self.args.subject, f'{self.args.subject}_{trials_str}_{self.args.alignment}_{self.args.model_name}{self.randomized_str}_{self.args.model_output}_{self.args.time_window}mswindow_results.parquet.gzip'))

    def get_dataframes(self, args):
        print(f'Getting {args.alignment}-aligned regression dataframes for time window {args.time_window}ms')
        subject_path = os.path.join(args.data_path, args.subject)
        assert os.path.exists(subject_path), 'Subject path needs to be fixed'

        trials_str = ''.join(args.trial_list)
        if args.alignment == 'language':
            response_data_path = os.path.join(subject_path, f'{trials_str}_word_response_data-{args.time_window}mswindow.parquet.gzip')
            stimulus_data_path = os.path.join(subject_path, f'{trials_str}_word_stimulus_metadata.csv')
        else:
            response_data_path = os.path.join(subject_path, f'{trials_str}_scene_response_data-{args.time_window}mswindow.parquet.gzip')
            stimulus_data_path = os.path.join(subject_path, f'{trials_str}_scene_stimulus_metadata.csv')
        assert os.path.exists(response_data_path), 'Response data path needs to be fixed'
        assert os.path.exists(stimulus_data_path), 'Stimulus data path needs to be fixed'
        electrode_metadata_path = os.path.join(subject_path, 'electrode_metadata.csv')
        assert os.path.exists(electrode_metadata_path), 'Electrode metadata path needs to be fixed'

        response_data = pd.read_parquet(response_data_path)

        stimulus_data = pd.read_csv(stimulus_data_path)
        if args.alignment == 'language':
            stimulus_data = stimulus_data.drop('Unnamed: 0', axis = 1)

        electrode_metadata = pd.read_csv(electrode_metadata_path)
        if args.subject != 'm00184':
            electrode_metadata = electrode_metadata.drop('Unnamed: 0', axis = 1).drop('Unnamed: 0.1', axis = 1).set_index('Electrode')
        else:
            electrode_metadata = electrode_metadata.drop('Unnamed: 0', axis = 1).set_index('Electrode')
        return response_data, stimulus_data, electrode_metadata

    def print_response_data_rows(self):
        print(self.response_data.head())

    def print_stimulus_data_rows(self):
        print(self.stimulus_data.head())

    def print_electrode_metadata_rows(self):
        print(self.electrode_metadata.head())

    def show_random_frame(self, savefig = False):
        num_stimuli = self.response_data.to_numpy().shape[1]
        random_idx = random.randint(1, num_stimuli-1)
        random_image_path = self.stimulus_data.loc[random_idx, 'image_path']
        img = cv2.imread(random_image_path)
        img = img[:, :, ::-1].copy()
        plt.imshow(img)
        if savefig:
            plt.savefig('example_frame.png')
        plt.close()
    
    def create_dataloader(self, args, stimulus_data):
        vis_processor = None
        text_processor = None
        if args.model_name in ['clip', 'slip', 'simclr', 'beit', 'convnext', 'albef', 'blip']:
            image_transforms = None
            use_cv2 = False
            if args.model_name in ['beit', 'convnext']:
                if args.model_name == 'beit':
                    model_str = 'beit_base_patch16_224'
                else:
                    model_str = 'convnext_base_in22k'
                config = resolve_data_config({}, model = model_str)
                image_transforms = create_transform(**config)
            elif args.model_name in ['albef', 'blip']:
                model_str = f'{args.model_name}_feature_extractor'
                _, vis_processor, text_processor = load_model_and_preprocess(model_str, is_eval = True, model_type = 'base')

        else:
            use_cv2 = True
            image_transforms = None
        vil_dataset = ViLDataset(image_paths = stimulus_data.image_path, contexts = stimulus_data.context, use_cv2 = use_cv2, 
                                    image_transforms = image_transforms, vis_processor = vis_processor, text_processor = text_processor)
        vil_dataloader = data.DataLoader(vil_dataset, batch_size = args.batch_size)
        return vil_dataset, vil_dataloader

    def get_embeddings(self, args, stimulus_data):
        vil_dataset, vil_dataloader = self.create_dataloader(args, stimulus_data)
        with open('vil_embeds/model_layer_dict.json', 'r') as f:
            model_layer_dict = json.load(f)
        if args.model_name != 'concat':
            assert args.model_name in model_layer_dict.keys(), f'Model is not an option: {list(model_layer_dict.keys())}'
            vil_layer_dict = model_layer_dict[args.model_name]
            assert args.model_output in vil_layer_dict.keys(), f'Output extraction is not an option in {args.model_name}, options: {list(vil_layer_dict.keys())}'
            layers_to_retain = vil_layer_dict[args.model_output]
            if args.layers: #Take subset of the layers
                assert all([x in layers_to_retain for x in args.layers]), f'Layers are not an option, choose from {layers_to_retain}'
                layers_to_retain = args.layers
                print(f'Taking subset of {args.model_output}, {args.layers}')
            flatten = False
            dim_reduction = False

            #This code checks whether sparse random projection is completely necessary. 
            if 'output' not in args.model_output and 'best_layer' not in args.model_output and 'best_vis_layer' not in args.model_output:
                flatten = True
                dim_reduction = True
            elif ('output' in args.model_output or 'best_layer' in args.model_output) and args.model_name == 'convnext':
                flatten = True
            if 'best_layer' in args.model_output and args.model_name in ['blip', 'albef', 'visual_bert']:
                flatten = True
                dim_reduction = True
            
            feature_maps = run_model(model = args.model_name, inputs = vil_dataloader, layers_to_retain = layers_to_retain, flatten = flatten, batch_size = args.batch_size, randomized = args.randomized)
            if dim_reduction:
                new_feature_maps = dict()
                new_feature_maps = srp_extraction(model_string = args.model_name, feature_maps = feature_maps, eps = 0.1, seed = 0, delete_original_feature_maps = True)
                feature_maps = new_feature_maps
        else:
            feature_maps = self.concat_vil(args, model_layer_dict, stimulus_data)
        self.print_feature_map_shape(feature_maps)
        return feature_maps

    def concat_vil(self, args, model_layer_dict, stimulus_data):
        args.model_name = 'simcse'
        _, vil_dataloader = self.create_dataloader(args, stimulus_data)
        lang_layer_dict = model_layer_dict['simcse']
        layers_to_retain = lang_layer_dict['best_layer']
        lang_feature_map = run_model(model = 'simcse', inputs = vil_dataloader, layers_to_retain = layers_to_retain, flatten = False, batch_size = 8, randomized = False)
        args.model_name = 'simclr'
        _, vil_dataloader = self.create_dataloader(args, stimulus_data)
        vis_layer_dict = model_layer_dict['simclr']
        layers_to_retain = vis_layer_dict['best_layer']
        vis_feature_map = run_model(model = 'simclr', inputs = vil_dataloader, layers_to_retain = layers_to_retain, flatten = False, batch_size = 8, randomized = False)
        assert len(lang_feature_map) == 1 and len(vis_feature_map) == 1
        lang_features = lang_feature_map[list(lang_feature_map.keys())[0]]
        vis_features = vis_feature_map[list(vis_feature_map.keys())[0]]
        concat_features = np.concatenate([lang_features, vis_features], axis = -1)
        return {'vil-concat': concat_features}

    def print_feature_map_shape(self, feature_maps):
        for feature_map in feature_maps:
            print(feature_map, feature_maps[feature_map].shape)
        
    def preprocess_response_data(self, response_data):
        response_data = torch.tensor(response_data.to_numpy()).type(torch.FloatTensor).to(DEVICE)
        return response_data

    def replace_sample(self, feature_map, response_data):
        new_indices = np.load('sampled_indices.npy')
        return feature_map[new_indices, :], response_data[:, new_indices]

    def preprocess_xy(self, args, feature_map, response_data):
        transformed_data = {'X': {}, 'y': {}}
        transformed_data['X'] = torch.tensor(feature_map).type(torch.FloatTensor).to(DEVICE)
        transformed_data['y'] = response_data
        return transformed_data
    
    def standard_scaler_fit(self, X):
        m = X.mean(0, keepdim=True)
        s = X.std(0, unbiased=False, keepdim=True)
        return m, s
    
    def standard_scaler_transform(self, X, m, s):
        return (X - m)/s
    
    def torch_pearsonr(self, pred, target):
        #From https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739
        v_pred = pred - torch.mean(pred)
        v_target = target - torch.mean(target)
        output = torch.sum(v_pred * v_target, dim = 0) / (torch.sqrt(torch.sum(v_pred ** 2, dim = 0)) * torch.sqrt(torch.sum(v_target ** 2, dim = 0)))
        return output

    def kfold_regression(self, X, y, n_splits = 9, use_tqdm = False, alpha = 100000.0):
        regression = TorchRidge(alpha = alpha, device = DEVICE)
        
        kfolds = KFold(n_splits, shuffle=False).split(np.arange(y.shape[0]))
        kfolds = tqdm(kfolds, total = n_splits, desc = f'Mapping (layers) for alpha {alpha}') if use_tqdm else kfolds
        
        val_scores = []
        train_scores = []
        test_scores = []
        for train_indices, val_test_indices in kfolds:
            val_indices = val_test_indices[:len(val_test_indices)//2]
            test_indices = val_test_indices[len(val_test_indices)//2:]
            X_train, X_val, X_test = X[train_indices, :], X[val_indices, :], X[test_indices, :]
            m, s = self.standard_scaler_fit(X_train)
            X_train, X_val, X_test = self.standard_scaler_transform(X_train, m, s), self.standard_scaler_transform(X_val, m, s), self.standard_scaler_transform(X_test, m, s)
            y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
            regression.fit(X_train, y_train)
            y_val_pred = regression.predict(X_val)
            y_test_pred = regression.predict(X_test)
            val_score = pearson_corrcoef(y_val_pred, y_val)
            val_scores.append(val_score.detach().cpu().numpy())
            test_score = pearson_corrcoef(y_test_pred, y_test)
            test_scores.append(test_score.detach().cpu().numpy())
        val_scores = np.vstack(val_scores)
        test_scores = np.vstack(test_scores)
        return regression, np.mean(test_scores, axis = 0), np.mean(val_scores, axis = 0)
    
    def regress_model_layer(self, args, response_tensor, feature_map, use_tqdm = True):
        xy = self.preprocess_xy(args, feature_map, response_tensor)
        if args.alpha_val is None:
            alpha_vals = np.logspace(-1, 5, 7)
        else:
            alpha_vals = [args.alpa_val]

        val_alpha = []
        test_alpha = []
        for alpha in alpha_vals:
            output = self.kfold_regression(xy['X'], xy['y'].T, n_splits = args.n_splits, alpha = alpha.item(), use_tqdm = False)
            val_alpha.append(output[2])
            test_alpha.append(output[1])
        
        val_alpha = np.stack(val_alpha, axis = -1)
        test_alpha = np.stack(test_alpha, axis = -1)
        best_alpha_indices = np.argmax(val_alpha, axis = -1)
        best_val_scores = np.array([val_alpha[i, idx] for i, idx in enumerate(best_alpha_indices)])
        best_test_scores = np.array([test_alpha[i, idx] for i, idx in enumerate(best_alpha_indices)])
        alpha_vals = alpha_vals[best_alpha_indices]
        return best_val_scores, best_test_scores, alpha_vals

    def run_regression(self, args, response_data, feature_maps, use_tqdm = True):
        scoresheet_lists = {'srpr': []}
        feature_map_names = tqdm(feature_maps, desc = 'Mapping (layers)') if use_tqdm else feature_maps
        response_tensor = self.preprocess_response_data(response_data)
        for model_layer in feature_map_names:
            best_val_scores, best_test_scores, alpha_vals = self.regress_model_layer(args, response_tensor, feature_maps[model_layer])
            electrodes = [idx[0] for idx in response_data.index]
            times = [idx[1] for idx in response_data.index]
            scoresheet = pd.DataFrame({'electrode': electrodes,
                                        'times': times,
                                        'test_score': best_test_scores,
                                        'val_score': best_val_scores,
                                        'alpha': alpha_vals,
                                        'score_set': 'test',
                                        'score_type': 'pearson_r',
                                        'model': args.model_name,
                                        'train_type': 'trained' if not args.randomized else 'randomized',
                                        'model_layer': model_layer})
            scoresheet_lists['srpr'].append(scoresheet)
        
        results = {}
        results['srpr'] = pd.concat(scoresheet_lists['srpr'])
        return results

    def show_results(self):
        print(self.results['srpr'].head())

if __name__ == '__main__':
    args = regression_args()
    vil_regression = ViLRegression(args)
    vil_regression.regression_pipeline()
    vil_regression.show_results()
    pass