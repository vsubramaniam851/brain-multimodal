import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from run_regression import *
from args import *

class Bootstrap(ViLRegression):
    def __init__(self, args):
        self.args = args
    
    def get_dataframes(self, args):
        return super().get_dataframes(args)

    def get_embeddings(self, args, stimulus_data):
        return super().get_embeddings(args, stimulus_data)

    def replace_sample_indices(self, args, response_data):
        bts_indices = np.empty(shape = (1000, response_data.shape[1]))
        for i in range(1000):
            new_indices = np.sort(np.random.randint(low = 0, high = response_data.shape[1], size = response_data.shape[1]))
            bts_indices[i] = new_indices
        if not os.path.exists('bootstrap_indices'):
            os.makedirs('bootstrap_indices')
        with h5py.File(os.path.join('bootstrap_indices', f'{args.subject}-{args.alignment}.h5'), 'w') as hf:
            hf.create_dataset('bootstrap_indices', data = bts_indices, dtype = int)

    def replace_sample_caller(self):
        response_data, _, _ = self.get_dataframes(args)
        self.replace_sample_indices(self.args, response_data)
    
    def replace_sample(self, feature_map, response_tensor, bts_indices, bts_idx):
        indices = bts_indices[bts_idx]
        return feature_map[indices, :], response_tensor[:, indices]
        
    def kfold_regression(self, X, y, n_splits=9, use_tqdm=True, alpha=100000):
        return super().kfold_regression(X, y, n_splits, use_tqdm, alpha)

    def get_actual_results(self, args):
        if self.args.randomized:
            randomized_str = '-randomized'
        else:
            randomized_str = ''
        self.actual_results =  pd.read_parquet(os.path.join(args.results_path, args.subject, f'{args.subject}_trial000_{args.alignment}_{args.model_name}{randomized_str}_{args.model_output}_200mswindow_results.parquet.gzip'))
        return self.actual_results
    
    def process_activity(self, response_data, actual_results):
        actual_results = actual_results[actual_results['model_layer'] == self.args.layer]
        best_per_electrode = actual_results.groupby(['electrode'])['test_score'].idxmax().tolist()
        return response_data[:, best_per_electrode]
    
    def get_alpha(self):
        self.alpha_results = self.actual_results['alpha'].to_numpy()
        return self.alpha_results

    def run_regression(self, args, bootstrap, response_tensor, response_data, feature_maps, use_tqdm=False):
        scoresheet_lists = {'srpr': []}
        feature_map_names = tqdm(feature_maps, desc = 'Mapping (layers)') if use_tqdm else feature_maps
        
        for model_layer in feature_map_names:
            _, bts_test_scores, _ = super().regress_model_layer(args, response_tensor, feature_maps[model_layer])
            electrodes = [idx[0] for idx in response_data.index]
            times = [idx[1] for idx in response_data.index]
            scoresheet = pd.DataFrame({'electrode': electrodes,
                                        'test_score': bts_test_scores,
                                        'times': times,
                                        'bootstrap': bootstrap,
                                        'score_type': 'pearson_r',
                                        'model': args.model_name,
                                        'train_type': 'trained' if not args.randomized else 'randomized',
                                        'model_layer': model_layer})
            scoresheet_lists['srpr'].append(scoresheet)
        
        results = pd.concat(scoresheet_lists['srpr'])
        return results

    def run_bootstrap(self, args, response_data, response_tensor, feature_maps, bootstrap_indices, bootstrap):
        new_feature_maps = {}
        new_feature_maps[list(feature_maps.keys())[0]], new_response_tensor = self.replace_sample(feature_maps[list(feature_maps.keys())[0]], response_tensor, bootstrap_indices, bootstrap)
        return self.run_regression(args, bootstrap, new_response_tensor, response_data, new_feature_maps)
    
    def bootstrap_test(self):
        results = []
        response_data, stimulus_data, _ = self.get_dataframes(args)
        response_tensor = self.preprocess_response_data(response_data)
        feature_maps = self.get_embeddings(args, stimulus_data)
        with h5py.File(f'bootstrap_indices/{args.subject}-{args.alignment}.h5', 'r') as hf:
            bts_indices = hf['bootstrap_indices'][:]
        for bootstrap in tqdm(range(self.args.num_bootstrap), desc = 'Iterating over bootstraps'):
            results.append(self.run_bootstrap(self.args, response_data, response_tensor, feature_maps, bts_indices, bootstrap))
        self.results = pd.concat(results).reset_index(drop = True)
        if self.args.randomized:
            randomized_str = '-randomized'
        else:
            randomized_str = ''
        if not os.path.exists(os.path.join(self.args.out_dir, self.args.subject)):
            os.makedirs(os.path.join(self.args.out_dir, self.args.subject))
        self.results.to_parquet(os.path.join(self.args.out_dir, self.args.subject, f'{self.args.subject}_trial000_{self.args.alignment}_{self.args.model_name}{randomized_str}_{args.model_output}_bootstrap.parquet.gzip'))

if __name__ == '__main__':
    args = bootstrap_args()
    bts_test = Bootstrap(args)
    # bts_test.replace_sample_caller()
    bts_test.bootstrap_test()