import argparse

def data_args():
    parser = argparse.ArgumentParser()

    add_required_args(parser)
    add_data_args(parser)
    args = parser.parse_args()
    return args

def regression_args():
    parser = argparse.ArgumentParser()

    add_required_args(parser)
    
    parser.add_argument('-o', '--out_dir', action = 'store', type = str, default = 'results', dest = 'out_dir')
    add_model_args(parser)
    add_reg_args(parser)
    args = parser.parse_args()
    return args

def permutation_args():
    parser = argparse.ArgumentParser()

    add_required_args(parser)
    parser.add_argument('-o', '--out_dir', action = 'store', type = str, default = 'permutation_results', dest = 'out_dir')
    add_model_args(parser)
    add_reg_args(parser)
    add_perm_args(parser)
    add_mul_args(parser)
    args = parser.parse_args()
    return args

def bootstrap_args():
    parser = argparse.ArgumentParser()

    add_required_args(parser)
    parser.add_argument('-o', '--out_dir', action = 'store', type = str, default = 'bootstrap_results', dest = 'out_dir')
    add_model_args(parser)
    add_reg_args(parser)
    add_bootstrap_args(parser)
    args = parser.parse_args()
    return args

def add_required_args(parser):
    required_args = parser.add_argument_group('required_args')
    required_args.add_argument('-s', '--subject', action = 'store', type = str, dest = 'subject', help = 'Subject id for running regression', required = True)
    required_args.add_argument('-t', '--trial_list', action='store', dest = 'trial_list', nargs = '*', help='only gather data from these trial ids', required = True)
    required_args.add_argument('-a', '--alignment', action = 'store', type = str, dest = 'alignment', help = 'align to vision or language stimuli', required = True)
    required_args.add_argument('-w', '--window', action = 'store', type = int, dest = 'time_window', help = 'Time window activity is averaged over', required = True)

def add_data_args(parser):
    dataset_args = parser.add_argument_group('dataset_args')
    dataset_args.add_argument('--dataset_dir', type=str, default='/storage/datasets/neuroscience/ecog/', help='path to ecog data')
    dataset_args.add_argument('--cached_transcript_aligns', type=str, default=None, help='path to save/load aligned/filtered data')
    dataset_args.add_argument('--context_duration', type=float, default=4.5, help='how many seconds to take after word onset')
    dataset_args.add_argument('--context_delta', type=float, default=-2.0, help='how many seconds to take before word onset')
    dataset_args.add_argument('--electrode_list', action='append', default=None, help='only gather data from these electrodes')
    dataset_args.add_argument('--dataset_n_words', type=int, help='number of words to take from each trial. this argument is useful in debugging, since it allows us to make the dataset small.')
    dataset_args.add_argument('--sentence_context', action = 'store_true', dest = 'sentence_context')
    dataset_args.add_argument('--normalization', type=str, choices=['zscore'], default=None, help='whether to normalize electrode activity')
    dataset_args.add_argument('--rereference', type=str, choices=['CAR', 'laplacian'], default=None, help='whether to rereference electrodes')
    dataset_args.add_argument('--high_gamma', action = 'store_true', dest = 'high_gamma')
    dataset_args.set_defaults(high_gamma=False)
    dataset_args.add_argument('--despike', action = 'store_true', dest = 'despike')
    dataset_args.set_defaults(despike=False)

def add_model_args(parser):
    model_args = parser.add_argument_group('model_args')
    model_args.add_argument('-mn', '--model_name', action = 'store', type = str, default = 'lxmert', dest = 'model_name')
    model_args.add_argument('-mo', '--model_output', action = 'store', type = str, default = 'fusion', dest = 'model_output')
    model_args.add_argument('-l', '--layer', action = 'store', nargs = '*', default = [], dest = 'layers')
    model_args.add_argument('-b', '--batch_size', action = 'store', type = int, default = 8, dest = 'batch_size')
    model_args.add_argument('-pr', '--projection', action = 'store', type = str, default = 'srp', choices=['pca', 'srp'], dest = 'projection')
    model_args.add_argument('-r', '--randomized', action = 'store_true', dest = 'randomized')
    model_args.set_defaults(randomized = False)

def add_reg_args(parser):
    reg_args = parser.add_argument_group('reg_args')
    reg_args.add_argument('-p', '--path', action = 'store', type = str, dest = 'data_path', default = '/storage/vsub851/ecog-multimodal/data-by-subject')
    reg_args.add_argument('-ts', '--train_split', action = 'store', type = float, default = 0.9, dest = 'train_split')
    reg_args.add_argument('-al', '--alpha_val', action = 'store', type = float, dest = 'alpha_val', default = None)
    reg_args.add_argument('-ns', '--n_splits', action = 'store', type = int, dest = 'n_splits', default = 5) 

def add_perm_args(parser):
    perm_args = parser.add_argument_group('perm_args')
    perm_args.add_argument('-tp', '--type_perm', action = 'store', type = str, dest = 'type_perm', default = 'normal')  
    perm_args.add_argument('-np', '--num_permute', action = 'store', type = int, dest = 'num_permute', default = 1000)
    perm_args.add_argument('-sv', '--permute_start', action = 'store', type = int, dest = 'permute_start', default = 0)

def add_mul_args(parser):
    mul_perm_args = parser.add_argument_group('mul_perm_args')
    mul_perm_args.add_argument('-se', '--save_embeds', action = 'store', type = str, dest = 'save_embeds', default = '/storage/vsub851/ecog-multimodal/vil_embeds/permutation_embeds')
    mul_perm_args.add_argument('-pl', '--permute_lang', action = 'store_true', dest = 'permute_lang')
    mul_perm_args.add_argument('-le', '--load_embeds', action = 'store_true', dest = 'load_embeds')
    mul_perm_args.set_defaults(permute_lang = False, load_embeds = False)

def add_bootstrap_args(parser):
    bootstrap_args = parser.add_argument_group('bootstrap')
    bootstrap_args.add_argument('-nb', '--num_bootstrap', action = 'store', type = int, dest = 'num_bootstrap', default = 1000)
    bootstrap_args.add_argument('-rp', '--results_path', action = 'store', type = str, dest = 'results_path', default = '/storage/vsub851/ecog-multimodal/results')