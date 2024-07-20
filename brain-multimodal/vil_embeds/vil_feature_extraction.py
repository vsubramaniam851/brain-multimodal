import os, sys, shutil
import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
from warnings import warn
from tqdm.auto import tqdm as tqdm
from collections import defaultdict, OrderedDict
import transformers
transformers.utils.logging.set_verbosity(40)
import clip
sys.path.append('vil_embeds/SLIP')
from models import CLIP_VITB16, SIMCLR_VITB16, SLIP_VITB16
from tokenizer import SimpleTokenizer

from PIL import Image
from torch.utils import data 
import torch.nn as nn
import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from clip.model import CLIP
from sentence_transformers import SentenceTransformer, models
from timm import create_model
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from lavis.models import load_model_and_preprocess, model_zoo
from .vil_dataset import ViLDataset

def convert_relu(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu(child)

def get_weights_dtype(model):
    module = list(model.children())[0]
    if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
        return module.weight.dtype
    if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
        return get_weights_dtype(module)

def get_module_name(module, module_list):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    class_count = str(sum(class_name in module for module in module_list) + 1)
    
    return '-'.join([class_name, class_count])

def get_inputs_sample(inputs):
    if isinstance(inputs, torch.Tensor):
        input_sample = inputs[:3]
        
    if isinstance(inputs, DataLoader):
        input_sample = next(iter(inputs))[:3]
        
    return input_sample
    
def get_feature_maps_(model, inputs):    
    def register_hook(module):
        def hook(module, input, output):
            module_name = get_module_name(module, feature_maps)
            feature_maps[module_name] = output
                    
        if not isinstance(module, nn.Sequential): 
            if not isinstance(module, nn.ModuleList):
                hooks.append(module.register_forward_hook(hook))
                            
    feature_maps = OrderedDict()
    hooks = []
    
    model.apply(register_hook)
    with torch.no_grad():
        model(inputs)

    for hook in hooks:
        hook.remove()
        
    return(feature_maps)

def remove_duplicate_feature_maps(feature_maps, method = 'hashkey', return_matches = False, use_tqdm = False):
    matches, layer_names = [], list(feature_maps.keys())
        
    if method == 'iterative':
        
        target_iterator = tqdm(range(len(layer_names)), leave = False) if use_tqdm else range(len(layer_names))
        
        for i in target_iterator:
            for j in range(i+1,len(layer_names)):
                layer1 = feature_maps[layer_names[i]].flatten()
                layer2 = feature_maps[layer_names[j]].flatten()
                if layer1.shape == layer2.shape and torch.all(torch.eq(layer1,layer2)):
                    if layer_names[j] not in matches:
                        matches.append(layer_names[j])

        deduplicated_feature_maps = {key:value for (key,value) in feature_maps.items()
                                         if key not in matches}
        
    if method == 'hashkey':
        
        target_iterator = tqdm(layer_names, leave = False) if use_tqdm else layer_names
        layer_lengths = [len(tensor.flatten()) for tensor in feature_maps.values()]
        random_tensor = torch.rand(np.array(layer_lengths).max())
        
        tensor_dict = defaultdict(lambda:[])
        for layer_name in target_iterator:
            target_tensor = feature_maps[layer_name].cpu().flatten()
            tensor_dot = torch.dot(target_tensor, random_tensor[:len(target_tensor)])
            tensor_hash = np.array(tensor_dot).tobytes()
            tensor_dict[tensor_hash].append(layer_name)
            
        matches = [match for match in list(tensor_dict.values()) if len(match) > 1]
        layers_to_keep = [tensor_dict[tensor_hash][0] for tensor_hash in tensor_dict]
        
        deduplicated_feature_maps = {key:value for (key,value) in feature_maps.items()
                                         if key in layers_to_keep}
    
    if return_matches:
        return(deduplicated_feature_maps, matches)
    
    if not return_matches:
        return(deduplicated_feature_maps)
    
def check_for_input_axis(feature_map, input_size):
    axis_match = [dim for dim in feature_map.shape if dim == input_size]
    return True if len(axis_match) == 1 else False

def reset_input_axis(feature_map, input_size):
    input_axis = feature_map.shape.index(input_size)
    return torch.swapaxes(feature_map, 0, input_axis)

def get_feature_maps(model_name, model, inputs, layers_to_retain = None, remove_duplicates = True, tokenizer = None, preprocess = None, image_preprocess = None, frcnn = None, frcnn_cfg = None, batch_size = 8, processor = None):
    enforce_input_shape = True
    
    def register_hook(module):
        def hook(module, input, output):
            def process_output(output, module_name):
                if layers_to_retain is None or module_name in layers_to_retain:
                    if isinstance(output, torch.Tensor):
                        outputs = output.cpu().detach().type(torch.FloatTensor)
                        # outputs = output.type(torch.cuda.FloatTensor)
                        if enforce_input_shape:
                            if outputs.shape[0] == inputs['image'].shape[0]:
                                feature_maps[module_name] = outputs
                            if outputs.shape[0] != inputs['image'].shape[0]:
                                if check_for_input_axis(outputs, inputs['image'].shape[0]):
                                    outputs = reset_input_axis(outputs, inputs['image'].shape[0])
                                    feature_maps[module_name] = outputs
                                if not check_for_input_axis(outputs, inputs['image'].shape[0]):
                                    feature_maps[module_name] = None
                                    warn('Ambiguous input axis in {}. Skipping...'.format(module_name))
                        if not enforce_input_shape:
                            feature_maps[module_name] = outputs
                if layers_to_retain is not None and module_name not in layers_to_retain:
                    feature_maps[module_name] = None
                            
            module_name = get_module_name(module, feature_maps)
            
            if not any([isinstance(output, type_) for type_ in (tuple,list)]):
                process_output(output, module_name)
            
            if any([isinstance(output, type_) for type_ in (tuple,list)]):
                for output_i, output_ in enumerate(output):
                    module_name_ = '-'.join([module_name, str(output_i+1)])
                    process_output(output_, module_name_)
                    
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))
            
    feature_maps = OrderedDict()
    hooks = []
    
    model.apply(convert_relu)
    model.apply(register_hook)
    with torch.no_grad():
        images = inputs['image']
        contexts = inputs['context']
        if len(contexts) == 1:
            contexts = ' '.join(contexts)
        if model_name in ['clip', 'simclr', 'slip', 'beit', 'convnext']:
            if tokenizer:
                contexts = tokenizer(contexts)
                if len(contexts.shape) == 1:
                    contexts = contexts.unsqueeze(0)
            else:
                contexts = clip.tokenize(contexts)
            if model_name == 'clip':
                model(images.cuda(), contexts.cuda())
            elif model_name == 'slip':
                model(image = images.cuda(), text = contexts.cuda())
            else:
                model(images.cuda())
        elif model_name == 'sbert' or model_name == 'simcse':
            encoded_input = tokenizer(contexts, return_tensors = 'pt', padding = 'max_length', max_length = 73, return_attention_mask = True)
            model(input_ids = encoded_input['input_ids'].cuda(), attention_mask = encoded_input['attention_mask'].cuda())
        elif model_name == 'flava':
            encoded_input = tokenizer(text = contexts, images = list(images), return_tensors = 'pt', padding = 'max_length', max_length = 73)
            model(input_ids = encoded_input['input_ids'].cuda(), token_type_ids = encoded_input['token_type_ids'].cuda(), attention_mask = encoded_input['attention_mask'].cuda(), pixel_values = encoded_input['pixel_values'].cuda())
        elif model_name in ['albef', 'blip']:
            model.extract_features({'image': images.cuda(), 'text_input': contexts})
        else:
            raise NotImplementedError(f'{model_name} is a choice but I have not implemented it yet!')

    for hook in hooks:
        hook.remove()
        
    feature_maps = {map:features for (map,features) in feature_maps.items()
                        if features is not None}

    if remove_duplicates == True:
        feature_maps = remove_duplicate_feature_maps(feature_maps)
        
    return(feature_maps)

def get_empty_feature_maps(model_name, model, inputs = None, dataset_size=3, batch_size = 8,
        layers_to_retain = None, remove_duplicates = True, names_only=False, average_subwords = False, tokenizer = None,
        preprocess = None, image_preprocess = None, frcnn = None, frcnn_cfg = None, processor = None):

    empty_feature_maps = get_feature_maps(model_name, model, inputs, layers_to_retain, remove_duplicates, tokenizer = tokenizer, preprocess = preprocess, image_preprocess = image_preprocess, frcnn = frcnn, frcnn_cfg = frcnn_cfg, batch_size = batch_size, processor = processor)
    for map_key in empty_feature_maps:
        if 'Dropout' in map_key:
            continue
        empty_feature_maps[map_key] = torch.empty(dataset_size, *empty_feature_maps[map_key].shape[1:])
    empty_feature_maps = {key:value for key, value in empty_feature_maps.items() if 'Dropout' not in key}
    if names_only:
        return list(empty_feature_maps.keys())
    return empty_feature_maps  

def get_feature_map_names(model, inputs = None, remove_duplicates = True):
    feature_map_names = get_empty_feature_maps(model, inputs, names_only = True,
                                                remove_duplicates = remove_duplicates)
    return(feature_map_names)

def get_feature_map_count(model, inputs = None, remove_duplicates = True):
    feature_map_names = get_feature_map_names(model, inputs, remove_duplicates)
    
    return(len(feature_map_names))

def get_all_feature_maps(model_name, model, inputs, layers_to_retain=None, remove_duplicates=True, batch_size = 8,
                         include_input_space = False, flatten=True, numpy=True, use_tqdm = True, average_subwords = False, 
                         tokenizer = None, preprocess = None, image_preprocess = None, frcnn = None, frcnn_cfg = None, 
                         processor = None):

    names_only = False
    if isinstance(inputs, DataLoader):
        dataset_size, start_index = len(inputs.dataset), 0
        feature_maps = get_empty_feature_maps(model_name, model, next(iter(inputs)), 
                                              dataset_size, batch_size, layers_to_retain, remove_duplicates, names_only, average_subwords, tokenizer, preprocess, image_preprocess, frcnn, frcnn_cfg, processor = processor)
                
        for batch in tqdm(inputs, desc = 'Feature Extraction (Batch)') if use_tqdm else inputs:
            batch_feature_maps = get_feature_maps(model_name, model, batch, layers_to_retain, remove_duplicates = False, tokenizer = tokenizer, preprocess = preprocess,
                                                    image_preprocess = image_preprocess, frcnn = frcnn, frcnn_cfg = frcnn_cfg, batch_size = batch_size)
            
            imgs = batch['image']
            for map_i, map_key in enumerate(feature_maps):
                if 'Dropout' in map_key:
                    continue
                feature_maps[map_key][start_index:start_index+imgs.shape[0],...] = batch_feature_maps[map_key]
            start_index += imgs.shape[0]
                    
    if not isinstance(inputs, DataLoader):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cuda() if next(model.parameters()).is_cuda else inputs
        feature_maps = get_feature_maps(model, inputs, layers_to_retain, remove_duplicates)
        
        if include_input_space:
            feature_maps = {**{'Input': inputs.cpu()}, **feature_maps}
    
    if remove_duplicates:
        feature_maps = remove_duplicate_feature_maps(feature_maps)
     
    if flatten:
        for map_key in feature_maps:
            incoming_map = feature_maps[map_key]
            feature_maps[map_key] = incoming_map.reshape(incoming_map.shape[0], -1)
            
    if numpy:
        for map_key in feature_maps:
            feature_maps[map_key] = feature_maps[map_key].numpy().astype(np.float16)

    return feature_maps

def init_weights(module):
    #From HuggingFace implementation
    ''' Initialize the weights '''
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=6.0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def load_huggingface_model(model_name, randomized = False):
    if model_name == 'sbert':
        model_str = 'sentence-transformers/all-mpnet-base-v2'
        tokenizer_str = model_str
    elif model_name == 'flava':
        model_str = 'facebook/flava-full'
    else:
        model_str = 'princeton-nlp/sup-simcse-bert-base-uncased'
        tokenizer_str = model_str
    if model_name != 'flava':
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_str)
    else:
        tokenizer = transformers.AutoProcessor.from_pretrained(model_str)
    if not randomized:
        model = transformers.AutoModel.from_pretrained(model_str) 
    else:
        print(f'Loading randomly initialized {model_name}')
        config = transformers.AutoConfig.from_pretrained(model_str)
        #NOTE: AutoModel.from_config doesn't have a cache directory for whatever reason...
        model = transformers.AutoModel.from_config(config) 
        model.apply(model._init_weights)
    model = model.eval()
    model_args = {'model': model.cuda(), 'tokenizer': tokenizer}
    return model_args

def load_slip_model(model_name, randomized = False):
    weights_path = '/storage/vsub851/ecog-multimodal/vil_embeds/pretrained_models'
    if model_name == 'clip':
        model = CLIP_VITB16()
        model_state_dict = torch.load(os.path.join(weights_path, 'clip_base_25ep.pt'))['state_dict']
    elif model_name == 'slip':
        model = SLIP_VITB16()
        model_state_dict = torch.load(os.path.join(weights_path, 'slip_base_25ep.pt'))['state_dict']
    elif model_name == 'simclr':
        model = SIMCLR_VITB16()
        model_state_dict = torch.load(os.path.join(weights_path, 'simclr_base_25ep.pt'))['state_dict']
    for key in list(model_state_dict.keys()):
        model_state_dict[key.replace('module.', '')] = model_state_dict.pop(key)
    if not randomized:
        model.load_state_dict(model_state_dict)
    else:
        print(f'Loading randomly initialized {model_name}')
    model = model.cuda()
    model = model.eval()
    tokenizer = SimpleTokenizer()
    return {'model': model, 'tokenizer': tokenizer}

def load_timm_model(model_name, randomized = False):
    if model_name == 'beit':
        model_str = 'beit_base_patch16_224'
    else:
        model_str = 'convnext_base_in22k'
    pretrained = not randomized
    if randomized:
        print(f'Loading randomly initialized {model_name}')
    model = create_model(model_str, pretrained = pretrained)
    model = model.cuda()
    model = model.eval()
    return {'model': model}

def load_lavis_model(model_name, randomized):
    model_str = f'{model_name}_feature_extractor'
    device = torch.device('cuda')
    model, _, _ = load_model_and_preprocess(model_str, model_type = 'base', is_eval = True, device = device)
    if randomized:
        print(f'Loading randomly initialized {model_name}')
        model.apply(init_weights)
    return {'model': model}

def mean_pooling(token_embeddings, attention_mask):
    #From HuggingFace
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def postprocess_sbert(feature_maps, inputs, tokenizer):
    attention_mask = []
    for batch in inputs:
        contexts = batch['context']
        encoded_input = tokenizer(contexts, return_tensors = 'pt', padding = 'max_length', max_length = 73, return_attention_mask = True)
        attention_mask.append(encoded_input['attention_mask'])
    attention_mask = torch.concat(attention_mask, dim = 0)
    output_embeddings = mean_pooling(torch.tensor(feature_maps['LayerNorm-25']), attention_mask)
    feature_maps['LayerNorm-25'] = output_embeddings.cpu().numpy()
    return feature_maps

def run_model(model, inputs, layers_to_retain, flatten = False, batch_size = 8, randomized = False):
    '''
    Applies per-layer feature extraction on a given model string. Assumes that specific layers to retain are passed in to the function. If not, 
    this will aggregate the tensors across all layers of the model (which is memory intensive). 
    '''
    if model in ['sbert', 'simcse', 'flava']:
        model_args = load_huggingface_model(model, randomized = randomized)
    elif model in ['simclr', 'clip', 'slip']:
        model_args = load_slip_model(model, randomized = randomized)
    elif model in ['beit', 'convnext']:
        model_args = load_timm_model(model, randomized = randomized)
    elif model in ['albef', 'blip']:
        model_args = load_lavis_model(model, randomized = randomized)
    else:
        raise NotImplementedError(f'{model} is not a choice')
    
    feature_maps = get_all_feature_maps(model, inputs = inputs, layers_to_retain = layers_to_retain, flatten = flatten, batch_size = batch_size, **model_args)
    if model == 'sbert' and 'LayerNorm-25' in feature_maps.keys():
        return postprocess_sbert(feature_maps, inputs, model_args['tokenizer'])
    return feature_maps

if __name__ == '__main__':
    ## FOR TESTING, to run do
    ## CUDA_VISIBLE_DEVICES=1 python -m vil_embeds.vil_feature_extraction
    transcript_df = pd.read_csv('data-by-subject/m00185/trial000_word_stimulus_metadata.csv')

    # _, vis_processor, text_processor = load_model_and_preprocess('albef_feature_extractor', is_eval = True, model_type = 'base')
    # model_str = 'convnext_base_in22k'
    # config = resolve_data_config({}, model = model_str)
    # timm_transforms = create_transform(**config)

    vil_dataset = ViLDataset(image_paths = transcript_df.image_path, contexts = transcript_df.context, use_cv2 = False)
    vil_dataset = data.Subset(vil_dataset, [i for i in range(100)])
    vil_dataloader = data.DataLoader(vil_dataset, batch_size = 8)

    feature_maps = run_model('slip', vil_dataloader, layers_to_retain = ['Identity-51'])
    for map in feature_maps:
        print(map, feature_maps[map].shape)
        # print(feature_maps[map].get_device())