# CORE PACKAGES
import sys, os
import nbimporter
import matplotlib.pyplot as plt
from multiprocessing import Pool 
import nilearn.plotting
import nibabel
import pandas as pd
import numpy as np
import h5py
import pickle


# SETTING DIRECTORY VARIABLES
SRC_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(PROJ_DIR, 'data')
MODELS_DIR = os.path.join(PROJ_DIR, 'models')
STIM_DIR = os.path.join(DATA_DIR, 'stimulus')
DATASET_X_DIR = os.path.join(DATA_DIR, 'dataset/', 'X/')
DATASET_Y_DIR = os.path.join(DATA_DIR, 'dataset/', 'Y/')
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, 'trained/')
UNTRAINED_MODELS_DIR = os.path.join(MODELS_DIR, 'fresh/')
# MODEL_INPUT_DIR = os.path.join(STIM_DIR, 'Model_Inputs')

IMAGES_DIR = {
    'imagenet': os.path.join(STIM_DIR, 'img', 'ImageNet'), 
    'coco': os.path.join(STIM_DIR, 'img', 'COCO'), 
    'scenes': os.path.join(STIM_DIR, 'img', 'Scene')}

# SCANNING PARAMETERS
TP_PER_STIM = 5
STIM_PER_RUN = 37
TR = 2
SESSIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15]
TP_RANGE = (3, 188)

# NILEARN PLOTTING FUNCTIONS
PLOTS = {'anat': nilearn.plotting.plot_anat, 
        'epi': nilearn.plotting.plot_epi,
        'glass_brain': nilearn.plotting.plot_glass_brain,
        'stat_map': nilearn.plotting.plot_stat_map,
        'roi': nilearn.plotting.plot_roi,
        'connectome': nilearn.plotting.plot_connectome,
        'prob_atlas': nilearn.plotting.plot_prob_atlas,
        'matrix': nilearn.plotting.plot_matrix,
        'html': nilearn.plotting.view_img}

# FOR EASY PLOTTING 
def make_img(mask, affine):
    return nibabel.Nifti1Image(mask, affine)

def extract_run(f):
    run_index_begin = f.find('run-')
    run_index_end = f.find('_', run_index_begin+1)
    run_substring = f[run_index_begin:run_index_end]
    run = int(run_substring.replace('run-', ''))
    return run

def get_subject_dir(subject):
    return 'sub-CSI{}/'.format(subject)

def get_session_dir(session):
    return 'ses-0{}/'.format(session) if session < 10 else 'ses-{}/'.format(session)

def save_pickle(d, name, subdir=''):
    pickle.dump(d, open(os.path.join(DUMP_DIR, subdir, '{}.p'.format(name)), 'w+'))    
    
def load_pickle(name, subdir=''):
     return pickle.load(open(os.path.join(DUMP_DIR, subdir, name)))

def load_file(path, delim='\t'):
    nilearn_files = ('.gii', '.nii.gz')
    pandas_files = ('.tsv', '.csv')
    h5_files = ('.h5')
    if path.endswith(nilearn_files):
        return (nibabel.load(path), 'nibabel')
    elif path.endswith(pandas_files):
        return (pd.read_csv(path, delimiter=delim), 'pandas')
    elif path.endswith(h5_files):
        return  (h5py.File(path, 'r'), 'h5')
    else:
        raise Exception('Unknown file type encountered: {}'.format(path))
        
def numpy_save(arr, filename, subdir=''):
    path = os.path.join(DUMP_DIR, subdir, filename)
    np.save(path, arr, allow_pickle=False)
    
def numpy_load(filename, subdir = ''):
    arr = np.load(os.path.join(DUMP_DIR, subdir, filename))
    return arr

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def get_activations(model, model_inputs, K, layer_name=None, ):
    '''
    Originally implement by Philippe Remy @ https://github.com/philipperemy. Modified to our needs.
    Used to extract activations from different layers of the VGG16 net.
    '''
    
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name in layer_name or layer_name is None]  # all layer outputs

    # we remove the placeholders (Inputs node in Keras). Not the most elegant though..
    outputs = [output for output in outputs if 'input_' not in output.name]
    layer_names = [layer.name for layer in model.layers if
                  layer.name in layer_name or layer_name is None]
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    activations = [func(list_inputs)[0] for func in funcs]
#     layer_names = [output.name for output in outputs]

    result = dict(zip(layer_names, activations))
    return result