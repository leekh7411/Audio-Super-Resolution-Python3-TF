
# coding: utf-8

# #### referenced by - https://github.com/kuleshov/audio-super-res

# # Training ASR model

# In[1]:


import os
os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
import numpy as np
import matplotlib
from asr_model import ASRNet, default_opt
from io_utils import upsample_wav
from io_utils import load_h5
import tensorflow as tf
#matplotlib.use('Agg')


# In[2]:


args = {
    'train'      : 'train.h5',
    'val'        : 'valid.h5',
    'alg'        : 'adam',
    'epochs'     : 10,
    'logname'    : 'default_log_name',
    'layers'     : 4,
    'lr'         : 0.0005,
    'batch_size' : 100
}
print(tf.__version__)


# In[3]:


# get data
X_train, Y_train = load_h5(args['train'])
X_val, Y_val = load_h5(args['val'])


# In[4]:


# determine super-resolution level
n_dim_y, n_chan_y = Y_train[0].shape
n_dim_x, n_chan_x = X_train[0].shape
print('number of dimension Y:',n_dim_y)
print('number of channel Y:',n_chan_y)
print('number of dimension X:',n_dim_x)
print('number of channel X:',n_chan_x)
r = int(Y_train[0].shape[0] / X_train[0].shape[0])
print('r:',r)
n_chan = n_chan_y
n_dim = n_dim_y
assert n_chan == 1 # if not number of channel is not 0 -> Error assert!


# In[5]:


# create model
def get_model(args, n_dim, r, from_ckpt=False, train=True):
    """Create a model based on arguments"""
    
    if train:
        opt_params = {
            'alg' : args['alg'], 
            'lr' : args['lr'], 
            'b1' : 0.9, 
            'b2' : 0.999,
            'batch_size': args['batch_size'], 
            'layers': args['layers']}
    else: 
        opt_params = default_opt

    # create model & init
    model = ASRNet(
        from_ckpt=from_ckpt, 
        n_dim=n_dim, 
        r=r,
        opt_params=opt_params, 
        log_prefix=args['logname'])
    
    return model

model = get_model(args, n_dim, r, from_ckpt=False, train=True)


# In[ ]:


# train model
model.fit(X_train, Y_train, X_val, Y_val, n_epoch=args['epochs'])

