#!/usr/bin/python

# Run this script to remove the data that are only useful for training
# from your model files in order to make them more compact.

import os
import numpy as np

if __name__=='__main__':
    files = os.listdir('.')
    model_files = [f for f in files if f.endswith('.npy')]

    for model_file in model_files:
        model = np.load(model_file).item()
        trimmed_model = {var_name: model[var_name] for var_name in model.keys()
                         if 'optimizer' not in var_name}
        os.rename(model_file, model_file[:-4]+'_old.npy')
        np.save(model_file, trimmed_model)
