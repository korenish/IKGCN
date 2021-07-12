from data_loader import load_data
import tensorflow as tf
from train import train
from time import time
import numpy as np
import argparse
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.disable_eager_execution()

# Experiments Properties:
datasets = ['music']
aggregators = ['concat']
music_properties = {
    'neighbor_sample_size': 8,
    'dim': 16,
    'n_iter': 1,
    'batch_size': 128,
    'l2_weight': 1e-4,
    'lr': 5e-4,
    'ratio': 1
}
movie_properties = {
    'neighbor_sample_size': 4,
    'dim': 32,
    'n_iter': 2,
    'batch_size': 65536,
    'l2_weight': 1e-7,
    'lr': 2e-2,
    'ratio': 1
}
book_properties = {
    'neighbor_sample_size': 8,
    'dim': 64,
    'n_iter': 3,
    'batch_size': 256,
    'l2_weight': 2e-5,
    'lr': 2e-4,
    'ratio': 1
}
model_properties = {
    'music': music_properties,
    'movie': movie_properties,
    'book': book_properties
}
method = 'ikgcn'

# Run experiments
for dataset in datasets:
    first = True
    print('-'*50)
    print(f"Start Experiments on {dataset} dataset")
    curr_props = model_properties[dataset]
    for agg in aggregators:
        print('-' * 25)
        print(f"Evaluating {dataset} model with {agg} as aggregator:")
        parser = argparse.ArgumentParser()
        # Parse Parameters:
        parser.add_argument('--method', type=str, default=f'{method}', help='which method to use')
        parser.add_argument('--dataset', type=str, default=f'{dataset}', help='which dataset to use')
        parser.add_argument('--aggregator', type=str, default=f'{agg}', help='which aggregator to use')
        parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
        parser.add_argument('--neighbor_sample_size', type=int, default=curr_props['neighbor_sample_size'],
                            help='the number of neighbors to be sampled')
        parser.add_argument('--dim', type=int, default=curr_props['dim'],
                            help='dimension of user and entity embeddings')
        parser.add_argument('--n_iter', type=int, default=1,
                            help='number of iterations when computing entity representation')
        parser.add_argument('--batch_size', type=int, default=curr_props['batch_size'], help='batch size')
        parser.add_argument('--l2_weight', type=float, default=curr_props['l2_weight'],
                            help='weight of l2 regularization')
        parser.add_argument('--lr', type=float, default=curr_props['lr'], help='learning rate')
        parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

        # Run Code:
        show_loss = False
        show_time = True
        show_topk = True

        t = time()

        args = parser.parse_args()
        if first:
            # Load dataset
            data = load_data(args)
            print(f"time used for loading the data: {time() - t}")
        # Create, train and evaluate model
        train(args, data, show_loss, show_topk)

        first = False

        if show_time:
            print('time used for this experiment: %d s' % (time() - t))
        print('-' * 25)
        # Reset all tensorflow models and variables
        tf.keras.backend.clear_session()

    # Done with experiments on a single dataset
    print(f"Done experimenting on {dataset} dataset")
    print('-' * 50)
