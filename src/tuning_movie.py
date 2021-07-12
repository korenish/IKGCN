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
dataset = 'movie'
aggregator = 'sum'
movie_properties = {
    'lr': 6e-3,
    'K': 5,
    'H': 2,
    'dim': 32,
    'batch_size': 65536,
    'l2_weight': 1e-7,
    'ratio': 1
}
method = 'ikgcn'
best_score = 0
best_tuned = ""
# Run experiments
print(f"Start Tuning on {dataset} dataset")
print(f"Evaluating {dataset} model with {aggregator} as aggregator:")
dims = [4, 8, 16, 32, 64, 128]
for dim in dims:
    np.random.seed(123)
    tf.random.set_seed(33)
    print('-' * 50)
    print(f"Tuned on:  dim={dim}")
    parser = argparse.ArgumentParser()
    # Parse Parameters:
    parser.add_argument('--tuning', type=bool, default=False, help='where to write results')
    parser.add_argument('--Hyperparameter', type=bool, default=True, help='where to write results')
    parser.add_argument('--method', type=str, default=f'{method}', help='which method to use')
    parser.add_argument('--dataset', type=str, default=f'{dataset}', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default=f'{aggregator}', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=movie_properties['K'],
                        help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=dim,
                        help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=movie_properties['H'],
                        help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=movie_properties['batch_size'], help='batch size')
    parser.add_argument('--l2_weight', type=float, default=movie_properties['l2_weight'],
                        help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=movie_properties['lr'], help='learning rate')
    parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

    # Run Code:
    show_loss = False
    show_time = True
    show_topk = True
    t = time()

    args = parser.parse_args()
    data = load_data(args)
    print(f"time used for loading the data: {time() - t}")
    # Create, train and evaluate model
    train(args, data, show_loss, show_topk)

    if show_time:
        print('time used for this experiment: %d s' % (time() - t))
    # Reset all tensorflow models and variables
    tf.keras.backend.clear_session()

# Done with experiments on a single dataset
print(f"Done experimenting on {dataset} dataset")
print('-' * 50)
