import random
from heapq import nlargest
import numpy as np
import os


data_path = '/home/korenish/Recsys_project/data'


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation


def load_rating(args):
    print('reading rating file ...')
    DATASET = args.dataset

    # reading rating file
    rating_file = f'{data_path}/{DATASET}/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')
    DATASET = args.dataset
    # reading kg file
    kg_file = f'{data_path}/{DATASET}/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    if args.method == 'kgcn':
        adj_entity, adj_relation = construct_adj(args, kg, n_entity)
    elif args.method == 'ikgcn':
        adj_entity, adj_relation = construct_informative_adj(args, kg, n_entity)
    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):
    """
    Construct normal KG via dictionary.
    """
    print('constructing informative knowledge graph ...')
    # Build undirected graph
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

    return kg


def construct_adj(args, kg, entity_num):
    """
    KGCN random sampler for neighbors selection approach.
    """
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation


def calculate_neighbors_scores(kg, entity, neig_items_scores, n_iter):
    """
    This is a recursive function that scores neighbors by the level of information they entail.
    level of information - number of relations, regarding their deepness.
    """
    neighbors = kg[entity]
    if n_iter == 1:
        for neig in neighbors:
            neig_item = neig[0]
            neig_items_scores[neig_item] += 1
    else:
        n_iter -= 1
        for neig in neighbors:
            neig_item = neig[0]
            neig_dict = {neig_neig[0]: 0 for neig_neig in kg[neig_item]}
            calculate_neighbors_scores(kg, neig_item, neig_dict, n_iter)
            for key in neig_dict.keys():
                neig_items_scores[neig_item] += neig_dict[key]/2


def construct_informative_adj(args, kg, entity_num):
    """
    Our informative sampler for neighbors selection approach.
    """
    print('constructing informative adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        dict_neighbor_relations = {}
        dict_neighbor_index = {}
        # Collect relations for each neighbor
        for neig in neighbors:
            if neig[0] in dict_neighbor_relations:
                dict_neighbor_relations[neig[0]].append(neig[1])
            else:
                dict_neighbor_index[neig[0]] = 0
                dict_neighbor_relations[neig[0]] = [neig[1]]
        # Making sure that the relations are shuffled
        for key in dict_neighbor_relations:
            if len(dict_neighbor_relations[key]) > 1:
                random.Random(1).shuffle(dict_neighbor_relations[key])

        # Creating dictionary of neighbors importance scores
        neig_items_scores = {neig[0]: 0 for neig in kg[entity]}
        calculate_neighbors_scores(kg, entity, neig_items_scores, args.n_iter)
        # Change scores to probabilities
        score_sum = sum(neig_items_scores.values())
        for key in neig_items_scores.keys():
            neig_items_scores[key] = neig_items_scores[key]/score_sum

        # Collect neighbors by importance sampler
        if len(neig_items_scores.keys()) >= args.neighbor_sample_size:
            # If there are more neighbors than the sampler size - pick top k
            k = args.neighbor_sample_size
            sampled_neighbors = nlargest(k, neig_items_scores, key=neig_items_scores.get)
            # Collect one relation for each, picks relation uniformly
            sampled_relations = []
            for neig in sampled_neighbors:
                neig_relations = dict_neighbor_relations[neig]
                relation = neig_relations[0]
                sampled_relations.append(relation)
        else:
            # If there are less neighbors than the sampler size - Pick with replacement k neighbors
            sampled_neighbors = np.random.choice(list(neig_items_scores.keys()), size=args.neighbor_sample_size,
                                                 replace=True, p=list(neig_items_scores.values()))
            # Collect relations
            sampled_relations = []
            for neig in sampled_neighbors:
                neig_relations = dict_neighbor_relations[neig]
                # Pick different relation next time if exists.
                curr_ind = dict_neighbor_index[neig]
                if curr_ind+1 < len(neig_relations):
                    dict_neighbor_index[neig] += 1
                else:
                    dict_neighbor_index[neig] = 0
                relation = neig_relations[curr_ind]
                sampled_relations.append(relation)

        # Append entity relations
        adj_entity[entity] = np.array(sampled_neighbors)
        adj_relation[entity] = np.array(sampled_relations)

    return adj_entity, adj_relation
