import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
import itertools
import logging
import sys
from time import strftime, localtime
from nltk import word_tokenize
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--useglove', type=bool,
                        help='Contact glove or not.')
    parser.add_argument('--word_embedding', type=str, default='glove.6B.50d.txt',
                        help='Path to word embedding file.')
    parser.add_argument('--dataset', type=str, default='dataset.csv',
                        help='Path to dataset file')
    parser.add_argument('--features', type=str, default='dataset_features.csv',
                        help='Path to features file')
    args = parser.parse_args()
    return args

def tokenize(q):
    return word_tokenize(q.lower())

def train_word2vec_svm_cv(dataset, k, word2vec_mat, features):
    p_list = []
    r_list = []
    f_list = []
    acc_list = []
    N = dataset.shape[0]
    index_perm = np.random.permutation(N).tolist()
    index_start = list(range(0, N, N // k))[:k] + [N]
    for i in range(k):
        s = index_start[i]
        t = index_start[i + 1]
        index_test = index_perm[s:t]
        index_train = index_perm[:s] + index_perm[t:]
        train_dataset = dataset.iloc[index_train]
        test_dataset = dataset.iloc[index_test]
        if features == []:
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for i in index_train:
                X_train.append(list(word2vec_mat[i, :].tolist()))
            y_train = list(train_dataset['label'])
            for i in index_test:
                X_test.append(list(word2vec_mat[i, :].tolist()))
            y_test = list(test_dataset['label'])

        else:
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for i, j in zip(index_train, range(len(train_dataset))):
                append_feature_list = []
                for feat in features:
                    append_feature_list.append(train_dataset.iloc[j][feat])
                X_train.append(list(word2vec_mat[i, :].tolist()) + append_feature_list)
            y_train = list(train_dataset['label'])
            for i, j in zip(index_test, range(len(test_dataset))):
                append_feature_list = []
                for feat in features:
                    append_feature_list.append(test_dataset.iloc[j][feat])
                X_test.append(list(word2vec_mat[i, :].tolist()) + append_feature_list)
            y_test = list(test_dataset['label'])
        clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = np.mean(y_test == y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        acc_list.append(acc)

    return np.mean(p_list), np.mean(r_list), np.mean(f_list), np.mean(acc_list)


def train_svm_cv(dataset, k, feature):
    p_list = []
    r_list = []
    f_list = []
    acc_list = []
    N = dataset.shape[0]
    index_perm = np.random.permutation(N).tolist()
    index_start = list(range(0, N, N // k))[:k] + [N]

    for i in range(k):
        s = index_start[i]
        t = index_start[i + 1]
        index_test = index_perm[s:t]
        index_train = index_perm[:s] + index_perm[t:]
        df_train = dataset.iloc[index_train]
        df_test = dataset.iloc[index_test]

        if len(feature) == 1:
            X_train = np.expand_dims(df_train[feature[0]].values, 1)
            y_train = df_train['label'].values
            X_test = np.expand_dims(df_test[feature[0]].values, 1)
            y_test = df_test['label'].values
        else:
            X_train = df_train[feature].values
            y_train = df_train['label'].values
            X_test = df_test[feature].values
            y_test = df_test['label'].values

        clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = np.mean(y_test == y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        acc_list.append(acc)

    return np.mean(p_list), np.mean(r_list), np.mean(f_list), np.mean(acc_list)

def getWord2id(df):
    setups = df['set-up'].tolist()
    punchlines = df['punchline'].tolist()
    word2id = dict()
    for setup, punchline in zip(setups, punchlines):
        setup = tokenize(setup)
        for s_word in setup:
            s_word = s_word.lower().strip()
            if word2id.get(s_word) == None:
                word2id[s_word] = len(word2id)

        punchline = tokenize(punchline)
        for p_word in punchline:
            p_word = p_word.lower().strip()
            if word2id.get(p_word) == None:
                word2id[p_word] = len(word2id)

    return word2id


def get_numpy_word_embed(word2ix):
    row = 0
    whole = 'glove.6B.50d.txt'
    words_embed = {}
    with open(whole, mode='r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            row += 1
    ix2word = {ix: w for w, ix in word2ix.items()}
    id2emb = {}
    for ix in range(len(word2ix)):
        if ix2word[ix] in words_embed:
            id2emb[ix] = words_embed[ix2word[ix]]
        else:
            id2emb[ix] = [1.0] * 50

    data = [id2emb[ix] for ix in range(len(word2ix))]
    return data


def run(args):
    df = pd.read_csv(args.features)
    feature_list = ['QE_Uncertainty', 'QE_Incongruity']

    log_file = 'results-{}-{}.log'.format(args.useglove, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))
    if args.useglove:
        dff = pd.read_csv(args.dataset)
        word2id = getWord2id(dff)
        numpy_embed = get_numpy_word_embed(word2id)
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(numpy_embed)).to('cuda')

        setups = dff['set-up'].tolist()
        punchlines = dff['punchline'].tolist()
        setups_final_embed = []
        punchlines_final_embed = []
        for setup, punchline in zip(setups, punchlines):
            setup = tokenize(setup)
            setup_ids = [word2id.get(w) for w in setup]
            setup_ids_tensor = torch.tensor(setup_ids).to('cuda')
            setup_embedding = embedding(setup_ids_tensor)
            setup_norm = torch.sum(setup_embedding, dim=0) / setup_embedding.shape[0]
            setups_final_embed.append(setup_norm)

            punchline = tokenize(punchline)
            punchline_ids = [word2id.get(w) for w in punchline]
            punchline_ids_tensor = torch.tensor(punchline_ids).to('cuda')
            punchline_embedding = embedding(punchline_ids_tensor)
            punchline_norm = torch.sum(punchline_embedding, dim=0) / punchline_embedding.shape[0]
            punchlines_final_embed.append(punchline_norm)

        setups_embed = torch.stack(setups_final_embed)
        punchlines_embed = torch.stack(punchlines_final_embed)
        word2vec_mat = torch.cat((setups_embed, punchlines_embed), dim=1)

        p_list, r_list, f_list, acc_list = train_word2vec_svm_cv(df, 10, word2vec_mat, [])
        logger.info('> feature is : {0}'.format('glove'))
        logger.info('> p : {:.4f}'.format(p_list))
        logger.info('> r : {:.4f}'.format(r_list))
        logger.info('> f : {:.4f}'.format(f_list))
        logger.info('> acc : {:.4f}'.format(acc_list))

        for i in np.arange(len(feature_list)):
            feature_combin_list = list(itertools.combinations(feature_list, i + 1))
            for feature in feature_combin_list:
                feature = list(feature)
                p_list, r_list, f_list, acc_list = train_word2vec_svm_cv(df, 10, word2vec_mat, feature)
                logger.info('> feature is : {0}'.format(feature))
                logger.info('> p : {:.4f}'.format(p_list))
                logger.info('> r : {:.4f}'.format(r_list))
                logger.info('> f : {:.4f}'.format(f_list))
                logger.info('> acc : {:.4f}'.format(acc_list))
    else:
        for i in np.arange(len(feature_list)):
            feature_combin_list = list(itertools.combinations(feature_list, i + 1))
            for feature in feature_combin_list:
                feature = list(feature)
                p_list, r_list, f_list, acc_list = train_svm_cv(df, 10, feature)
                logger.info('> feature is : {0}'.format(feature))
                logger.info('> p : {:.4f}'.format(p_list))
                logger.info('> r : {:.4f}'.format(r_list))
                logger.info('> f : {:.4f}'.format(f_list))
                logger.info('> acc : {:.4f}'.format(acc_list))


if __name__ == '__main__':
    args = parse_args()
    run(args)
