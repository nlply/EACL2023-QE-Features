import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn

import numpy as np
import math
from nltk import word_tokenize
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset.csv',
                        help='Path to dataset file')
    parser.add_argument('--word_embedding', type=str, default='glove.6B.50d.txt',
                        help='Path to word embedding file.')
    parser.add_argument('--features', type=str, default='dataset_features.csv',
                        help='Path to save the features file')
    args = parser.parse_args()
    return args


def getWord2id(df):
    setups = df['set-up'].tolist()
    punchlines = df['punchline'].tolist()
    word2id = dict()
    for setup, punchline in zip(setups, punchlines):
        setup = word_tokenize(setup)
        for s_word in setup:
            s_word = s_word.lower().strip()
            if word2id.get(s_word) == None:
                word2id[s_word] = len(word2id)

        punchline = word_tokenize(punchline)
        for p_word in punchline:
            p_word = p_word.lower().strip()
            if word2id.get(p_word) == None:
                word2id[p_word] = len(word2id)

    return word2id


def get_numpy_word_embed(args, word2ix):
    row = 0
    words_embed = {}
    with open(args.word_embedding, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            norm = 0.0
            for i in embed:
                norm = norm + i * i
            embed = [num / np.sqrt(norm) for num in embed]
            words_embed[word] = embed
            if row > 20000:
                break
            row += 1
    ix2word = {ix: w for w, ix in word2ix.items()}
    id2emb = {}
    for ix in range(len(word2ix)):
        if ix2word[ix] in words_embed:
            id2emb[ix] = words_embed[ix2word[ix]]
        else:
            id2emb[ix] = [0.0] * 50
    data = [id2emb[ix] for ix in range(len(word2ix))]

    return data


def getVonNeumannEntropy(word_density_matrix):
    e_vs = torch.linalg.eigvals(word_density_matrix)
    s_rho = torch.tensor(0.0).to('cuda')
    for e_v in e_vs:
        real_value = torch.view_as_real(e_v)[0]
        if real_value.item() != 0:
            s_rho_item = real_value * torch.log(real_value)
            if torch.isnan(s_rho_item).item() is False:
                s_rho += -s_rho_item
    if torch.isnan(s_rho).item():
        s_rho = torch.tensor(-1.0)
    return s_rho.item()


def getConditionalEntropy(rho_matrix, sigma_matrix):
    sigma_rho_matrix = sigma_matrix * rho_matrix
    sr_vs = torch.linalg.eigvals(sigma_rho_matrix)
    s_rho_sigma = torch.tensor(0.0).to('cuda')
    for sr_v in sr_vs:
        sr_v_real = torch.view_as_real(sr_v)[0]
        if sr_v_real.item() != 0:
            s_rho_sigma_item = sr_v_real * torch.log(sr_v_real)
            if torch.isnan(s_rho_sigma_item).item() is False:
                s_rho_sigma += s_rho_sigma_item
    e_vs = torch.linalg.eigvals(rho_matrix)
    s_rho = torch.tensor(0.0).to('cuda')
    for e_v in e_vs:
        e_v_real = torch.view_as_real(e_v)[0]
        if e_v_real.item() != 0:
            s_rho_item = e_v_real * torch.log(e_v_real)
            if torch.isnan(s_rho_item).item() is False:
                s_rho += s_rho_item
    c_v = s_rho_sigma - s_rho
    if torch.isnan(c_v).item():
        c_v = torch.tensor(-1.0)
    return c_v.item()


def isfloat(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def get_features(df, embedding, word2id):
    setups = df['set-up'].tolist()
    punchlines = df['punchline'].tolist()
    lb = df['label'].tolist()
    qe_uncertainty_values = []
    qe_incongruity_values = []
    for setup, punchline in tqdm(zip(setups, punchlines), total=len(setups)):
        setup = word_tokenize(setup)
        setup_ids = [word2id.get(w.lower().strip()) for w in setup]
        setup_ids_tensor = torch.tensor(setup_ids).to('cuda')
        setup_embedding = embedding(setup_ids_tensor)

        setup_matrix = torch.matmul(setup_embedding.unsqueeze(-1), setup_embedding.unsqueeze(-2))
        setup_matrix = torch.mean(setup_matrix, dim=0)

        punchline = word_tokenize(punchline)
        punchline_ids = [word2id.get(w.lower().strip()) for w in punchline]
        punchline_ids_tensor = torch.tensor(punchline_ids).to('cuda')
        punchline_embedding = embedding(punchline_ids_tensor)

        punchline_matrix = torch.matmul(punchline_embedding.unsqueeze(-1), punchline_embedding.unsqueeze(-2))
        punchline_matrix = torch.mean(punchline_matrix, dim=0)

        uncertainty = getVonNeumannEntropy(setup_matrix)

        if math.isnan(uncertainty) or not isfloat(uncertainty):
            uncertainty = torch.tensor(-1).item()

        qe_uncertainty_values.append(uncertainty)

        incongruity = getConditionalEntropy(setup_matrix, punchline_matrix)

        if math.isnan(incongruity) or not isfloat(incongruity):
            incongruity = torch.tensor(-1).item()

        qe_incongruity_values.append(incongruity)

    return qe_uncertainty_values, qe_incongruity_values, lb



if __name__ == '__main__':
    args = parse_args()
    df = pd.read_csv(args.dataset)
    word2id = getWord2id(df)
    numpy_embed = get_numpy_word_embed(args, word2id)
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(numpy_embed)).to('cuda')
    QE_Uncertainty, QE_Incongruity, labels = get_features(df, embedding, word2id)
    pd.DataFrame({
        'QE_Uncertainty': QE_Uncertainty,
        'QE_Incongruity': QE_Incongruity,
        'label': labels,
    }).to_csv(args.features, index=False)
