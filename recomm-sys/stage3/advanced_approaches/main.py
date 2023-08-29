import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from models.features import Number, Category, Sequence, Features
from models.transformers.column import (StandardScaler, CategoryEncoder, SequenceEncoder)
from models.pytorch.data import Dataset
from models.pytorch import WideDeep, DeepFM, DNN, DIN, AttentionGroup
from functions import fit, predict, create_dataloader_fn



train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.movieId = train_df.movieId.astype(str)
test_df.movieId = test_df.movieId.astype(str)

train_df.histHighRatedMovieIds = train_df.histHighRatedMovieIds.astype(str)
test_df.histHighRatedMovieIds = test_df.histHighRatedMovieIds.astype(str)

train_df.negHistMovieIds = train_df.negHistMovieIds.astype(str)
test_df.negHistMovieIds = test_df.negHistMovieIds.astype(str)

#print(train_df.dtypes)
#print(train_df.head())
#print(test_df.dtypes)
#print(test_df.head())
print(train_df.shape, test_df.shape)

number_features = [
    Number('age', StandardScaler())
]

category_features = [
    Category('gender', CategoryEncoder(min_cnt=1)),
    Category('movieId', CategoryEncoder(min_cnt=1)),
    Category('occupation', CategoryEncoder(min_cnt=1)),
    Category('zipCode', CategoryEncoder(min_cnt=1))
]

sequence_features = [
    Sequence('genres', SequenceEncoder(sep='|', min_cnt=1)),
    Sequence('histHighRatedMovieIds', SequenceEncoder(sep='|', min_cnt=1)),
    Sequence('negHistMovieIds', SequenceEncoder(sep='|', min_cnt=1))
]

features, train_loader, valid_loader = create_dataloader_fn(
    number_features, category_features, sequence_features, 64, train_df, 'label', test_df, 4)

def evaluation(df, dataloader):
    preds = predict(model, dataloader)
    return roc_auc_score(df['label'], preds.ravel())


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

din_attention_groups = [
    AttentionGroup(
        name='group1',
        pairs=[{'ad': 'movieId', 'pos_hist': 'histHighRatedMovieIds'}],
        hidden_layers=[16, 8], att_dropout=0.1)]

gru_attention_groups = [
    AttentionGroup(
        name='group1',
        pairs=[{'ad': 'movieId', 'pos_hist': 'histHighRatedMovieIds'}],
        hidden_layers=[16, 8], att_dropout=0.1, gru_type='GRU')]

aigru_attention_groups = [
    AttentionGroup(
        name='group1',
        pairs=[{'ad': 'movieId', 'pos_hist': 'histHighRatedMovieIds'}],
        hidden_layers=[16, 8], att_dropout=0.1, gru_type='AIGRU')]

agru_attention_groups = [
    AttentionGroup(
        name='group1',
        pairs=[{'ad': 'movieId', 'pos_hist': 'histHighRatedMovieIds'}],
        hidden_layers=[16, 8], att_dropout=0.1, gru_type='AGRU')]

augru_attention_groups = [
    AttentionGroup(
        name='group1',
        pairs=[{'ad': 'movieId', 'pos_hist': 'histHighRatedMovieIds'}],
        hidden_layers=[16, 8], att_dropout=0.1, gru_type='AUGRU')]

augru_attention_groups_with_neg = [
    AttentionGroup(
        name='group1',
        pairs=[{'ad': 'movieId', 'pos_hist': 'histHighRatedMovieIds', 'neg_hist': 'negHistMovieIds'}],
        hidden_layers=[16, 8], att_dropout=0.1, gru_type='AUGRU')]

models = [
    DNN(features, 2, 16, (32, 16), final_activation='sigmoid', dropout=0.3),
    WideDeep(
        features,
        wide_features=['genres', 'movieId'],
        deep_features=['age', 'gender', 'movieId', 'occupation',
         'zipCode', 'histHighRatedMovieIds'],
        cross_features=[('movieId', 'histHighRatedMovieIds')],
        num_classes=2, embedding_size=16, hidden_layers=(32, 16),
        final_activation='sigmoid', dropout=0.3),
    DeepFM(features, 2, 16, (32, 16), final_activation='sigmoid', dropout=0.3),
    DIN(features, din_attention_groups, 2, 16, (32, 16), final_activation='sigmoid', dropout=0.3),
]

scores = []
for model in models:
    print(model)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    fit(10, model, loss_func, optimizer,
        train_loader, valid_loader, notebook=True, auxiliary_loss_rate=0.1)
    scores.append(evaluation(test_df, valid_loader))

print(scores)














