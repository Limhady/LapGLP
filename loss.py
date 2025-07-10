from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import sys

def get_weights(n_user, m_item, items_D, users_D, interaction_dict, omegas, device):
	items_inv_sqrtD = 1 / np.sqrt(items_D)
	users_inv_sqrtD = 1 / np.sqrt(users_D)

	print('Doing Weight-I.')
	normalized_train_mat = sp.dok_matrix((n_user, m_item), dtype = np.float32)
	for u, neighborlist in interaction_dict.items():
		normalized_train_mat[u, neighborlist] = 1.0 * users_inv_sqrtD[0, u] * items_inv_sqrtD[0, neighborlist]
	weightI = torch.from_numpy(normalized_train_mat.T.dot(normalized_train_mat).toarray()).to(device) # P_ij = sum_(A_ui & A_uj)(1 / d_u) * 1 / sqrt(d_i * d_j)
	weightI[np.diag_indices(m_item)] = 0.0

	print('Doing Weight-U.')
	weightU = omegas[5] * users_inv_sqrtD.reshape(-1, 1).dot(items_inv_sqrtD.reshape(1, -1)) #np.ones((n_user, m_item), dtype = np.float32)
	for uid, neighborlist in interaction_dict.items():
		weightU[uid, neighborlist] = omegas[2] * items_inv_sqrtD[0, neighborlist] * users_inv_sqrtD[0, uid]
	weightU = torch.from_numpy(weightU).to(device)
	print('Doing item_inv_sqrtD.')
	items_inv_sqrtD = torch.from_numpy(items_inv_sqrtD).reshape(-1).to(device)

	return weightI, weightU, items_inv_sqrtD

def rr_items_D(items_D, n_user, m_item, p_):#Item degree calculation with randomized response
	#items_D = np.sum(train_mat, axis = 0).reshape(-1)
	#if p == 0 or p == 1:
	#	return items_D
	for i in range(m_item):
		di = items_D[0, i]
		noise = np.random.binomial(n_user - di, 1 - p_) + np.random.binomial(di, p_)
		items_D[0, i] = (p_ - 1) / (2 * p_ - 1) * n_user + noise / (2 * p_ - 1)
	return items_D

def cnr_items_gradient(items_gradient, n_user, m_item, delta_, lambda_, device):
	return torch.clamp(items_gradient, -delta_, delta_) + torch.from_numpy(np.random.laplace(0, lambda_, (n_user, items_gradient.shape[0], items_gradient.shape[1]))).sum(dim = 0).to(device)

def cal_smoothing_and_oversmoothing_loss(m_item, n_user, item_embeddings, user_embeddings, weightI, weightQ, pos_samples, num_negative,omegas, device):
	ii_loss = cal_ii_loss(m_item, item_embeddings, weightI, num_negative, omegas, device)
	ui_loss = cal_ui_loss(n_user, item_embeddings, user_embeddings, weightQ, pos_samples, omegas, device)
	return ii_loss + ui_loss

def cal_ii_loss(m_item, item_embeddings, weightI, num_negative, omegas, device):
	ii_dots = item_embeddings.mm(item_embeddings.T)
	lapI_loss = omegas[1] * weightI * ii_dots.sigmoid().log().to(device)
	lapI_loss = - lapI_loss.sum()

	if num_negative > 0: # randomly choose part of the items to reduce the computational cost
		indices_i = np.arange(m_item).repeat(num_negative)
		neg_samples = np.random.choice(m_item, m_item * num_negative, replace = True)
		lapI_neg_loss = omegas[4] * weightI[indices_i, neg_samples] * (-ii_dots[indices_i, neg_samples]).sigmoid().log().to(device)
		lapI_neg_loss = lapI_neg_loss.sum()
		return lapI_loss - lapI_neg_loss

	return lapI_loss

def cal_ui_loss(n_user, item_hidden, user_hidden, weightQ, pos_samples, omegas, device):
	ui_dots = (-user_hidden).mm(item_hidden.T).to(device)
	for u in pos_samples.keys():
		ui_dots[u, pos_samples[u]] = -ui_dots[u, pos_samples[u]]
	ui_loss = -(weightQ * ui_dots.sigmoid().log()).sum()
	return ui_loss

def cal_rating_loss(r_score, item_embeddings, user_embeddings, score_mats, pos_samples, score_interaction_dict, omegas, device):
	rating_loss = 0.0
	score_indices = list(score_interaction_dict.keys())
	for si in range(r_score):
		score = score_indices[si]
		usi_dots = (-user_embeddings).mm(score_mats[si]).mm(item_embeddings.T)
		for u, neighbors in score_interaction_dict[score].items():
			usi_dots[u, neighbors] = -usi_dots[u, neighbors]
		rating_loss -= (usi_dots.sigmoid().log()).sum()

	return omegas[0] * rating_loss

def cal_norm_loss(item_embeddings, user_embeddings, omegas, device):
	norm_loss = (item_embeddings * item_embeddings).sum() + (user_embeddings * user_embeddings).sum()
	return omegas[3] * norm_loss

def cal_norm_loss_with_fusion(item_embeddings, user_embeddings, user_fusion, omegas, device):
	norm_loss = (item_embeddings * item_embeddings).sum() + (user_embeddings * user_embeddings).sum() + (user_fusion *user_fusion).sum()
	return omegas[3] * norm_loss