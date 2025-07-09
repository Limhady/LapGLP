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


def load_social(user_file, n_user, nf_user_feat):
	rows, cols, values = [], [], []
	with open(user_file) as f:
		for l in f.readlines():
			if len(l) > 0:
				l = l.strip('\n').split(' ')
				rows.append(int(l[0]))
				cols.append(int(l[1]))
				values.append(float(l[2]))
	#user_feat_mat = sp.dok_matrix((n_user, nf_user_feat), dtype=np.float32)
	#user_feat_mat[rows, cols] = values
	#degrees = np.sum(user_feat_mat, axis = 1).getA().flatten()
	#num_nonzeros =np.array([user_feat_mat[i].count_nonzero() for i in range(n_user)])
	#user_feat_mat[np.eye(n_user)]=degrees / num_nonzeros
	user_feat_tensor = torch.sparse.FloatTensor(torch.tensor([rows, cols]), torch.from_numpy(np.array(values)), (n_user, nf_user_feat))


	return user_feat_tensor


def load_train_data(train_file):
	train_item, train_user = [], []
	n_user, m_item = 0, 0
	interaction_dict = dict()
	score_interaction_dict = dict()
	with open(train_file) as f:
		for l in f.readlines():
			if len(l) > 0:
				l = l.strip('\n').split(' ')
				uid = int(l[0])
				iid = int(l[1])
				rating = float(l[2])
				if rating not in score_interaction_dict.keys():
					score_interaction_dict[rating] = dict()
				if uid not in interaction_dict.keys():
					interaction_dict[uid] = []
				interaction_dict[uid].append(iid)
				if uid not in score_interaction_dict[rating].keys():
					score_interaction_dict[rating][uid]=[]
				score_interaction_dict[rating][uid].append(iid)
				train_item.append(iid)
				train_user.append(uid)
				m_item = max(m_item, iid)
				n_user = max(n_user, uid)
	n_user += 1
	m_item += 1
	r_score = len(list(score_interaction_dict.keys()))

	train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)
	train_mat[train_user, train_item] = 1.0

	items_D = np.sum(train_mat, axis = 0).reshape(-1)
	items_D[np.where(items_D == 0)] = 1
	users_D = np.sum(train_mat, axis = 1).reshape(-1)
	users_D[np.where(users_D == 0)] = 1
	return n_user, m_item, r_score, interaction_dict, score_interaction_dict, list(score_interaction_dict.keys()), items_D, users_D

def load_test_data(test_file):
	interaction_dict = dict()
	rating_dict = dict()
	with open(test_file) as f:
		for l in f.readlines():
			if len(l) > 0:
				l = l.strip('\n').split(' ')
				uid = int(l[0])
				iid = int(l[1])
				rating = float(l[2])
				if uid not in interaction_dict.keys():
					interaction_dict[uid] = []
				interaction_dict[uid].append(iid)
				if uid not in rating_dict.keys():
					rating_dict[uid] = []
				rating_dict[uid].append(rating)

	for u in rating_dict.keys():
		rating_dict[u] = torch.tensor(rating_dict[u])

	return interaction_dict, rating_dict

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

def initialize(config_file):
	config = configparser.ConfigParser()
	config.read(config_file)

	params = {}

	params['dim_embedding'] = config.getint('Model', 'dim_embedding')
	#params['dim_hidden'] = config.getint('Model', 'dim_hidden')
	
	params['train_file'] = config['Dataset']['train_file']
	params['test_file'] = config['Dataset']['test_file']
	params['user_side_file'] = config['Dataset']['user_side_file']
	params['item_side_file'] = config['Dataset']['item_side_file']
	params['nf_user_feat'] = config.getint('Dataset', 'dim_user_feature')
	params['mf_item_feat'] = config.getint('Dataset', 'dim_item_feature')

	params['max_iteration'] = config.getint('Training', 'max_iteration')	
	params['lr'] = config.getfloat('Training', 'learning_rate')
	params['criterion'] = config.getfloat('Training', 'stopping_criterion')
	params['early_stop'] = config.getint('Training','early_stop')
	omegas = list()
	omegas.append(config.getfloat('Training', 'omega_0'))
	omegas.append(config.getfloat('Training', 'omega_1'))
	omegas.append(config.getfloat('Training', 'omega_2'))
	omegas.append(config.getfloat('Training', 'omega_3'))
	omegas.append(config.getfloat('Training', 'omega_4'))
	omegas.append(config.getfloat('Training', 'omega_5'))
	params['omegas'] = omegas
	params['device'] = torch.device('cuda:'+ config['Training']['gpu'] if torch.cuda.is_available() else "cpu")
	params['init_scale'] = config.getfloat('Training', 'initial_scale')

	params['ipi'] = config.getint('Debugging', 'iter_per_info')
	params['upi'] = config.getint('Debugging', 'user_per_info')
	params['ips'] = config.getint('Debugging', 'iter_per_saving')
	params['model_path'] = config['Debugging']['model_saving_path']
	params['continue'] =config.getboolean('Debugging','continue')

	params['topk'] = config.getint('Testing', 'topk')

	randomization = config.getboolean('Security', 'randomization')
	params['randomization'] = randomization
	if randomization:
		params['lambda'] = config.getfloat('Security', 'lambda') 
		params['delta'] = config.getfloat('Security', 'delta')
		params['p'] =config.getfloat('Security', 'p')

	params['num_negative'] = config.getint('Training', 'num_negative')

	return params

def train(params, n_user, m_item, r_score, interaction_dict, score_interaction_dict, score_list, items_D, users_D, save_model = True, continue_ = False, pre_user_embeddings = None, pre_item_embeddings = None, test_interaction_dict = {}, test_rating_dict = {}):
	print('Dataset info: ', n_user, 'users,', m_item, 'items.')
	device = params['device']
	if params['randomization']:
		items_D = rr_items_D(items_D, n_user, m_item, params['p'])
	weightI, weightU, items_inv_sqrtD = get_weights(n_user, m_item, items_D, users_D, interaction_dict, params['omegas'], device)

	user_embeddings = params['init_scale'] * torch.randn((n_user, params['dim_embedding'])).to(device)
	item_embeddings = params['init_scale'] * torch.randn((m_item, params['dim_embedding'])).to(device)
	score_mats = params['init_scale'] * torch.randn((r_score, params['dim_embedding'], params['dim_embedding'] )).to(device)
	#vert_scale = torch.rand(1).to(device)
	#hori_scale = torch.rand(1).to(device)
	#hori_shift = torch.rand(1).to(device)

	user_embeddings.requires_grad = True
	item_embeddings.requires_grad = True
	score_mats.requires_grad = True
	#vert_scale.requires_grad = True
	#hori_scale.requires_grad = True
	#hori_shift.requires_grad = True

	print('Start training on', device, '.')
	loss = torch.tensor(0.0)
	early_count = 0
	best_rmse = np.inf
	for i in range(params['max_iteration']):
		norm_loss = cal_norm_loss(item_embeddings, user_embeddings, params['omegas'], device)
		smoothing_and_oversmoothing_loss = cal_smoothing_and_oversmoothing_loss(m_item, n_user, item_embeddings, user_embeddings, weightI, weightU, interaction_dict, params['num_negative'], params['omegas'], device)
		rating_loss = cal_rating_loss(r_score, item_embeddings, user_embeddings, score_mats, interaction_dict, score_interaction_dict, params['omegas'], device)
		last_loss = torch.clone(loss)		
		loss = smoothing_and_oversmoothing_loss + rating_loss + norm_loss
		
		loss.backward()
		user_embedding_gradients = params['lr'] * user_embeddings.grad
		item_embedding_gradients = params['lr'] * item_embeddings.grad
		score_mat_gradients = params['lr'] * score_mats.grad

		#vert_scale_gradient = params['lr'] * vert_scale
		#hori_scale_gradient = params['lr'] * hori_scale
		#hori_shift_gradient = params['lr'] * hori_shift
		if params['randomization']:
			item_embedding_gradients = cnr_items_gradient(item_embedding_gradients, n_user, m_item, params['delta'], params['lambda'], device)
		with torch.no_grad():
			user_embeddings -= user_embedding_gradients
			item_embeddings -= item_embedding_gradients
			score_mats -= score_mat_gradients
			#vert_scale -= vert_scale_gradient
			#hori_scale -= hori_scale_gradient
			#hori_shift -= hori_shift_gradient
			_ = user_embeddings.grad.zero_()
			_ = item_embeddings.grad.zero_()
			_ = score_mats.grad.zero_()
			#_ = vert_scale.grad.zero_()
			#_ = hori_scale.grad.zero_()
			#_ = hori_shift.grad.zero_()
			change = (loss - last_loss).abs() / loss
		if params['ipi'] > 0 and i % params['ipi'] == 0:
			rmse = 'None'
			if len(test_interaction_dict) > 0:
				rmse = test(test_interaction_dict, test_rating_dict, user_embeddings, item_embeddings, score_mats, score_list, device)
				if rmse <= best_rmse:
					early_count = 0
					best_rmse = rmse
				else:
					early_count += 1
				if early_count > params['early_stop']:
					print('Early_stop!')
					break
			print('Iteration',i,',','change:',change.data,', RMSE:', rmse, ', smoothing_and_oversmoothing_loss:', smoothing_and_oversmoothing_loss, ', rating_loss:', rating_loss, ', norm_loss:', norm_loss)
		if save_model and params['ips'] > 0 and i % params['ips'] == 0:
			esave(user_embeddings, params['model_path']+r'/user_embeddings.mod', '\nSaving at '+str(i)+'th iteration.', params['model_path']+r'/log.txt')
			esave(item_embeddings, params['model_path']+r'/item_embeddings.mod', '\nChange: '+str(change), params['model_path']+r'/log.txt')
			esave(score_mat, params['model_path']+r'/score_mat.mod', '\nrmse: '+str(rmse), params['model_path']+r'/log.txt')
		if change < params['criterion']:
			break
	print('Best RMSE:',best_rmse)
	return user_embeddings, item_embeddings, score_mats

def train_social(params, n_user, m_item, r_score, interaction_dict, score_interaction_dict, score_list, items_D, users_D, save_model = True, continue_ = False, pre_user_embeddings = None, pre_item_embeddings = None, test_interaction_dict = {}, test_rating_dict = {}):
	print('Dataset info: ', n_user, 'users,', m_item, 'items.', r_score, 'degrees of ratings.')
	device = params['device']
	if params['randomization']:
		items_D = rr_items_D(items_D, n_user, m_item, params['p'])
	weightI, weightU, items_inv_sqrtD = get_weights(n_user, m_item, items_D, users_D, interaction_dict, params['omegas'], device)
	
	print('Loading Side Info.')
	user_feats = load_social(params['user_side_file'], n_user, params['nf_user_feat']).to(device).to(torch.float)

	user_embeddings = params['init_scale'] * torch.randn((n_user, params['dim_embedding'])).to(device)
	item_embeddings = params['init_scale'] * torch.randn((m_item, params['dim_embedding'])).to(device)
	score_mats = params['init_scale'] * torch.randn((r_score, params['dim_embedding'], params['dim_embedding'] )).to(device)

	user_feat_W1 = params['init_scale'] * torch.randn((params['nf_user_feat'], params['dim_embedding'])).to(device)
	user_feat_b = params['init_scale'] * torch.randn(params['dim_embedding']).to(device)
	user_feat_W2 = params['init_scale'] * torch.randn((params['dim_embedding'], params['dim_embedding'])).to(device)

	user_embeddings.requires_grad = True
	item_embeddings.requires_grad = True
	score_mats.requires_grad = True
	user_feat_W1.requires_grad = True
	user_feat_b.requires_grad = True
	user_feat_W2.requires_grad = True

	user_fusions = user_embeddings + (user_feats.mm(user_feat_W1) + user_feat_b).sigmoid().mm(user_feat_W2)

	print('Start training on', device, '.')
	loss = torch.tensor(0.0)
	early_count = 0
	best_rmse = np.inf
	for i in range(params['max_iteration']):
		smoothing_and_oversmoothing_loss = cal_smoothing_and_oversmoothing_loss(m_item, n_user, item_embeddings, user_embeddings, weightI, weightU, interaction_dict, params['num_negative'], params['omegas'], device)
		norm_loss = cal_norm_loss_with_fusion(item_embeddings, user_embeddings, user_fusions, params['omegas'], device)
		rating_loss = cal_rating_loss(r_score, item_embeddings, user_fusions, score_mats, interaction_dict, score_interaction_dict, params['omegas'], device)
		last_loss = torch.clone(loss)		
		loss = smoothing_and_oversmoothing_loss + rating_loss + norm_loss
		
		loss.backward()
		user_embedding_gradients = params['lr'] * user_embeddings.grad
		item_embedding_gradients = params['lr'] * item_embeddings.grad
		score_mat_gradients = params['lr'] * score_mats.grad
		user_feat_W1_gradients = params['lr'] * user_feat_W1.grad
		user_feat_b_gradients = params['lr'] * user_feat_b.grad
		user_feat_W2_gradients = params['lr'] * user_feat_W2.grad

		#vert_scale_gradient = params['lr'] * vert_scale
		#hori_scale_gradient = params['lr'] * hori_scale
		#hori_shift_gradient = params['lr'] * hori_shift
		if params['randomization']:
			item_gradients = cnr_items_gradient(items_gradients, n_user, m_item, params['delta'], params['lambda'])
		with torch.no_grad():
			user_embeddings -= user_embedding_gradients
			item_embeddings -= item_embedding_gradients
			score_mats -= score_mat_gradients
			user_feat_W1 -=user_feat_W1_gradients
			user_feat_b -=user_feat_b_gradients
			user_feat_W2 -=user_feat_W2_gradients

			_ = user_embeddings.grad.zero_()
			_ = item_embeddings.grad.zero_()
			_ = score_mats.grad.zero_()
			_ = user_feat_W1.grad.zero_()
			_ = user_feat_b.grad.zero_()
			_ = user_feat_W2.grad.zero_()
			
			change = (loss - last_loss).abs() / loss
		user_fusions = user_embeddings + (user_feats.mm(user_feat_W1) + user_feat_b).sigmoid().mm(user_feat_W2)
		if params['ipi'] > 0 and i % params['ipi'] == 0:
			rmse = 'None'
			if len(test_interaction_dict) > 0:
				rmse = test(test_interaction_dict, test_rating_dict, user_fusions, item_embeddings, score_mats, score_list, device)
				if rmse <= best_rmse:
					early_count = 0
					best_rmse = rmse
				else:
					early_count += 1
				if early_count > params['early_stop']:
					print('Early_stop!')
					break
			print('Iteration',i,',','change:',change.data,', RMSE:', rmse, ', smoothing_and_oversmoothing_loss:', smoothing_and_oversmoothing_loss, ', rating_loss:', rating_loss, ', norm_loss:', norm_loss)
		if save_model and params['ips'] > 0 and i % params['ips'] == 0:
			esave(user_embeddings, params['model_path']+r'/side_user_embeddings.mod', '\nSaving at '+str(i)+'th iteration.', params['model_path']+r'/log.txt')
			esave(item_embeddings, params['model_path']+r'/side_item_embeddings.mod', '\nChange: '+str(change), params['model_path']+r'/log.txt')
			esave(user_feat_W1, params['model_path']+r'/user_feat_W1.mod', '\nrmse: '+str(rmse), params['model_path']+r'/log.txt')
			esave(user_feat_b, params['model_path']+r'/user_feat_b.mod', '\n', params['model_path']+r'/log.txt')
			esave(user_feat_W2, params['model_path']+r'/user_feat_b.mod', '\n', params['model_path']+r'/log.txt')
		if change < params['criterion']:
			break
	print('Best RMSE:',best_rmse)
	return user_fusions, item_embeddings, score_mats


def test(test_interaction_dict, test_rating_dict, user_embeddings, item_embeddings, score_mats, score_list, device):
	m_ratings = 0
	square_error = 0
	score_tensor = torch.tensor(score_list).to(device)
	with torch.no_grad():
		for user, items in test_interaction_dict.items():
			m_ratings += len(items)
			for i in range(len(items)):
				item = items[i]
				score_candidates = (user_embeddings[user].reshape(-1,1) * score_mats * item_embeddings[item].reshape(1,-1)).sum((1,2)).sigmoid()
				
				score_prop = score_candidates / score_candidates.sum()
				pred_score = score_tensor.reshape(1,-1).mm(score_prop.reshape(-1,1))
				
				#pred_score = score_tensor[score_candidates.argmax()]
				square_error += (pred_score - test_rating_dict[user][i])**2
	rmse = torch.sqrt(square_error / m_ratings)
	return  rmse

def eload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	return res

def esave(x, x_path, info, log_path):
	with open(x_path, 'wb') as f:
		pickle.dump(x, f)
	with open(log_path, 'a') as f:
		f.write(info)

if __name__ == "__main__":
	if len(sys.argv) > 1:
		config_file = sys.argv[1]
	else:
		config_file = 'config.ini'
	params = initialize(config_file)
	print('Hyper parameters (omega 1-5):',params['omegas'])
	print('Hyper parameters initial scale:', params['init_scale'])
	print('Dimensions of Embeddings:', params['dim_embedding'])
	user_embeddings_path = params['model_path']+r'/user_embeddings.mod'
	item_embeddings_path = params['model_path']+r'/item_embeddings.mod'
	if params['continue'] and os.path.exists(user_embeddings_path):
		print('Continue with last-time embeddings')
		pre_user_embeddings = eload(user_embeddings_path)
		pre_item_embeddings = eload(item_embeddings_path)
		continue_ = True
	else:
		pre_user_embeddings = None
		pre_item_embeddings = None
		continue_ = False
	n_user, m_item, r_score, interaction_dict, score_interaction_dict, score_list, items_D, users_D = load_train_data(params['train_file'])
	test_interaction_dict, test_rating_dict = load_test_data(params['test_file'])
	if len(params['user_side_file']) > 0:
		user_embeddings, item_embeddings, score_mats = train_social(params, n_user, m_item, r_score, interaction_dict, score_interaction_dict, score_list, items_D, users_D, True, continue_, pre_user_embeddings, pre_item_embeddings, test_interaction_dict, test_rating_dict)
	else:	
		user_embeddings, item_embeddings, score_mats = train(params, n_user, m_item, r_score, interaction_dict, score_interaction_dict, score_list, items_D, users_D, True, continue_, pre_user_embeddings, pre_item_embeddings, test_interaction_dict, test_rating_dict)
	
	rmse = test(test_interaction_dict, test_rating_dict, user_embeddings, item_embeddings, score_mats, score_list, params['device'])
	print('RMSE:', rmse)