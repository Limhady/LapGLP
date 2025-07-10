import torch
import numpy as np
import scipy.sparse as sp


def load_social(user_file, n_user, nf_user_feat):
	rows, cols, values = [], [], []
	with open(user_file) as f:
		for l in f.readlines():
			if len(l) > 0:
				l = l.strip('\n').split(' ')
				rows.append(int(l[0]))
				cols.append(int(l[1]))
				values.append(float(l[2]))
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