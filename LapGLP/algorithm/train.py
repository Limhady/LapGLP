import torch
import numpy as np
import configparser
import pickle
from LapGLP.algorithm.loss import get_weights, rr_items_D, cnr_items_gradient, cal_smoothing_and_oversmoothing_loss, cal_rating_loss, cal_norm_loss, cal_norm_loss_with_fusion
from LapGLP.util.data_loader import load_social

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