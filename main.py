import os
import sys
from LapGLP.util.data_loader import load_train_data, load_test_data
from LapGLP.algorithm.train import initialize, train, train_social, eload

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