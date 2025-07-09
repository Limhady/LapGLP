#LapGLP: Approximating Infinite-layer Graph Convolutions with Laplacian for Federated Recommendation

##Abstract
Recommender systems (RS) have become crucial in helping users navigate the vast amount of online content available today. 
Graph neural networks (GNNs) have been applied to RS to capture complex user-item relationships, but existing methods compromise privacy or require centralized data storage.
Current works attempt to perform GNN-based RS under federated learning settings to prevent privacy leakage.
However, these works need to perform explicit graph propagation during training, which still introduces potential privacy leakage and data collusions.
To address these challenges, we propose a Laplacian-based model called Laplacian Graded Link Prediction (LapGLP) that leverages infinite graph propagation with a constant weight matrix.
Instead of actually performing the infinite graph propagation, the model abstracts the underlying relations between embeddings after propagation with a weighted minimum squared error problem.
Furthermore, we propose a federated framework named FedLapGLP to improve privacy in federated GNN-based RS, which splits the objective loss function into independent parts calculated by each user.
Experimental comparisons with state-of-the-art federated RS methods demonstrate the advantages of our proposed approach in terms of high-order connectivity, comprehensive graph information, social relations, full-interaction protection, collusion resistance, and user-embedding protection.
