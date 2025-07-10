# LapGLP: Approximating Infinite-layer Graph Convolutions with Laplacian for Federated Recommendation
## Abstract
Recommender systems (RS) have become crucial in helping users navigate the vast amount of online content available today. 
Graph neural networks (GNNs) have been applied to RS to capture complex user-item relationships, but existing methods compromise privacy or require centralized data storage.
Current works attempt to perform GNN-based RS under federated learning settings to prevent privacy leakage.
However, these works need to perform explicit graph propagation during training, which still introduces potential privacy leakage and data collusions.
To address these challenges, we propose a Laplacian-based model called Laplacian Graded Link Prediction (LapGLP) that leverages infinite graph propagation with a constant weight matrix.
Instead of actually performing the infinite graph propagation, the model abstracts the underlying relations between embeddings after propagation with a weighted minimum squared error problem.
Furthermore, we propose a federated framework named FedLapGLP to improve privacy in federated GNN-based RS, which splits the objective loss function into independent parts calculated by each user.
Experimental comparisons with state-of-the-art federated RS methods demonstrate the advantages of our proposed approach in terms of high-order connectivity, comprehensive graph information, social relations, full-interaction protection, collusion resistance, and user-embedding protection.

## Highlights
![frs](https://github.com/user-attachments/assets/405bc648-8222-4e4e-af80-9f87f4e092f3)
- Prevents privacy leakage with ​**gradient randomization techniques**​
- Captures ​**high-order user-item relationships**​ without explicit graph propagation
- Avoids data collusions through ​**decentralized computation**​
- Incorporates ​**social relations**​ via user-side features

## Architectural Overview
![fedlapglp](https://github.com/user-attachments/assets/4162a29b-8a5c-4cc6-8dd5-3945f46cd521)
### Abstraction of Infinite Graph Convolutions through Laplacian Constraints
This component replaces explicit multi-layer graph propagation with a Laplacian-based constraint mechanism to approximate infinite graph convolutions. Instead of performing iterative message-passing steps (which risk privacy leaks and computational overhead), LapGLP formulates the graph convolution as a weighted minimum squared error (MSE) problem. 
### Multi-Loss Optimization Framework
The architecture incorporates a multi-objective loss function to handle explicit feedback data and graph properties simultaneously. This optimization involves:
- ​Weighted MSE for Graph Smoothness: Penalizes deviations in embeddings to maintain structural consistency, using a smoothness loss derived from Laplacian constraints.
​- Probabilistic Rating Loss for Explicit Feedback: Transforms explicit ratings (e.g., 1–5 stars) into graded categorical distributions.
### Federated Computation Flow for Privacy Preservation
This component outlines the federated learning workflow, splitting computations into client-local and server-global phases to protect user data.

## Installation
```bash
python main.py
```
git clone https://github.com/yourusername/LapGLP.git
cd LapGLP
pip install -r requirements.txt

## Requirements
This project requires the following environment:
```
Python==3.12.7
torch==2.5.1  
numpy==1.26.4  
scipy==1.13.1  
IPython==8.12.3
```

## Usage
### Default Configuration
```bash
python main.py
```
Uses `config.ini` in working directory automatically
### Custom Configuration
```bash
python main.py your_config_file.ini
```
## Configuration Parameters (config.ini)
[Model]
- dim_embedding: Dimensionality of user/item embeddings
- batch_size: Training batch size

[Dataset]
- train_file: Path to training data
- test_file: Path to test data
- user_side_file: Path to user social relationship file
- dim_user_feature: Dimensionality of user features
- dim_item_feature: Dimensionality of item features

[Training]
- gpu: GPU device ID to use
- max_iteration: Maximum training iterations
- learning_rate: Optimization learning rate
- stopping_criterion: Early stopping threshold
- early_stop: Validation patience iterations
- num_negative: Negative items per user in oversmoothing loss computation
- omega_0 to omega_5: Multi-task loss weights

[Debugging]
- iter_per_info: Training log frequency
- iter_per_saving: Model save frequency
- model_saving_path: Model storage directory
- continue: Resume training flag
  
[Security]
- randomization: Privacy protection toggle
- lambda: Laplacian noise scale
- delta: Gradient clipping threshold
- p: Random response probability

## Dataset File Format
To successfully run this recommender system with your own data, please prepare the following files in the specified formats. All files are ​space-delimited text files​ with no header row.

### Training Data File (train_file)
- Format: user_id item_id rating_value
​- Purpose: Contains user-item interactions for model training
​- Required: Yes
​- Characteristics:
-- Each line represents one interaction
-- IDs must be non-negative integers (start from 0)
-- Ratings should be continuous values
-- File must include all items that appear in test file

### Test Data File (test_file)
- Format: user_id item_id rating_value
- ​Purpose: Used for model evaluation
​- Required: Yes
​- Characteristics:
-- Same format as training file
-- Must include only users present in training data

### User Social File (user_side_file)
- Format: user_id user_id relation_weight
​- Purpose: Provides supplementary user social information
​- Required: No (can be omitted)
​- Characteristics:
-- Represents user social relations in sparse format
-- relation weights typically in [0,1] range

## Directory Structure
```
LapGLP/
├── datasets/              # Sample data
├── LapGLP/
│   ├── util               # Utilities
│   │   └── data_loader.py # Load and process training, testing and social-information data
│   └── algorithm          # Code for the LapGLP framework
│       ├── loss.py        # multi-loss computation
│       └── train.py       # Training process
├── main.py                # Entry point
├── config.ini             # Default configuration templates
└── requirements.txt       # Dependencies 
```
