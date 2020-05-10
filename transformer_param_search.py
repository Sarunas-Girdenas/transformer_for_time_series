import pandas as pd
import numpy as np
import torch
import configparser
import torch.optim as optim
from ast import literal_eval
from sklearn.metrics import roc_auc_score
from torch import nn
from training_data_loader import Dataset
from torch.utils import data
from transformer import Transformer
import optuna

# Read in config
config = configparser.ConfigParser()
config.read('../ml_models_for_airflow/dbs3_config.ini')

pairs_mapping = literal_eval(config['MODEL']['pairs_mapping'])
pairs = tuple(pairs_mapping.values())

GRAD_CLIPPING_VAL = 1.0

def define_model(trial):
    """
    Define model structure
    """

    seq_lenght = trial.suggest_int('seq_length', 3, 100)

    transformer = Transformer(
        emb=seq_lenght,
        heads=trial.suggest_int('num_attention_heads', 2, 20),
        depth=trial.suggest_int('num_transformer_blocks', 2, 50),
        num_classes=2,
        num_features=3,
        )

    return transformer, seq_lenght

def objective(trial):
    """
    Prepare training data (lenght of sequences)
    and train the model.
    """

    transformer, seq_lenght = define_model(trial)

    full_data_set = Dataset(
        config_location='../ml_models_for_airflow/dbs3_config.ini',
        pairs=pairs,
        seq_lenght=seq_lenght,
        num_features=3)
    
    train_set_size = int(len(full_data_set)*0.85)
    test_set_size = len(full_data_set) - train_set_size

    trainset, testset = data.random_split(full_data_set,
                                     [train_set_size, test_set_size]
                                    )
    
    batch_size = trial.suggest_int('batch_size', 16, 612)

    train_generator = data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    test_generator = data.DataLoader(
        testset,
        batch_size=len(testset),
        shuffle=True,
        num_workers=1)
    
    num_epochs = trial.suggest_int('num_epochs', 1, 50)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(transformer.parameters(), lr=lr)

    criterion = torch.nn.NLLLoss()
    train_auc = []
    test_auc = []

    for ep in range(num_epochs):
        transformer.train()
        epoch_loss = 0
        temp_train_auc = 0
        
        for train_x, train_y in train_generator:
            
            predictions = transformer(train_x)
            loss = criterion(predictions, train_y)
            epoch_loss += loss.item()
            try:
                temp_train_auc += roc_auc_score(
                    train_y.numpy(), torch.exp(predictions)[:, 1].detach().numpy())
            except ValueError:
                temp_train_auc += 0.5
            
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm(transformer.parameters(), GRAD_CLIPPING_VAL)

            optimizer.step()
        
        train_auc.append(temp_train_auc/len(train_generator))
        
        with torch.no_grad():
            transformer.eval()
            temp_test_auc = 0
            for test_x, test_y in test_generator:
                predictions = transformer(test_x)
                temp_test_auc += roc_auc_score(
                    test_y.numpy(), torch.exp(predictions)[:, 1].numpy())

        test_auc.append(temp_test_auc/len(test_generator))

        if ep % 5 == 0:
            print('Epoch: {:03d}, Loss: {:.5f}'.format(ep, epoch_loss/(ep+1)))
    
    return test_auc[-1]

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=12000, n_jobs=2)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))