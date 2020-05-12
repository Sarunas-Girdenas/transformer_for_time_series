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
TRAIN_SET_SIZE = 0.85
SEQ_LENGTH = 16

full_data_set = Dataset(
    config_location='../ml_models_for_airflow/dbs3_config.ini',
    pairs=pairs,
    seq_lenght=SEQ_LENGTH,
    num_features=3)

train_set_size = int(len(full_data_set)*TRAIN_SET_SIZE)
test_set_size = len(full_data_set) - train_set_size

trainset, testset = data.random_split(full_data_set,
                                    [train_set_size, test_set_size]
                                )

def define_model(trial):
    """
    Define model structure
    """

    transformer = Transformer(
        emb=SEQ_LENGTH,
        heads=trial.suggest_int('num_attention_heads', 1, 5),
        depth=trial.suggest_int('num_transformer_blocks', 1, 5),
        num_features=3,
        interpolation_factor=trial.suggest_int('interpolation_factor', 1, 20),
        dropout=trial.suggest_uniform("dropout", 0.0, 0.5)
        )

    transformer.apply(Transformer.init_weights)

    return transformer

def objective(trial):
    """
    Prepare training data (lenght of sequences)
    and train the model.
    """

    transformer = define_model(trial)

    batch_size = trial.suggest_int('batch_size', 16, 512)

    train_generator = data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    test_generator = data.DataLoader(
        testset,
        batch_size=len(testset),
        shuffle=True,
        num_workers=4)

    num_epochs = trial.suggest_int('num_epochs', 1, 70)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(transformer.parameters(), lr=lr)
    learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(i / (10.0 / batch_size), 1.0))

    criterion = torch.nn.BCELoss()
    train_auc = []
    test_auc = []

    for ep in range(num_epochs):
        transformer.train()
        epoch_loss = 0
        temp_train_auc = 0
        
        for train_x, train_y in train_generator:
            
            predictions = transformer(train_x)
            loss = criterion(predictions, train_y.view(-1, 1))
            epoch_loss += loss.item()
            try:
                temp_train_auc += roc_auc_score(
                    train_y.numpy(), predictions.detach().numpy())
            except ValueError:
                temp_train_auc += 0.5
            
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(transformer.parameters(), GRAD_CLIPPING_VAL)

            optimizer.step()
            learning_rate_scheduler.step()
        
        train_auc.append(temp_train_auc/len(train_generator))
        
        with torch.no_grad():
            transformer.eval()
            temp_test_auc = 0
            for test_x, test_y in test_generator:
                predictions = transformer(test_x)
                temp_test_auc += roc_auc_score(
                    test_y.numpy(), predictions.numpy())

        test_auc.append(temp_test_auc/len(test_generator))

        if ep % 5 == 0:
            print('Epoch: {:03d}, Loss: {:.5f}'.format(ep, epoch_loss/(ep+1)))
    
    return test_auc[-1]

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=12000, n_jobs=1)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))