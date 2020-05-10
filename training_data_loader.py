import configparser
from sqlalchemy import create_engine
import torch
from torch.utils import data
import pandas as pd
from numpy import array_split, concatenate
from random import shuffle
from itertools import product

class Dataset(data.Dataset):
    """
    Loads data from DB
    """

    def __init__(self, config_location: str,
                 pairs: tuple, seq_lenght: int,
                 num_features: int):
        """
        Load config file with DB connections
        Inputs:
        =======
        config_location (str): location of config file
        pairs (tuple): currency pairs
        seq_length (int): lenght of sequence to use for training
        num_features (int): number of features to use
        """
        
        config = configparser.ConfigParser()
        config.read(config_location)

        db_engine = create_engine(config['AIRFLOW']['postgres_conn'])

        book_data = pd.read_sql(
            f"select * from current_book where symbol in {tuple(pairs)}", db_engine)
        db_engine.dispose()

        # load data from DB
        book_data['askPrice'] = book_data['askPrice'].astype(float)
        book_data['bidPrice'] = book_data['bidPrice'].astype(float)
        book_data['askQty'] = book_data['askQty'].astype(float)
        book_data['bidQty'] = book_data['bidQty'].astype(float)
        book_data.index = pd.to_datetime(book_data['timestamp'])
        book_data.drop('timestamp', axis=1, inplace=True)

        book_data['bidQty/askQty'] = book_data['bidQty'] / (book_data['bidQty'] + book_data['askQty'])
        book_data.drop(['bidQty', 'askQty'], inplace=True, axis=1)

        self.labels = Dataset._compute_labels(data=book_data, pairs=pairs, columns=('bidPrice', 'askPrice'))

        self.stacked_sequences, self.stacked_labels = Dataset._stack_sequences(
                input_data=book_data,
                input_labels=self.labels,
                seq_lenght=seq_lenght,
                num_features=num_features,
                pairs=pairs)

        return None
    
    @staticmethod
    def _stack_sequences(input_data: pd.core.frame.DataFrame,
                         input_labels: pd.core.frame.DataFrame,
                         seq_lenght: int, num_features: int,
                         pairs: tuple):
        """
        Stack sequences for training & testing
        Inputs:
        =======
        input_data (pd.core.frame.DataFrame): raw data to be stacked
        input_labels (pd.core.frame.DataFrame): labels
        seq_length (int): length of the sequence
        num_features (int): number of features to be stacked
        pairs (tuple): pairs
        """

        indices = list(range(int(len(input_data)/len(pairs))-seq_lenght))

        indices_split = array_split(indices, len(indices)/seq_lenght)
        shuffle(indices_split)

        sequence_indices = product(indices_split, pairs)

        stacked_labels = []

        stacked = None

        # stack together
        for seq in sequence_indices:
            if len(seq[0]) == seq_lenght:
                if not stacked:
                    stacked = input_data.query(
                        f"symbol == '{seq[1]}'")[['bidPrice', 'askPrice', 'bidQty/askQty']].iloc[seq[0]].values.reshape(
                            1, num_features, seq_lenght)
                    # label index
                    temp_idx = input_data.query(f"symbol == '{seq[1]}'").iloc[seq[0][-1]].name
                    # label value
                    label_value = input_labels.query(f"index == '{temp_idx}'")[f"{seq[1]}_label"].values[0]
                    stacked_labels.append(label_value)
                    
                if stacked:
                    temp_len = len(input_data.query(
                        f"symbol == '{seq[1]}'")[['bidPrice', 'askPrice', 'bidQty/askQty']].iloc[seq[0]])
                    temp_stacked = input_data.query(
                        f"symbol == '{seq[1]}'")[['bidPrice', 'askPrice', 'bidQty/askQty']].iloc[seq[0]].values.reshape(
                            1, num_features, temp_len)
                    stacked = concatenate([stacked, temp_stacked])
                    # label index
                    temp_idx = input_data.query(f"symbol == '{seq[1]}'").iloc[seq[0][-1]].name
                    # label value
                    label_value = input_labels.query(f"index == '{temp_idx}'")[f"{seq[1]}_label"].values[0]
                    stacked_labels.append(label_value)
 
            else:
                pass
        
        return stacked, stacked_labels
    
    @staticmethod
    def _compute_labels(data: pd.core.frame.DataFrame,
                        pairs: tuple,
                        columns: tuple,):
        """
        Given the data, compute labels
        Label - is the next bid price higher than the current spread?
        """
        
        for idx, cols in enumerate(columns):
            if idx == 0:
                prices_pivoted = pd.pivot_table(data, index=data.index, columns='symbol', values=cols)
                prices_pivoted.rename(
                    columns=dict(
                        zip(prices_pivoted.columns, [f"{i}_{cols}" for i in prices_pivoted.columns])), inplace=True)
                labels = prices_pivoted
            else:
                prices_pivoted = pd.pivot_table(data, index=data.index, columns='symbol', values=cols)
                prices_pivoted.rename(
                    columns=dict(
                        zip(prices_pivoted.columns, [f"{i}_{cols}" for i in prices_pivoted.columns])), inplace=True)
                labels = pd.merge(labels, prices_pivoted, left_index=True, right_index=True)
        
        labels_columns = []
        
        for pair in pairs:
            # 1. Shift bidPrice up by one
            labels[f"{pair}_bidPrice_shifted"] = labels[f"{pair}_bidPrice"].shift(-1)
            
            # 2. Calculate difference between current ask and future bid
            labels['bid_-1_ask'] = labels[f"{pair}_bidPrice_shifted"].values - labels[f"{pair}_askPrice"].values
            
            # 3. Convert to label
            labels[f"{pair}_label"] = labels['bid_-1_ask'].map(lambda x: 1 if x > 0 else 0)
            
            # 4. Drop used columns
            labels.drop(['bid_-1_ask', f"{pair}_bidPrice_shifted"], axis=1, inplace=True)
            
            # 5. Select only labels columns
            labels_columns.append(f"{pair}_label")
        
        return labels[labels_columns]

    def __len__(self):
        """
        Returns total number of samples in
        the dataset
        """

        return len(self.stacked_sequences)
    
    def __getitem__(self, index):
        """
        Generate samples of data
        """

        x = torch.tensor(self.stacked_sequences).float()[index]
        y = torch.tensor(self.stacked_labels).long()[index]

        return x, y