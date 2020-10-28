import os
import shutil
import urllib.request
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd


def get_dataloader(config):
    """Dispatcher that calls dataloader function depending on the configs.

    Parameters
    ----------
    config : SimpleNameSpace
        Contains configs values. Needs to at least have a `dataloader` field.
    
    Returns
    -------
    Corresponding dataloader.
    """
    if config.dataloader.lower() == 'umc':
        return load_umc(**config.__dict__)
    elif config.dataloader.lower() == 'compas':
        return load_compas(**config.__dict__)
    else:
        RuntimeError("The specified dataset has not been found")

#  --------------- Compas Dataset  ---------------

class CompasDataset(Dataset): #replace Dataset
    def __init__(self, data_path, verbose=True):
        """ProPublica Compas dataset.

        Dataset is read in from preprocessed compas data: `propublica_data_for_fairml.csv`
        from fairml github repo.
        Source url: 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'
        
        Following approach of Alvariz-Melis et al (SENN).
        
        Parameters
        ----------
        data_path : str
            Location of Compas data.
        """
        df = pd.read_csv(data_path)  # gonna be = pd.read_spss(data_path) for UMC data

        # don't know why square root
        # maybe because it weights it a bit 10 are way worse than 0,
        # but the difference between 10 and 15 is not that important
        df['Number_of_Priors'] = (df['Number_of_Priors'] / df['Number_of_Priors'].max()) ** (1 / 2)  # to normalize
        # get target
        compas_rating = df.score_factor.values  # This is the target?? (-_-)
        df = df.drop('score_factor', axis=1)

        pruned_df, pruned_rating = find_conflicting(df, compas_rating)
        if verbose:
            print('Finish preprocessing data..')

        self.X = pruned_df
        self.y = pruned_rating.astype(float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):  # probably to be removed
        return self.X.iloc[idx].values.astype(float), self.y[idx]


def load_compas(data_path='senn/datasets/data/compas/compas.csv', train_percent=0.8, batch_size=200,
                num_workers=0, valid_size=0.1, **kwargs):
    """Return compas dataloaders.
    
    If compas data can not be found, will download preprocessed compas data: `propublica_data_for_fairml.csv`
    from fairml github repo.
    
    Source url: 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'

    Parameters
    ----------
    data_path : str
        Path of compas data.
    train_percent : float
        What percentage of samples should be used as the training set. The rest is used
        for the test set.
    batch_size : int
        Number of samples in minibatches.

    Returns
    -------
    train_loader
        Dataloader for training set.
    valid_loader
        Dataloader for validation set.
    test_loader
        Dataloader for testing set.
    """
    if not os.path.isfile(data_path): #remove for UMC data
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        compas_url = 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'
        download_file(data_path, compas_url)
    dataset = CompasDataset(data_path)

    # Split into training and test
    train_size = int(train_percent * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size]) #replace with tf equivalent

    indices = list(range(train_size))
    validation_split = int(valid_size * train_size)
    train_sampler = SubsetRandomSampler(indices[validation_split:]) #replace with tf equivalent
    valid_sampler = SubsetRandomSampler(indices[:validation_split]) #replace with tf equivalent

    # Dataloaders
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **dataloader_args) #replace
    valid_loader = DataLoader(train_set, sampler=valid_sampler, **dataloader_args) #replace
    test_loader = DataLoader(test_set, shuffle=False, **dataloader_args) #replace

    return train_loader, valid_loader, test_loader


def find_conflicting(df, labels, consensus_delta=0.2):
    """
    Find examples with same exact feature vector but different label.

    Finds pairs of examples in dataframe that differ only in a few feature values.

    From SENN authors' code.

    Parameters
    ----------
    df : pd.Dataframe
        Containing compas data.
    labels : iterable
        Containing ground truth labels
    consensus_delta : float
        Decision rule parameter.

    Return
    ------
    pruned_df:
        dataframe with `inconsistent samples` removed.
    pruned_lab:
        pruned labels
    """

    def finder(df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df

    groups = []
    all_seen = set([])
    full_dups = df.duplicated(keep='first')
    for i in (range(len(df))):
        if full_dups[i] and (i not in all_seen):
            i_dups = finder(df, df.iloc[i])
            groups.append(i_dups.index)
            all_seen.update(i_dups.index)

    pruned_df = []
    pruned_lab = []
    for group in groups:
        scores = np.array([labels[i] for i in group])
        consensus = round(scores.mean())
        for i in group:
            if (abs(scores.mean() - 0.5) < consensus_delta) or labels[i] == consensus:
                # First condition: consensus is close to 50/50, can't consider this "outliers", so keep them all
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)


def download_file(store_path, url):
    """Download a file from `url` and write it to a file `store_path`.

    Parameters
    ----------
    store_path : str
        Data storage location.
    """
    # Download the file from `url` and save it locally under `file_name`
    with urllib.request.urlopen(url) as response, open(store_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
