# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import os
import numpy as np
import pandas as pd

def GermanCredit(path_to_folder: str):
    """Loads the UCI German Credit dataset.
    
    https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    
    Parameters
    ----------
    
    path_to_folder
        Path to the folder where the file haberman.data is located
        
    Returns
    -------
    
    Tuple (X, y)
    
    """
    
    data = np.genfromtxt(os.path.join(path_to_folder, "german.data-numeric"))
    X = data[:,:-1]
    y = data[:,-1:].squeeze()
    return X, y

def HabermanSurvival(path_to_folder: str):
    """Loads the UCI Haberman dataset.
    
    https://archive.ics.uci.edu/ml/datasets/haberman%27s+survival
    
    Parameters
    ----------
    
    path_to_folder
        Path to the folder where the file haberman.data is located
        
    Returns
    -------
    
    Tuple (X, y)
    
    """
    
    data = np.genfromtxt(os.path.join(path_to_folder, "haberman.data"), delimiter=",")
    X = data[:,:-1]
    y = data[:,-1:].squeeze() - 1
    return X, y

def LiverDisorder(path_to_folder: str):
    """Loads the UCI Liver disorder dataset.
    
    https://archive.ics.uci.edu/ml/datasets/liver+disorders
    
    Parameters
    ----------
    
    path_to_folder
        Path to the folder where the file bupa.data is located
        
    Returns
    -------
    
    Tuple (X, y)
    
    """
    
    data = np.genfromtxt(os.path.join(path_to_folder, "bupa.data"), delimiter=",")
    X = data[:,:-1]
    y = data[:,-1:].squeeze() - 1
    return X, y

def BreastCancer():
    """Loads the breast cancer dataset with sklearn directly.
    
    Returns
    -------
    
    Tuple (X, y)
    
    """
    
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    return X, y

def Authentification(path_to_folder: str):
    """Loads the UCI authentification dataset.
    
    https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    
    Parameters
    ----------
    
    path_to_folder
        Path to the folder where the file data_banknote_authentication.txt is located
        
    Returns
    -------
    
    Tuple (X, y)    
    
    """
    
    data = np.genfromtxt(os.path.join(path_to_folder, "data_banknote_authentication.txt"), delimiter=",")
    X = data[:,:-1]
    y = data[:,-1:].squeeze()
    return X, y

def Ionosphere(path_to_folder: str):
    """Loads the UCI ionosphere dataset.
    
    https://archive.ics.uci.edu/ml/datasets/ionosphere
    
    Parameters
    ----------
    
    path_to_folder
        Path to the folder where the file ionosphere.data is located
        
    Returns
    -------
    
    Tuple (X, y)    
    
    """
    
    # data = np.genfromtxt(os.path.join(path_to_folder, "ionosphere.data"), delimiter=",")
    data = pd.read_csv(os.path.join(path_to_folder, "ionosphere.data"))
    X = data.iloc[:,:-1].values
    y_raw = data.iloc[:,-1:].values
    def map_fn(y):
        if y=='b':
            return 1
        elif y=="g":
            return 0
    y = [map_fn(yi) for yi in y_raw]
    return X, np.array(y)

def Sonar(path_to_folder: str):
    """Loads the UCI Sonar dataset.
    
    http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
    
    Parameters
    ----------
    
    path_to_folder
        Path to the folder where the file sonar.all-data is located
        
    Returns
    -------
    
    Tuple (X, y)    
    
    """
    
    data = pd.read_csv(os.path.join(path_to_folder, "sonar.all-data"))
    X = data.iloc[:,:-1].values
    y_raw = data.iloc[:,-1:].values
    def map_fn(y):
        if y=='R':
            return 1
        elif y=="M":
            return 0
    y = [map_fn(yi) for yi in y_raw]
    return X, np.array(y)

def Diabetes(path_to_folder: str):
    """Loads the UCI Diabetes dataset.
    
    Parameters
    ----------
    
    path_to_folder
        Path to the folder where the file diabetes.data is located
        
    Returns
    -------
    
    Tuple (X, y)
    
    """
    
    data = pd.read_csv(os.path.join(path_to_folder, "diabetes.data"))
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1:].values
    return X, y