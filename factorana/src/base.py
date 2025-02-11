# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

Base class for factor analysis

@author: tadahaya
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, LinAlgError
from scipy.stats import wishart, gamma
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange

class BaseFA:
    def __init__(self):
        """
        Base class for Factor Analysis

        """
        pass

    def fit(self, X, num_iter=2000, burn_in=500):
        """
        Fit the model to the data

        Parameters
        ----------
        X : np.ndarray
            Data matrix (N x D)

        num_iter : int, optional
            Number of iterations. The default is 2000.

        burn_in : int, optional
            Burn-in period. The default is 500.

        Returns
        -------
        None.

        """
        raise NotImplementedError


    def get_contributions(self):
        """ Get the contributions of each factor """
        if self.W is None:
            raise ValueError("!! Model is not fitted yet. !!")
        factor_variance = np.sum(self.W**2, axis=0)
        total_variance = np.sum(factor_variance)
        contributions = factor_variance / total_variance
        return contributions


    def plot_hinton(self, feature_labels=None, factor_labels=None, figscale=0.5):
        """
        Plot the Hinton diagram of the factor matrix W with axis labels.

        Parameters
        ----------
        feature_labels : list, optional
            List of feature labels. The default is None.
        factor_labels : list, optional
            List of factor labels. The default is None. 
        figscale : float, optional
            Figure scale. The default is 0.5.   
        
        """
        # get the shape of the factor matrix
        rows, cols = self.W.shape  # (D, K)
        
        plt.figure(figsize=(cols * figscale, rows * figscale))
        ax = plt.gca()

        for (i, j), value in np.ndenumerate(self.W):
            # the size of the square is proportional to the absolute value
            size = np.sqrt(np.abs(value))  # sqrt for better visibility
            facecolor = 'white' if value > 0 else 'black'  # positive: white, negative: black
            # draw a square
            ax.add_patch(
                plt.Rectangle((j - size / 2, rows - i - size / 2), size, size, facecolor=facecolor, edgecolor="black")
            )
        
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        
        # Add labels to x-axis (factor labels) and y-axis (feature labels)
        if factor_labels:
            ax.set_xticklabels(factor_labels)
        if feature_labels:
            ax.set_yticklabels(feature_labels)
        
        # Set axis titles
        ax.set_xlabel('Factors')
        ax.set_ylabel('Features')
        
        ax.set_aspect('equal')
        plt.gca().invert_yaxis()  # invert y axis
        plt.grid(False)
        plt.show()


def generate_toy_data(N, D, K, noise_level=0.1):
    """
    Generate toy data for factor analysis

    Parameters
    ----------
    N : int
        Number of samples.
    D : int
        Number of features.
    K : int
        Number of factors. 
    noise_level : float, optional
        Noise level. The default is 0.1.

    """
    # True factor loading W (D x K)
    W_true = np.random.randn(D, K)    
    # True factor matrix Z (N x K)
    Z_true = np.random.randn(N, K)
    # Observed data X (N x D)
    noise = np.random.randn(N, D) * noise_level
    X = Z_true @ W_true.T + noise
    return X, W_true, Z_true