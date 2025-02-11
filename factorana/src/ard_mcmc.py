# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

Bayesian Factor Analysis with MCMC sampling and ARD

250211 ToDo
- ARDが機能していないので修正する

@author: tadahaya
"""
import numpy as np
from numpy.linalg import solve
from sklearn.preprocessing import StandardScaler
from scipy.stats import wishart, gamma
from tqdm.auto import trange

from .base import BaseFA

class ARDMcmc(BaseFA):
    def __init__(
            self, K_max=10, a0=None, b0=None, nu=None, S=None, alpha_threshold=10, alpha_decay=0.5,
            lambda_w=1e-1, lambda_z=1e-1, sampling=True
            ):
        """
        Bayesian Factor Analysis with MCMC sampling and ARD

        Parameters
        ----------
        K_max : int, optional
            Maximum number of factors. The default is 10.
        a0 : float, optional
            Hyperparameter for alpha. The default is None.
        b0 : float, optional
            Hyperparameter for alpha. The default is None.
        nu : int, optional
            Hyperparameter for Psi. The default is None.
        S : np.ndarray, optional
            Hyperparameter for Psi. The default is None.
        alpha_threshold : float, optional
            Threshold for alpha. The default is 10.
        alpha_decay : float, optional
            Decay rate for alpha. The default is 0.5.
        lambda_w : float, optional
            Regularization parameter for W. The default is 1e-1.
        lambda_z : float, optional
            Regularization parameter for Z. The default is 1e-1.
        sampling : bool, optional
            Sampling flag. The default is True.
        
        """
        self.N, self.D = None, None
        self.K_max = K_max
        self.a0, self.b0 = a0, b0
        self.nu = nu
        self.S = S
        self.Z, self.W, self.Psi, self.alpha, self.K = None, None, None, None, None
        self.alpha_threshold = alpha_threshold
        self.alpha_decay = alpha_decay
        self.sampling = sampling
        self.lambda_w, self.lambda_z = lambda_w, lambda_z
        self.samples = {"W": 0, "Z": 0, "Psi": 0, "alpha": 0, "K": []}


    def fit(self, X, num_iter=2000, burn_in=500, tol=1e-3, eps=1e-6, normalize=True):
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
        tol : float, optional
            Tolerance for convergence. The default is 1e-3.
        eps : float, optional
            Epsilon for convergence. The default is 1e-6.
        normalize : bool, optional
            Normalize the data. The default is True.

        """
        if normalize:
            X = StandardScaler().fit_transform(X)
        N, D = X.shape
        self.D, self.N = D, N
        if self.nu is None:
            self.nu = D + 2
        if self.S is None:
            self.S = np.eye(D) * 0.1

        # init
        W = np.random.randn(D, self.K_max)
        Z = np.random.randn(N, self.K_max)
        Psi = np.eye(D) * 0.1
        alpha = alpha_init = np.random.uniform(1e-3, 1e-2, size=(self.K_max,))
        prev_W = np.copy(W)
        if self.a0 is None:
            self.a0 = D / 2 # based on D
        if self.b0 is None:
            self.b0 = np.mean(np.sum(W ** 2, axis=0)) # based on W

        # MCMC sampling
        for i in trange(num_iter):
            # 0. preparation
            alive_k = alpha <= self.alpha_threshold
            K = np.sum(alive_k)
            Z_a = Z[:, alive_k]
            W_a = W[:, alive_k]
            alpha_a = alpha[alive_k]

            # 1. update Z
            Sigma_Z = solve(np.eye(K) + np.einsum('ik,kj->ij', W_a.T, solve(Psi, W_a)), np.eye(K))  # (K, K)
            M_Z = np.einsum('ni,ij,jk->nk', X, solve(Psi, W_a), Sigma_Z)  # (N, K)
            # Ridge regularization
            Sigma_Z_reg = solve(self.lambda_z * np.eye(K) + np.einsum('nk,nl->kl', M_Z, M_Z), np.eye(K))  # (K, K)
            M_Z_reg = np.einsum('ni,nk->nk', X, M_Z) @ Sigma_Z_reg  # (N, K)
            if self.sampling:
                Z_a = M_Z_reg + np.random.randn(N, K) @ Sigma_Z_reg  # sampling Z
            else:
                Z_a = M_Z_reg

            # 2. update W
            Sigma_W = solve(np.diag(alpha_a) + np.einsum('nk,nl->kl', Z_a, Z_a) + self.lambda_w * np.eye(K), np.eye(K))  # (K, K)
            Z_Sigma_W = Z_a @ Sigma_W
            W_a = np.einsum('nd,nk->dk', X, Z_Sigma_W)  # (D, K)
            if self.sampling:
                 W_a = W_a + np.random.randn(D, K) @ Sigma_W  # sampling W

            # 3. update Psi
            error = X - np.einsum('nk,kl->nl', Z_a, W_a.T)
            Psi = wishart.rvs(df=self.nu, scale=self.S + np.einsum('ni,nj->ij', error, error))

            # 4. update entire W, Z
            W[:, alive_k] = W_a
            Z[:, alive_k] = Z_a

            # 5. update alpha (EMA)
            for k in range(self.K_max):
                a_old = alpha[k]
                if a_old <= self.alpha_threshold: # update only for alive factors
                    a_sample = gamma.rvs(self.a0 + D / 2, scale=1 / (self.b0 + 0.5 * np.sum(W[:, k] ** 2)))
                    a_new = self.alpha_decay * a_old + (1 - self.alpha_decay) * a_sample
                    alpha[k] = a_new

            # 6. save samples after burn-in
            if i >= burn_in:
                self.samples["W"] = (self.samples["W"] * (i - burn_in) + W.copy()) / (i - burn_in + 1)
                self.samples["Z"] = (self.samples["Z"] * (i - burn_in) + Z.copy()) / (i - burn_in + 1)
                self.samples["Psi"] = (self.samples["Psi"] * (i - burn_in) + Psi.copy()) / (i - burn_in + 1)
                self.samples["alpha"] = (self.samples["alpha"] * (i - burn_in) + alpha.copy()) / (i - burn_in + 1)
                self.samples["K"].append(K) # K is updated every iteration while the others are averaged due to cost

            # 7. convergence check
            prev_W_norm = np.linalg.norm(prev_W[:, alive_k], ord='fro')
            if i > burn_in and prev_W_norm > eps:
                change_ratio = np.linalg.norm(W_a - prev_W, ord='fro') / prev_W_norm
                if change_ratio < tol:
                    print(f"Converged: Iteration {i}, K={K}")
                    break
            prev_W = np.copy(W) # for checking convergence
            if K == 0:
                raise ValueError("!! All factors dropped. Check the conditions such as alpha_threshold !!")
            if i % 200 == 0:
                print(f"Iteration {i}, K={K}")

        self.W = self.samples["W"]
        self.Z = self.samples["Z"]
        self.Psi = self.samples["Psi"]
        self.alpha = self.samples["alpha"]
        self.K = self.samples["K"][-1]  # use the last K