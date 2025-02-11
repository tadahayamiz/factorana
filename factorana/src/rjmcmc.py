# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

Bayesian Factor Analysis with Reversible Jump MCMC

250211 ToDo
- BaseFAを継承させた形にする

@author: tadahaya
"""
import numpy as np
from tqdm.auto import trange
from sklearn.preprocessing import StandardScaler

from .base import BaseFA

class RJMcmc(BaseFA):
    def __init__(
            self, X, K_init=3, max_K=10, lambda_reg=0.1, sigma2=0.1, normalize=True, l2norm=False
            ):
        # X (N, D)
        self.X = X
        if normalize:
            self.X = StandardScaler().fit_transform(self.X)  # 標準化
        if l2norm:
            self.X = self.X / np.linalg.norm(self.X, axis=1, keepdims=True) # L2 normによる正規化
        self.N, self.D = X.shape
        self.K = K_init  # 初期K
        self.max_K = max_K  # 最大K
        self.lambda_reg = lambda_reg  # 正則化項
        self.sigma2 = sigma2  # ノイズ分散

        # 初期化
        self.W = np.random.randn(self.D, self.K)
        self.Z = np.random.randn(self.N, self.K)

        # MCMC のトレース保存
        self.trace_K = []
        self.trace_W = []
        self.trace_Z = []

    def update_Z(self):
        """Z の更新"""
        K = self.W.shape[1]  # 現在のK
        ZZt = self.W.T @ self.W + self.sigma2 * np.eye(K)
        self.Z = self.X @ self.W @ np.linalg.pinv(ZZt)

    def update_W(self):
        """W の更新"""
        K = self.Z.shape[1]  # 現在のK
        ZZt = self.Z.T @ self.Z + self.lambda_reg * np.eye(K)
        self.W = np.linalg.solve(ZZt, self.Z.T @ self.X).T

    def rj_mcmc_step(self):
        """RJ-MCMC による K の増減"""
        # K に応じて p_birth, p_death を動的に決定
        p_birth = 0.5 * (1 - self.K / self.max_K)
        p_death = 0.5 * (self.K / self.max_K)

        if np.random.rand() < p_birth and self.K < self.max_K:
            # K を増やす (Birth move)
            W_new = np.random.randn(self.D, 1) * 2  # 広めにサンプリング
            Z_new = np.random.randn(self.N, 1) * 2  # 広めにサンプリング
            self.W = np.hstack([self.W, W_new])
            self.Z = np.hstack([self.Z, Z_new])
            self.K += 1

        elif np.random.rand() < p_death and self.K > 1:
            # K を減らす (Death move)
            idx = np.random.choice(self.K)  # 削除する列を選択
            self.W = np.delete(self.W, idx, axis=1)
            self.Z = np.delete(self.Z, idx, axis=1)
            self.K -= 1

    def fit(self, iterations=1000, burn_in_ratio=0.3):
        """MCMC の実行"""
        burn_in = int(iterations * burn_in_ratio)

        for t in trange(iterations):
            self.update_W()
            self.update_Z()
            self.rj_mcmc_step()

            # バーンイン後のサンプルを保存
            if t >= burn_in:
                self.trace_K.append(self.K)
                self.trace_W.append(self.W.copy())
                self.trace_Z.append(self.Z.copy())

    def get_estimated_K(self):
        """推定された K の最頻値を返す"""
        return max(set(self.trace_K), key=self.trace_K.count)`