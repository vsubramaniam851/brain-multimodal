import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class TorchRidge:
    # Adapted from https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12
    # Runs ridge regression using a given alpha penalty
    def __init__(self, alpha, device, fit_intercept = True):
        self.alpha = alpha
        self.device = device
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0]), f'Number of stimuli do not match; X shape: {X.shape}, y shape: {y.shape}'
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1).to(self.device), X], dim = 1)
        
        if isinstance(self.alpha, float):
            lhs = X.T @ X 
            rhs = X.T @ y
            ridge = self.alpha*torch.eye(lhs.shape[0]).to(self.device)
            self.w = torch.linalg.lstsq(lhs + ridge, rhs)[0]
        else:
            #Multi-alpha operation. This can probably happen in one operation but my pytorch skills aren't good enough for that.
            assert self.alpha.shape[0] == y.shape[1]
            self.w = torch.empty(X.shape[1], y.shape[1]).to(self.device)
            for i in range(y.shape[1]):
                y_val = y[:, i].unsqueeze(1)
                alpha = self.alpha[i]
                lhs = X.T @ X
                rhs = X.T @ y_val
                ridge = alpha * torch.eye(lhs.shape[0]).to(self.device)
                w= torch.linalg.lstsq(rhs, lhs+ridge)[0] #NOTE: torch.linalg.lstsq switches the order of arguments from torch.lstsq.
                self.w[:, i] = w.squeeze()
    
    def predict(self, X):
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1).to(self.device), X], dim = 1)
        return X @ self.w

if __name__ == '__main__':
    #For testing
    device = torch.device('cuda')
    X = torch.randn(6070, 768).to(device)
    y = torch.randn(6070, 24974).to(device)
    alpha = torch.randn(24974)

    start_time = time.time()
    model = TorchRidge(alpha = alpha, device = device)
    model.fit(X, y)
    model.predict(X)
    print(f'Time taken: {time.time() - start_time}')