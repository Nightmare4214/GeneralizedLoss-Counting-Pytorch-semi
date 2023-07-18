#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import torch


class PerCost:
    def __call__(self, X, Y):
        x_col = X.unsqueeze(-2)
        y_lin = Y.unsqueeze(-3)
        C = torch.linalg.vector_norm(x_col - y_lin, ord=2, dim=-1)
        s = (x_col[:, :, :, -1] + y_lin[:, :, :, -1]) / 2
        s = s * 0.2 + 0.5
        return torch.exp(C / s) - 1


class ExpCost:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, X, Y):
        x_col = X.unsqueeze(-2)
        y_lin = Y.unsqueeze(-3)
        C = torch.linalg.vector_norm(x_col - y_lin, ord=2, dim=-1)
        return torch.exp(C / self.scale) - 1.


class L2_DIS:
    factor = 1 / 32

    @staticmethod
    def __call__(X, Y):
        """
        X.shape = (batch, M, D)
        Y.shape = (batch, N, D)
        returned cost matrix's shape is ()
        """
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        C = ((x_col - y_row) ** 2).sum(dim=-1) / 2
        return C * L2_DIS.factor


class PNormCost:
    def __init__(self, p):
        self.p = p

    def __call__(self, X, Y):
        x_col = X.unsqueeze(-2)
        y_lin = Y.unsqueeze(-3)
        C = torch.linalg.vector_norm(x_col - y_lin, ord=self.p, dim=-1)
        return C
