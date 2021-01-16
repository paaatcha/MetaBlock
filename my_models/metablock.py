#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the Context Guider Cell (GCell)

If you find any bug or have some suggestion, please, email me.
"""

import torch.nn as nn
import torch

class MetaBlock(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U):
        super(MetaBlock, self).__init__()
        self.fb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V, U):
        t1 = self.fb(U)
        t2 = self.gb(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V