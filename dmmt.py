"""DMMT
author: Dr. Michail Mamalakis
email:mixalis.mamalakis@gmail.com
email:mm2703@cam.ac.uk
affiliation: University of Cambridge
use agreement: please cite the follow work https://doi.org/10.1007/s00521-024-10182-6
"""

import sys
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
import os


class DMMT(nn.Module):
    def __init__(self,N=4,delta_k=0.02,delta_t=10):
        super(DMMT, self).__init__()
        self.sma = nn.AvgPool1d(kernel_size=delta_t, stride = 1)
        self.delta_k=delta_k

    def forward(self,loss,loss_prev,M,Mprev):
        m_count=0
        Mp=[]
        state=False
        for i in range(len(M)):
            Mi=[torch.tensor(mi) for mi in M[i]]
            Mk=torch.stack(Mi,dim=0)
            Mcompute=Mk.view(1,1,-1)
            Mnew=self.sma(Mcompute)
            print('metric sma new: ',Mnew, ' old metric: ',Mprev[i])
            if abs(Mnew)<=(abs(Mprev[i])+(self.delta_k)) and abs(Mnew)>=(abs(Mprev[i])-(self.delta_k)):
                m_count+=1
            Mp.append(Mnew)
        print('total Metric passes: ',m_count)
        print('new_loss: ',loss,' old loss: ',loss_prev)
        if m_count==len(M):
            if loss<=loss_prev:
                print('DMMT converted in loss: ',loss)
                state=True
                return state, Mp, loss
            else:
                return state, Mp, loss
        else:
            return state, Mp, loss

