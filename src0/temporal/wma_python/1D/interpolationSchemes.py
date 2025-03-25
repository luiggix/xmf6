# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:26:49 2024

@author: leo_teja
"""

import numpy as np

def upwind_1D(i, h, q):
    
    if(h[i]<=h[i+1]):
        qx_e=q[i+1]
    else:
        qx_e=q[i]
        
    if(h[i-1]<=h[i]):
        qx_w=q[i]
    else:
        qx_w=q[i-1]
        
    return (qx_e, qx_w)