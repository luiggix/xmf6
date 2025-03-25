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



def upwind_2D(nx, ny, i, j,  h, qx, qy):
    
    if((i>0)and(i<nx-1)and(j>0)and(j<ny-1)): 
    
        if (h[i][j]<=h[i+1][j]):
            qx_e=qx[i+1][j]
        else:
            qx_e=qx[i][j]
    
    
        if (h[i][j]<=h[i-1][j]):
            qx_w=qx[i-1][j]
        else:
            qx_w=qx[i][j]
    
    
        if (h[i][j]<=h[i][j+1]):
            qy_n=qy[i][j+1]
        else:
            qy_n=qy[i][j]
    
    
        if (h[i][j]<=h[i][j-1]):
            qy_s=qy[i][j-1]
        else:
            qy_s=qy[i][j]


    if((i==0)and(j>0)and(j<ny-1)): #Volumenes en frontera oeste W

        if (h[i][j]<=h[i+1][j]):
            qx_e=qx[i+1][j]
        else:
            qx_e=qx[i][j]
            
        qx_w=qx[i][j]
    
        if (h[i][j]<=h[i][j+1]):
            qy_n=qy[i][j+1]
        else:
            qy_n=qy[i][j]
    
    
        if (h[i][j]<=h[i][j-1]):
            qy_s=qy[i][j-1]
        else:
            qy_s=qy[i][j]


    if((i==nx-1)and(j>0)and(j<ny-1)):  #Volumenes en frontera Este
        
        qx_e=qx[i][j]
            
        if (h[i][j]<=h[i-1][j]):
            qx_w=qx[i-1][j]
        else:
            qx_w=qx[i][j]
    
        if (h[i][j]<=h[i][j+1]):
            qy_n=qy[i][j+1]
        else:
            qy_n=qy[i][j]
    
    
        if (h[i][j]<=h[i][j-1]):
            qy_s=qy[i][j-1]
        else:
            qy_s=qy[i][j]


    if((i>0)and(i<nx-1)and(j==0)): #Volumenes en frontera sur S

        if (h[i][j]<=h[i+1][j]):
            qx_e=qx[i+1][j]
        else:
            qx_e=qx[i][j]
    
    
        if (h[i][j]<=h[i-1][j]):
            qx_w=qx[i-1][j]
        else:
            qx_w=qx[i][j]
    
    
        if (h[i][j]<=h[i][j+1]):
            qy_n=qy[i][j+1]
        else:
            qy_n=qy[i][j]
        
        qy_s=qy[i][j]


    if((i>0)and(i<nx-1)and(j==ny-1)): #Volumenes en frontera  N

        if (h[i][j]<=h[i+1][j]):
            qx_e=qx[i+1][j]
        else:
            qx_e=qx[i][j]
    
    
        if (h[i][j]<=h[i-1][j]):
            qx_w=qx[i-1][j]
        else:
            qx_w=qx[i][j]
            
        qy_n=qy[i][j]
       
        if (h[i][j]<=h[i][j-1]):
            qy_s=qy[i][j-1]
        else:
            qy_s=qy[i][j]


 
    if((i==0)and(j==0)):   #Nodo de esquina (0,0) solo se hacen las comparaciones con los nodos internos
        
        if (h[i][j]<=h[i+1][j]):
            qx_e=qx[i+1][j]
        else:
            qx_e=qx[i][j]
    
    
        qx_w=qx[i][j]   #Se asigna frontera oeste
    
    
        if (h[i][j]<=h[i][j+1]):
            qy_n=qy[i][j+1]
        else:
            qy_n=qy[i][j]
            
        
        qy_s=qy[i][j]  #Se asigna frontera sur
        



    if((i==nx-1)and(j==0)):   #Nodo de esquina (nx-1,0) solo se hacen las comparaciones con los nodos internos
        
        qx_e=qx[i][j]
    
        if (h[i][j]<=h[i-1][j]):
            qx_w=qx[i-1][j]
        else:
            qx_w=qx[i][j]
   
    
        if (h[i][j]<=h[i][j+1]):
            qy_n=qy[i][j+1]
        else:
            qy_n=qy[i][j]
            
        qy_s=qy[i][j]  #Se asigna frontera sur


    if((i==0)and(j==ny-1)):   #Nodo de esquina (0,ny-1) solo se hacen las comparaciones con los nodos internos
        
        if (h[i][j]<=h[i+1][j]):
            qx_e=qx[i+1][j]
        else:
            qx_e=qx[i][j]
    
    
        qx_w=qx[i][j]   #Se asigna frontera oeste
        
        qy_n=qy[i][j]  #Se asigna frontera norte
                    
        if (h[i][j]<=h[i][j-1]):
            qy_s=qy[i][j-1]
        else:
            qy_s=qy[i][j]


    if((i==nx-1)and(j==ny-1)):   #Nodo de esquina (nx-1,ny-1) solo se hacen las comparaciones con los nodos internos
        
        qx_e=qx[i][j]
    
        if (h[i][j]<=h[i-1][j]):
            qx_w=qx[i-1][j]
        else:
            qx_w=qx[i][j]
        
        qy_n=qy[i][j]
    
        if (h[i][j]<=h[i][j-1]):
            qy_s=qy[i][j-1]
        else:
            qy_s=qy[i][j]

        
    return (qx_e, qx_w, qy_n, qy_s)

