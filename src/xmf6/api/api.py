"""
@author: Luis M. de la Cruz [Updated on Wed May 14 18:00:33 CST 2025].
"""

import os, sys
import numpy as np
import flopy

# Agrega la carpeta que contiene el módulo osys
mod_path = os.path.abspath(os.path.join("../.."))
if mod_path not in sys.path:
    sys.path.append(mod_path)
import xmf6

def build_mat(mf6):
    """
    Construye la matriz del sistema.

    Parameters
    ----------
    mf6: ModflowApi
        Objeto para accedar a toda la funcionalidad de la API.

    Return
    ------
    Atest: numpy.ndarray
        Matriz del sistema en formato denso.
        
    A, IA, JA: numpy.ndarray
        Arreglos de numpy con la información de la matriz en formato CRS
    """
    # Obtiene el número de renglones y columnas del sistema
    NCOL = mf6.get_value(mf6.get_var_address("NCOL", "SLN_1"))
    NROW = mf6.get_value(mf6.get_var_address("NROW", "SLN_1"))

    # Obtiene los coeficientes de la matriz en formato CRS (Compressed Row Storage)
    # A: Coeficientes, JA: índices de la columna, IA: índice de inicio del renglón en JA.
    A = mf6.get_value(mf6.get_var_address("AMAT", "SLN_1"))
    IA = mf6.get_value(mf6.get_var_address("IA", "SLN_1"))
    JA = mf6.get_value(mf6.get_var_address("JA", "SLN_1"))

    # Arreglo para almacenar la matriz en formato completo.
    Atest = np.zeros((NROW[0], NCOL[0]))
    idx = 0
    i = 0
    istart = IA[0] # Inicio del renglón en IA
    for iend in IA[1:]: # Recorremos desde el inicio de cada renglón
        for j in range(istart, iend): # Recorremos todos los elementos del renglón
            Atest[idx, JA[j-1]-1] = A[i] # Agregamos el coeficiente en la matriz completa
            i += 1
        istart = iend
        idx += 1
    return Atest, A, IA, JA # Regresamos la matriz densa y en el format CRS

    