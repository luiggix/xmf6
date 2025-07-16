import numpy as np
import matplotlib.pyplot as plt
import os, sys
import flopy
from modflowapi import ModflowApi
import xmf6

def build_mat(mf6):
    """
    Construye la matriz del sistema.

    Parameters
    ----------
    mf6: ModflowApi
        Objeto para accedar a toda la funcionalidad de la API.
    """
    # Obtiene el número de renglones y columnas del sistema
    NCOL = mf6.get_value(mf6.get_var_address("NCOL", 'SLN_1'))
    NROW = mf6.get_value(mf6.get_var_address("NROW", 'SLN_1'))

    # Obtiene los coeficientes de la matriz en formato CRS (Compressed Row Storage)
    # A: Coeficientes, JA: índices de la columna, IA: índice de inicio del renglón en JA.
    A = mf6.get_value(mf6.get_var_address("AMAT", 'SLN_1'))
    IA = mf6.get_value(mf6.get_var_address("IA", 'SLN_1'))
    JA = mf6.get_value(mf6.get_var_address("JA", 'SLN_1'))

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

# --- Preparación de la simulación ---

# --- Componentes ---

# Parámetros de la simulación (flopy.mf6.MFSimulation)
init = {
    'sim_name' : "flow",
    'exe_name' : "C:\\Users\\luiggi\\Documents\\GitSites\\xmf6\\mf6\\windows\\mf6",
#    'exe_name' : "../../mf6/macosarm/mf6",
    'sim_ws' : "sandbox4"
}

# Parámetros para el tiempo (flopy.mf6.ModflowTdis)
tdis = {
    'units': "DAYS",
    'nper' : 3,
    'perioddata': [(1.0, 5, 1.0), (2.0, 3, 1.2), (3.0, 4, 1.1)] #[(1.0, 1, 1.0)]
}

# Parámetros para la solución numérica (flopy.mf6.ModflowIms)
ims = {}

# Parámetros para el modelo de flujo (flopy.mf6.ModflowGwf)
gwf = { 
    'modelname': init["sim_name"],
    'model_nam_file': f"{init["sim_name"]}.nam",
    'save_flows': True
}

# --- Paquetes del modelo de flujo ---
lx = 25
ly = 25
nrow = int(input("nrow = "))
ncol = int(input("ncol = "))
if nrow > 10 or nrow < 0 or ncol > 10 or ncol < 0:
    print("Este ejemplo está limitado a:")
    print(" 0 < nrow =< 10")
    print(" 0 < ncol =< 10")
    sys.exit()
delr = lx / ncol
delc = ly / nrow 

# Parámetros para la discretización espacial (flopy.mf6.ModflowGwfdis)
dis = {
    'length_units': "meters",
    'nlay': 1, 
    'nrow': nrow, 
    'ncol': ncol,
    'delr': delr, 
    'delc': delc, 
    'top' : 1.0, 
    'botm': 0.0 
}

# Parámetros para las condiciones iniciales (flopy.mf6.ModflowGwfic)
ic = {
    'strt': 1.0
}

# Parámetros para las condiciones de frontera (flopy.mf6.ModflowGwfchd)
chd_data = []
for row in range(dis['nrow']):
    chd_data.append([(0, row, 0), 10.0])       # Condición en la pared izquierda
    chd_data.append([(0, row, dis['ncol'] - 1), 5.0]) # Condición en la pared derecha

chd = {
    'stress_period_data': chd_data,     
}

# Parámetros para las propiedades de flujo (flopy.mf6.ModflowGwfnpf)
npf = {
    'save_specific_discharge': True,
    'save_saturation' : True,
    'icelltype' : 0,
    'k' : 0.01,
}

# Parámetros para almacenar y mostrar la salida de la simulación (flopy.mf6.ModflowGwfoc)
oc = {
    'budget_filerecord': f"{init['sim_name']}.bud",
    'head_filerecord': f"{init['sim_name']}.hds",
    'saverecord': [("HEAD", "ALL"), ("BUDGET", "ALL")],
    'printrecord': [("HEAD", "ALL")]
}

# Inicialización de la simulación
o_sim = xmf6.gwf.init_sim(init = init, tdis = tdis, ims = ims, silent = True)

# Configuración de los paquetes para el modelo de flujo
o_gwf, package_list = xmf6.gwf.set_packages(o_sim, silent = True,
                                            gwf = gwf, dis = dis, ic = ic, chd = chd, npf = npf, oc = oc)

# Escritura de los archivos de entrada
o_sim.write_simulation(silent = False)

# Ejecución de la simulación
o_sim.run_simulation()

# --- Recuperamos los resultados de la simulación ---
head = xmf6.gwf.get_head(o_gwf)

headfile = os.path.join(o_gwf.model_ws, f"{o_gwf.name}.hds")
hds = flopy.utils.HeadFile(headfile)
budfile = os.path.join(o_gwf.model_ws, f"{o_gwf.name}.bud")
bud  = flopy.utils.CellBudgetFile(budfile)

head_x = o_gwf.output.head()

print(type(head), type(head_x), type(hds), type(bud))

# Obtener tiempos disponibles
times = head_x.get_times()
print("Tiempos disponibles:")
[print(t) for t in times]

otimes = bud.get_times()
print("Tiempos disponibles:")
[print(t) for t in otimes]

# Obtener número de periodos y pasos
nper = len(times)
print(nper)
