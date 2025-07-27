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

def init_sim(init, tdis, ims, silent = False):
    """
    Iniciliza la simulación con las componentes del tiempo y la solución numérica.
    
    Parameters:
    -----------
    init: dict
        Diccionario de inicialización de la simulación.
        
    tdis: dict
        Diccionario con los datos del tiempo.
        
    ims: dict
        Diccionario con los datos para la solución numérica.

    silent: bool
        Cuando es True se imprime toda la información.
        
    Return:
    -------
    o_sim: flopy.mf6.modflow.mfsimulation.MFSimulation
        Objeto para controlar la simulación.
    """
    par = set_par(init, get_sim_par, "\nsim configuration", silent)
    o_sim = flopy.mf6.MFSimulation(
        sim_name = par["sim_name"],
        version  = par["version"],
        exe_name = par["exe_name"],
        sim_ws   = par["sim_ws"],
        verbosity_level = par["verbosity_level"],
        continue_ = par["continue_"],
        nocheck = par["nocheck"],
        memory_print_option = par["memory_print_option"],
        write_headers = par["write_headers"]
    )

    par = set_par(tdis, get_tdis_par, "\ntime configuration", silent)     
    o_tdis = flopy.mf6.ModflowTdis(
        simulation = o_sim,  
        loading_package = par["loading_package"], 
        time_units = par["time_units"], 
        start_date_time = par["start_date_time"], 
        ats_perioddata = par["ats_perioddata"], 
        nper = par["nper"], 
        perioddata = par["perioddata"], 
        filename = par["filename"], 
        pname = par["pname"]
    )

    par = set_par(ims, get_ims_par, "\nnumerical solution configuration", silent)
    o_ims = flopy.mf6.ModflowIms(
        simulation = o_sim, 
        loading_package=par["loading_package"], 
        print_option=par["print_option"], 
        complexity=par["complexity"], 
        csv_output_filerecord=par["csv_output_filerecord"], 
        csv_outer_output_filerecord=par["csv_outer_output_filerecord"], 
        csv_inner_output_filerecord=par["csv_inner_output_filerecord"], 
        no_ptcrecord=par["no_ptcrecord"], 
        outer_hclose=par["outer_hclose"], 
        outer_dvclose=par["outer_dvclose"], 
        outer_rclosebnd=par["outer_rclosebnd"], 
        outer_maximum=par["outer_maximum"], 
        under_relaxation=par["under_relaxation"], 
        under_relaxation_gamma=par["under_relaxation_gamma"], 
        under_relaxation_theta=par["under_relaxation_theta"], 
        under_relaxation_kappa=par["under_relaxation_kappa"], 
        under_relaxation_momentum=par["under_relaxation_momentum"], 
        backtracking_number=par["backtracking_number"], 
        backtracking_tolerance=par["backtracking_tolerance"], 
        backtracking_reduction_factor=par["backtracking_reduction_factor"], 
        backtracking_residual_limit=par["backtracking_residual_limit"], 
        inner_maximum=par["inner_maximum"], 
        inner_hclose=par["inner_hclose"], 
        inner_dvclose=par["inner_dvclose"], 
        rcloserecord=par["rcloserecord"], 
        linear_acceleration=par["linear_acceleration"], 
        relaxation_factor=par["relaxation_factor"], 
        preconditioner_levels=par["preconditioner_levels"], 
        preconditioner_drop_tolerance=par["preconditioner_drop_tolerance"], 
        number_orthogonalizations=par["number_orthogonalizations"], 
        scaling_method=par["scaling_method"], 
        reordering_method=par["reordering_method"], 
        filename=par["filename"], 
        pname=par["pname"], 
        parent_file=par["parent_file"]
    )

    return o_sim
    
def set_par(key_par, function, message, silent = False):
    """
    Parameters:
    -----------
    key_par: dict
        Diccionario con los parámetros que se van a definir.
        
    function: function
        Función que obtiene  los parámetros.
        
    message: str
        Mensaje a imprimir.

    silent: bool
        Cuando es True se imprime información.
        
    Return:
    -------
    par: dict
        Diccionario con los parámetros para la clave solicitada.
    """
    par = function()
    for k, v in key_par.items():
        par[k] = v  

    if not silent:
        xmf6.nice_print(key_par, message)
    return par

def get_sim_par():
    return dict(sim_name='sim', 
                version='mf6', 
                exe_name='mf6.exe', 
                sim_ws='.', 
                verbosity_level=1, 
                continue_=None, 
                nocheck=None, 
                memory_print_option=None, 
                write_headers=True
               )

def get_tdis_par():
    return dict(loading_package=False, 
                time_units=None, 
                start_date_time=None, 
                ats_perioddata=None, 
                nper=1, 
                perioddata=[[1.0, 1, 1.0]], 
                filename=None, 
                pname=None
               )

def get_ims_par():
    return dict(loading_package=False, 
                 print_option=None, 
                 complexity=None, 
                 csv_output_filerecord=None, 
                 csv_outer_output_filerecord=None, 
                 csv_inner_output_filerecord=None, 
                 no_ptcrecord=None, 
                 outer_hclose=None, 
                 outer_dvclose=None, 
                 outer_rclosebnd=None, 
                 outer_maximum=None, 
                 under_relaxation=None, 
                 under_relaxation_gamma=None, 
                 under_relaxation_theta=None, 
                 under_relaxation_kappa=None, 
                 under_relaxation_momentum=None, 
                 backtracking_number=None, 
                 backtracking_tolerance=None, 
                 backtracking_reduction_factor=None, 
                 backtracking_residual_limit=None, 
                 inner_maximum=None, 
                 inner_hclose=None, 
                 inner_dvclose=None, 
                 rcloserecord=None, 
                 linear_acceleration=None, 
                 relaxation_factor=None, 
                 preconditioner_levels=None, 
                 preconditioner_drop_tolerance=None, 
                 number_orthogonalizations=None, 
                 scaling_method=None, 
                 reordering_method=None, 
                 filename=None, 
                 pname=None, 
                 parent_file=None
               )


def set_obs(model, obs, silent = False):
    """
    Iniciliza los puntos de observación de la simulación.
    
    Parameters:
    -----------
    obs: dict
        Diccionario de inicialización de la simulación.

    silent: bool
        Cuando es True se imprime toda la información.
        
    Return:
    -------
    o_obs: flopy.mf6.ModflowUtlobs
        Objeto para definir los puntos de observación.
    """
    par = set_par(obs, get_obs_par, "\nOBS configuration", silent)
    
    o_obs = flopy.mf6.ModflowUtlobs(
        model,
        loading_package=par["loading_package"], 
        digits=par["digits"], 
        print_input=par["print_input"], 
        continuous=par["continuous"], 
        filename=par["filename"], 
        pname=par["pname"], 
        parent_file=par["parent_file"]
    )
    return o_obs
    
def get_obs_par():
    return dict(loading_package=False, 
                digits=None, 
                print_input=None, 
                continuous=None, 
                filename=None, 
                pname=None, 
                parent_file=None
               )


if __name__ == '__main__':
    init = {
        'sim_name' : "flow",
        'exe_name' : "C:\\Users\\luiggi\\Documents\\GitSites\\xmf6\\mf6\\windows\\mf6",
    #    'exe_name' : "../../mf6/macosarm/mf6",
        'sim_ws' : "sandbox4"
    }
    
    tdis = {
        'units': "DAYS",
        'nper' : 1,
        'perioddata': [(1.0, 1, 1.0)]
    }
    
    ims = {}

    o_sim = init_sim(init = init, tdis = tdis, ims = ims, silent = True)   
#    print(o_sim.ims)
#    print(o_sim.tdis)


    gwt = { 
        'modelname': init["sim_name"],
        'save_flows': True
    }
    
    # Parámetros para la discretización espacial (flopy.mf6.ModflowGwfdis)
    dis = {
        'length_units' : "centimeters",
        'nlay': 1, 
        'nrow': 1, 
        'ncol': 120,
        'delr': 0.1, 
        'delc': 0.1, 
        'top' : 1.0, 
        'botm': 0.0 
    }

    ic = {
        'strt': 0.0
    }


    # Parámetros para almacenar y mostrar la salida de la simulación (flopy.mf6.ModflowGwtoc)
    oc = {
        'budget_filerecord': f"{init['sim_name']}.bud",
        'head_filerecord': f"{init['sim_name']}.hds",
        'saverecord': [("HEAD", "ALL"), ("BUDGET", "ALL")],
        'printrecord': [("HEAD", "ALL")]
    }


    
    # Configuración de los paquetes para el modelo de flujo
    o_gwt, package_list = xmf6.gwt.set_packages(o_sim, silent = True,
                                       gwt = gwt)#, dis = dis)#, ic = ic, oc = oc)

#    print(o_gwt)
#    print(o_gwt.get_package_list())
#    print(package_list.keys())


    obs = {
    "digits" : 10, 
    "print_input" : True, 
    "continuous" : {
        "transporte.obs.csv": [
            ("X005", "CONCENTRATION", (0, 0, 0)),
            ("X405", "CONCENTRATION", (0, 0, 40)),
            ("X1105", "CONCENTRATION", (0, 0, 110)),
        ],
    }  
    }

    o_obs = set_obs(o_gwt, obs, silent = False)