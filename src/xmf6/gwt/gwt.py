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

def set_packages(o_sim, silent = False, **kwargs):
    """
    Construye el objeto para el modelo de flujo agregándole los paquetes
    definidos por el usuario.
    
    Parameters:
    -----------
    o_sim: dict
        Diccionario de inicialización de la simulación.
        
    kwargs: dict
        Diccionario que contiene a su vez los diccionarios con los datos
        para inicializar y agregar cada paquete definido por el usuario.

    silent: bool
        Cuando es True se imprime toda la información.
        
    Return:
    -------
    o_gwt, packages: flopy.mf6.modflow.mfgwf.ModflowGwt, dict
        Objeto del modelo de transporte y diccionario con los paquetes agregados al modelo.
    """
    packages = {} # Diccionario de paquetes agregados
    par = xmf6.common.set_par(kwargs["gwt"], get_gwt_par, "\nnumerical model configuration", silent)
    o_gwt = flopy.mf6.ModflowGwt(
        simulation = o_sim, 
        modelname=par["modelname"], 
        model_nam_file=par["model_nam_file"], 
        version=par["version"], 
        exe_name=par["exe_name"], 
        model_rel_path=par["model_rel_path"], 
        list=par["list"], 
        print_input=par["print_input"], 
        print_flows=par["print_flows"], 
        save_flows=par["save_flows"], 
 #       packages=par["packages"],    
    )
    
    if "dis" in kwargs:
        par = xmf6.common.set_par(kwargs["dis"], get_dis_par, "\nspatial discretization configuration", silent)
        o_dis = flopy.mf6.ModflowGwtdis(
            model = o_gwt,
            loading_package=par["loading_package"], 
            length_units=par["length_units"], 
            nogrb=par["nogrb"], 
            xorigin=par["xorigin"], 
            yorigin=par["yorigin"], 
            angrot=par["angrot"], 
            export_array_ascii=par["export_array_ascii"],
            nlay=par["nlay"], 
            nrow=par["nrow"], 
            ncol=par["ncol"], 
            delr=par["delr"], 
            delc=par["delc"], 
            top=par["top"], 
            botm=par["botm"], 
            idomain=par["idomain"], 
            filename=par["filename"], 
            pname=par["pname"], 
        )
        packages["dis"] = o_dis

    if "ic" in kwargs:
        par = xmf6.common.set_par(kwargs["ic"], get_ic_par, "\ninitial conditions configuration", silent)
        o_ic = flopy.mf6.ModflowGwtic(
            model = o_gwt,
            loading_package=par["loading_package"], 
            strt=par["strt"], 
            filename=par["filename"], 
            pname=par["pname"],
            parent_file=par["parent_file"]
        )
        packages["ic"] = o_ic

    if "mst" in kwargs:
        par = xmf6.common.set_par(kwargs["mst"], get_mst_par, "\nTODO: what is MST?", silent)
        o_mst = flopy.mf6.ModflowGwtmst(
            model = o_gwt,
            loading_package=par["loading_package"], 
            save_flows=par["save_flows"], 
            first_order_decay=par["first_order_decay"],
            zero_order_decay=par["zero_order_decay"],
            sorption=par["sorption"],
            porosity=par["porosity"],
            decay=par["decay"],
            decay_sorbed=par["decay_sorbed"],
            bulk_density=par["bulk_density"],
            distcoef=par["distcoef"],
            filename=par["filename"], 
            pname=par["pname"], 
            parent_file=par["parent_file"]
        )
        packages["mst"] = o_mst

    if "adv" in kwargs:
        par = xmf6.common.set_par(kwargs["adv"], get_adv_par, "\nTODO: what is ADV?", silent)
        o_adv = flopy.mf6.ModflowGwtadv(
            model = o_gwt,
            loading_package=par["loading_package"], 
            scheme=par["scheme"],
            filename=par["filename"], 
            pname=par["pname"], 
            parent_file=par["parent_file"]
        )
        packages["adv"] = o_adv

    if "dsp" in kwargs:
        par = xmf6.common.set_par(kwargs["dsp"], get_dsp_par, "\nTODO: what is DSP?", silent)
        o_dsp = flopy.mf6.ModflowGwtdsp(
            model = o_gwt,
            loading_package=par["loading_package"], 
            xt3d_off=par["xt3d_off"], 
            xt3d_rhs=par["xt3d_rhs"], 
            diffc=par["diffc"], 
            alh=par["alh"], 
            alv=par["alv"], 
            ath1=par["ath1"], 
            ath2=par["ath2"], 
            atv=par["atv"], 
            filename=par["filename"], 
            pname=par["pname"], 
            parent_file=par["parent_file"]
        )
        packages["dsp"] = o_dsp

    if "fmi" in kwargs:
        par = xmf6.common.set_par(kwargs["fmi"], get_fmi_par, "\nTODO: what is FMI?", silent)
        o_fmi = flopy.mf6.ModflowGwtfmi(
            model = o_gwt,
            loading_package=par["loading_package"], 
            flow_imbalance_correction=par["flow_imbalance_correction"], 
            packagedata=par["packagedata"],
            filename=par["filename"], 
            pname=par["pname"], 
            parent_file=par["parent_file"]
        )
        packages["fmi"] = o_fmi       

    if "ssm" in kwargs:
        par = xmf6.common.set_par(kwargs["ssm"], get_ssm_par, "\nTODO: what is SSM?", silent)
        o_ssm = flopy.mf6.ModflowGwtssm(
            model = o_gwt,
            loading_package=par["loading_package"],             
            print_flows=par["print_flows"], 
            save_flows=par["save_flows"], 
            sources=par["sources"], 
            fileinput=par["fileinput"], 
            filename=par["filename"], 
            pname=par["pname"], 
        )
        packages["ssm"] = o_ssm 
        
    if "oc" in kwargs:
        par = xmf6.common.set_par(kwargs["oc"], get_oc_par, "\noutput configuration", silent)
        o_oc = flopy.mf6.ModflowGwtoc(
            model = o_gwt,            
            loading_package=par["loading_package"], 
            budget_filerecord=par["budget_filerecord"], 
            budgetcsv_filerecord=par["budgetcsv_filerecord"], 
            concentration_filerecord=par["concentration_filerecord"], 
            concentrationprintrecord=par["concentrationprintrecord"], 
            saverecord=par["saverecord"], 
            printrecord=par["printrecord"], 
            filename=par["filename"], 
            pname=par["pname"],
            parent_file=par["parent_file"]
        )
        packages["oc"] = o_oc

    return o_gwt, packages


def get_head(o_gwf, binary = False, **par):
    """
    Obtiene el vector de carga hidráulica.
    El archivo de donde se almacena la carga hidráulica debe
    tener extension ".hds"

    Parameters:
    -----------
    o_gwf: flopy.mf6.modflow.mfgwf.ModflowGwf
        Objeto de la simulación de flujo.

    binary: bool
        Cuando es True regresa el objeto de tipo flopy.utils.binaryfile.HeadFile,
        además del arrego de carga hidráulica. Valor por omisión: False.

    **par: dict
        Parámetros para la función get_data() del objeto flopy.utils.binaryfile.HeadFile.

    Return:
    -------
        Cuando binary = False: arreglo de carga hidráulica.
        Cuando binary = True: objeto binario de tipo HeadFile y arreglo de carga hidráulica.
    """
    headfile = os.path.join(o_gwf.model_ws, f"{o_gwf.name}.hds")
    hds = flopy.utils.HeadFile(headfile)
    
    if binary:
        return hds, hds.get_data(**par)
    else:
        return hds.get_data(**par)

def get_specific_discharge(o_gwf, binary = False, **par):
    """
    Obtiene el vector de descarga específica.
    El archivo de donde se almacena el budget debe
    tener extension ".bud"

    Parameters:
    -----------
    o_gwf: flopy.mf6.modflow.mfgwf.ModflowGwf
        Objeto de la simulación de flujo.
        
    binary: bool
        Cuando es True regresa el objeto de tipo flopy.utils.binaryfile.CellBudgetFile,
        además de los arreglos de la descarga específica. Valor por omisión: False.

    **par: dict
        Parámetros para la función get_data() del objeto flopy.utils.binaryfile.CellBudgetFile.

    Return:
    -------
        Cuando binary = False: arreglos de la descarga específica (qx, qy, qz y n_q que es la norma del vector de flujo).
        Cuando binary = True: objeto binario de tipo HeadFile y arreglos de la descarga específica (qx, qy, qz y n_q que es la norma del vector de flujo).
    """
    budfile = os.path.join(o_gwf.model_ws, f"{o_gwf.name}.bud")
    bud  = flopy.utils.CellBudgetFile(budfile)
    spdis = bud.get_data(**par)[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, o_gwf)
    n_q = np.sqrt(np.square(qx[0]) + np.square(qy[0]) + np.square(qz[0]))
    
    if binary:
        return bud, qx, qy, qz, n_q
    else:
        return qx, qy, qz, n_q


def get_gwt_par():
    return dict(modelname='model', 
                model_nam_file=None, 
                version='mf6', 
                exe_name='mf6.exe', 
                model_rel_path='.', 
                list=None, 
                print_input=None, 
                print_flows=None, 
                save_flows=None, 
                packages=None
               )    

def get_dis_par():
    return dict(loading_package=False, 
                length_units=None, 
                nogrb=None, 
                xorigin=None, 
                yorigin=None, 
                angrot=None, 
                export_array_ascii=None, 
                nlay=1, 
                nrow=2, 
                ncol=2, 
                delr=1.0, 
                delc=1.0, 
                top=1.0, 
                botm=0.0, 
                idomain=None, 
                filename=None, 
                pname=None
               )
        
def get_ic_par():
    return dict(loading_package=False, 
                strt=0.0, 
                filename=None, 
                pname=None, 
                parent_file=None
               )

def get_mst_par():
    return dict(loading_package=False, 
                save_flows=None, 
                first_order_decay=None, 
                zero_order_decay=None, 
                sorption=None, 
                porosity=None, 
                decay=None, 
                decay_sorbed=None, 
                bulk_density=None, 
                distcoef=None, 
                filename=None, 
                pname=None, 
                parent_file=None
               )

def get_adv_par():
    return dict(loading_package=False, 
                scheme=None, 
                filename=None, 
                pname=None, 
                parent_file=None
               )

def get_dsp_par():
    return dict(loading_package=False, 
                xt3d_off=None, 
                xt3d_rhs=None, 
                diffc=None, 
                alh=None, 
                alv=None, 
                ath1=None, 
                ath2=None, 
                atv=None, 
                filename=None, 
                pname=None, 
                parent_file=None
               )

def get_fmi_par():
    return dict(loading_package=False, 
                flow_imbalance_correction=None, 
                packagedata=None, 
                filename=None, 
                pname=None, 
                parent_file=None
               )

def get_ssm_par():
    return dict(loading_package=False, 
                print_flows=None, 
                save_flows=None, 
                sources=None, 
                fileinput=None, 
                filename=None, 
                pname=None
               )

def get_oc_par():
    return dict(loading_package=False, 
                budget_filerecord=None, 
                budgetcsv_filerecord=None, 
                concentration_filerecord=None, 
                concentrationprintrecord=None, 
                saverecord=None, 
                printrecord=None, 
                filename=None, 
                pname=None, 
                parent_file=None
               )
    
    
if __name__ == '__main__':
    init = {
        'sim_name' : "trans",
        'exe_name' : "C:\\Users\\luiggi\\Documents\\GitSites\\xmf6\\mf6\\windows\\mf6",
    #    'exe_name' : "../../mf6/macosarm/mf6",
        'sim_ws' : "sandbox_gwt"
    }
    
    tdis = {
        'units': "DAYS",
        'nper' : 1,
        'perioddata': [(1.0, 1, 1.0)]
    }
    
    ims = {}

    o_sim = xmf6.common.init_sim(init = init, tdis = tdis, ims = ims, silent = False)  
    print(o_sim)
    print(o_sim.ims)
    print(o_sim.tdis)

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
    o_gwt, package_list = set_packages(o_sim, silent = False,
                                       gwt = gwt)#, dis = dis)#, ic = ic, oc = oc)

    print(o_gwt)
    print(o_gwt.get_package_list())
    print(package_list.keys())