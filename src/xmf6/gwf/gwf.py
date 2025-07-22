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
    o_gwf, packages: flopy.mf6.modflow.mfgwf.ModflowGwf, dict
        Objeto del modelo de flujo y diccionario con los paquetes agregados al modelo.
    """
    packages = {} # Diccionario de paquetes agregados
    par = xmf6.common.set_par(kwargs["gwf"], get_gwf_par, "\nnumerical model configuration", silent)
    o_gwf = flopy.mf6.ModflowGwf(
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
        newtonoptions=par["newtonoptions"], 
    )
    
    if "dis" in kwargs:
        par = set_par(kwargs["dis"], get_dis_par, "\nspatial discretization configuration", silent)
        o_dis = flopy.mf6.ModflowGwfdis(
            model = o_gwf,
            loading_package=par["loading_package"], 
            length_units=par["length_units"], 
            nogrb=par["nogrb"], 
            xorigin=par["xorigin"], 
            yorigin=par["yorigin"], 
            angrot=par["angrot"], 
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
            parent_file=par["parent_file"]
        )
        packages["dis"] = o_dis

    if "ic" in kwargs:
        par = set_par(kwargs["ic"], get_ic_par, "\ninitial conditions configuration", silent)
        o_ic = flopy.mf6.ModflowGwfic(
            model = o_gwf,
            loading_package=par["loading_package"], 
            export_array_ascii=par["export_array_ascii"], 
            export_array_netcdf=par["export_array_netcdf"], 
            strt=par["strt"], 
            filename=par["filename"], 
            pname=par["pname"]
        )
        packages["ic"] = o_ic

    if "chd" in kwargs:
        par = set_par(kwargs["chd"], get_chd_par, "\nboundary conditions configuration", silent)
        o_chd = flopy.mf6.ModflowGwfchd(
            model = o_gwf,
            loading_package=par["loading_package"], 
            auxiliary=par["auxiliary"], 
            auxmultname=par["auxmultname"], 
            boundnames=par["boundnames"], 
            print_input=par["print_input"], 
            print_flows=par["print_flows"], 
            save_flows=par["save_flows"], 
            timeseries=par["timeseries"], 
            observations=par["observations"], 
            maxbound=par["maxbound"], 
            stress_period_data=par["stress_period_data"], 
            filename=par["filename"], 
            pname=par["pname"], 
            parent_file=par["parent_file"]
        )
        packages["chd"] = o_chd

    if "npf" in kwargs:
        par = set_par(kwargs["npf"], get_npf_par, "\nflow properties configuration", silent)
        o_npf = flopy.mf6.ModflowGwfnpf(
            model = o_gwf,
            loading_package=par["loading_package"], 
            save_flows=par["save_flows"], 
            alternative_cell_averaging=par["alternative_cell_averaging"], 
            thickstrt=par["thickstrt"], 
            cvoptions=par["cvoptions"], 
            perched=par["perched"], 
            rewet_record=par["rewet_record"], 
            xt3doptions=par["xt3doptions"], 
            save_specific_discharge=par["save_specific_discharge"], 
            save_saturation=par["save_saturation"], 
            k22overk=par["k22overk"], 
            k33overk=par["k33overk"], 
            icelltype=par["icelltype"], 
            k=par["k"], 
            k22=par["k22"], 
            k33=par["k33"], 
            angle1=par["angle1"], 
            angle2=par["angle2"], 
            angle3=par["angle3"], 
            wetdry=par["wetdry"], 
            filename=par["filename"], 
            pname=par["pname"], 
            parent_file=par["parent_file"]
        )
        packages["npf"] = o_npf

    if "oc" in kwargs:
        par = set_par(kwargs["oc"], get_oc_par, "\noutput configuration", silent)
        o_oc = flopy.mf6.ModflowGwfoc(
            model = o_gwf,
            loading_package=par["loading_package"], 
            budget_filerecord=par["budget_filerecord"], 
            budgetcsv_filerecord=par["budgetcsv_filerecord"], 
            head_filerecord=par["head_filerecord"], 
            headprintrecord=par["headprintrecord"], 
            saverecord=par["saverecord"], 
            printrecord=par["printrecord"], 
            filename=par["filename"], 
            pname=par["pname"]
        )
        packages["oc"] = o_oc

    if "well" in kwargs:
        par = set_par(kwargs["well"], get_well_par, "\nwells configuration", silent)
        o_well = flopy.mf6.ModflowGwfwel(
            model = o_gwf,
            loading_package=par["loading_package"], 
            auxiliary=par["auxiliary"], 
            auxmultname=par["auxmultname"], 
            boundnames=par["boundnames"], 
            print_input=par["print_input"], 
            print_flows=par["print_flows"], 
            save_flows=par["save_flows"], 
            auto_flow_reduce=par["auto_flow_reduce"], 
            timeseries=par["timeseries"], 
            observations=par["observations"], 
            mover=par["mover"], 
            maxbound=par["maxbound"], 
            stress_period_data=par["stress_period_data"], 
            filename=par["filename"], 
            pname=par["pname"], 
            parent_file=par["parent_file"]
        )
        packages["well"] = o_well

    return o_gwf, packages


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
    
def get_gwf_par():
    return dict(modelname='model', 
                model_nam_file=None, 
                version='mf6', 
                exe_name='mf6.exe', 
                model_rel_path='.', 
                list=None, 
                print_input=None, 
                print_flows=None, 
                save_flows=None, 
                newtonoptions=None, 
                packages=None
               )    


def get_dis_par():
    return dict(loading_package=False, 
                length_units=None, 
                nogrb=None, 
                xorigin=None, 
                yorigin=None, 
                angrot=None, 
                nlay=1, 
                nrow=2, 
                ncol=2, 
                delr=1.0, 
                delc=1.0, 
                top=1.0, 
                botm=0.0, 
                idomain=None, 
                filename=None, 
                pname=None, 
                parent_file=None
               )
        
def get_ic_par():
    return dict(loading_package=False, 
                export_array_ascii=None, 
                export_array_netcdf=None, 
                strt=1.0, 
                filename=None, 
                pname=None
               )

def get_chd_par():
    return dict(loading_package=False, 
                auxiliary=None, 
                auxmultname=None, 
                boundnames=None, 
                print_input=None, 
                print_flows=None, 
                save_flows=None, 
                timeseries=None, 
                observations=None, 
                maxbound=None, 
                stress_period_data=None, 
                filename=None, 
                pname=None, 
                parent_file=None
               )

def get_npf_par():
    return dict(loading_package=False, 
                save_flows=None, 
                alternative_cell_averaging=None, 
                thickstrt=None, 
                cvoptions=None, 
                perched=None, 
                rewet_record=None, 
                xt3doptions=None, 
                save_specific_discharge=None, 
                save_saturation=None, 
                k22overk=None, 
                k33overk=None, 
                icelltype=0, 
                k=1.0, 
                k22=None, 
                k33=None, 
                angle1=None, 
                angle2=None, 
                angle3=None, 
                wetdry=None, 
                filename=None, 
                pname=None, 
                parent_file=None
               )

def get_oc_par():
    return dict(loading_package=False, 
                budget_filerecord=None, 
                budgetcsv_filerecord=None, 
                head_filerecord=None, 
                headprintrecord=None, 
                saverecord=None, 
                printrecord=None, 
                filename=None, 
                pname=None
               )

def get_well_par():
    return dict(loading_package=False, 
                auxiliary=None, 
                auxmultname=None, 
                boundnames=None, 
                print_input=None, 
                print_flows=None, 
                save_flows=None, 
                auto_flow_reduce=None, 
                timeseries=None, 
                observations=None, 
                mover=None, 
                maxbound=None, 
                stress_period_data=None, 
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

    o_sim = xmf6.common.init_sim(init = init, tdis = tdis, ims = ims, silent = False)   
    print(o_sim.ims)
    print(o_sim.tdis)

    gwf = { 
        'modelname': init["sim_name"],
        'model_nam_file': f"{init["sim_name"]}.nam",
        'save_flows': True
    }
    
    dis = {
    'length_units': "meters",
    'nlay': 3, 
    'nrow': 3, 
    'ncol': 3,
    'delr': 1.0, 
    'delc': 1.0, 
    'top' : 1.0, 
    'botm': 0.0 
    }

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

    # Configuración de los paquetes para el modelo de flujo
    o_gwf, package_list = xmf6.gwf.set_packages(o_sim, silent = True,
                                            gwf = gwf, dis = dis, ic = ic, chd = chd, npf = npf, oc = oc)

    print(o_gwf)
    print(o_gwf.get_package_list())
    print(package_list.keys())