"""
@author: Luis M. de la Cruz [Updated on Wed May 14 18:00:33 CST 2025].
"""

import os
import numpy as np
import flopy

def initialize(silent = False, **kwargs):
    par = set_par(kwargs["init"], get_sim_par, "\nsim configuration:", silent)
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

    par = set_par(kwargs["time"], get_time_par, "\ntime configuration:", silent)     
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

    par = set_par(kwargs["ims"], get_ims_par, "\nnumerical solution configuration:", silent)
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

def build(o_sim, silent = False, **kwargs):

    par = set_par(kwargs["gwf"], get_gwf_par, "\nnumerical model configuration:", silent)
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
#        packages=par["packages"]
    )
    
    if "dis" in kwargs:
        par = set_par(kwargs["dis"], get_dis_par, "\nspatial discretization configuration:", silent)
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

    if "ic" in kwargs:
        par = set_par(kwargs["ic"], get_ic_par, "\ninitial conditions configuration:", silent)
        o_ic = flopy.mf6.ModflowGwfic(
            model = o_gwf,
            loading_package=par["loading_package"], 
            export_array_ascii=par["export_array_ascii"], 
            export_array_netcdf=par["export_array_netcdf"], 
            strt=par["strt"], 
            filename=par["filename"], 
            pname=par["pname"]
        )

    if "chd" in kwargs:
        par = set_par(kwargs["chd"], get_chd_par, "\nboundary conditions configuration:", silent)
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

    if "npf" in kwargs:
        par = set_par(kwargs["npf"], get_npf_par, "\nflow properties configuration:", silent)
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

    if "oc" in kwargs:
        par = set_par(kwargs["oc"], get_oc_par, "\noutput configuration:", silent)
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

    if "well" in kwargs:
        par = set_par(kwargs["well"], get_well_par, "\nwells configuration:", silent)
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

    return o_gwf


def get_head(o_sim, o_gwf):
    headfile = os.path.join(f"{o_sim.sim_path}", f"{o_gwf.name}.hds")
    hds = flopy.utils.HeadFile(headfile)
    return hds.get_data()

def get_specific_discharge(o_sim, o_gwf):
    budfile = os.path.join(f"{o_sim.sim_path}", f"{o_gwf.name}.bud")
    bud  = flopy.utils.CellBudgetFile(budfile)
    spdis = bud.get_data(text="DATA-SPDIS")[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, o_gwf)
    n_q = np.sqrt(np.square(qx[0]) + np.square(qy[0]))
    return qx, qy, qz, n_q
    
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
    if not silent: print(f"\n{message}:")
    par = function()
    for k, v in key_par.items():
        if not silent: print(f"  {k} = {v}")
        par[k] = v  
    if not silent: print(f"---\n")
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

def get_time_par():
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