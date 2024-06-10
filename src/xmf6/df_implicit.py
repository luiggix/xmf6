import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix 
from scipy.sparse import csr_matrix 
from scipy.sparse.linalg import spsolve  
from time import time  

def solve(ph_par, mesh, tdis, qx, verb = 1):
    sdis = ph_par["specific_discharge"]   # Specific discharge ($cm s^{-1}$
    h    = ph_par["hydraulic_conductivity"]  # Hydraulic conductivity ($cm s^{-1}$)
    cs   = ph_par["source_concentration"]   #dimensionless
    c0   = ph_par["initial_concentration"]  #dimensionless
    Por  = ph_par["porosity"]
    Dl   = ph_par["longitudinal_dispersivity"]
    R    = ph_par["retardation_factor"]
    Dr   = ph_par["decay_rate"]
    Dh   = ph_par["dispersion_coefficient"]
    
    Lx = mesh.row_length
    Ly = mesh.col_length
    Lz = mesh.lay_length
    nx = mesh.ncol
    ny = mesh.nrow
    nz = mesh.nlay
    delta_x = mesh.delr
    delta_y = mesh.delc
    delta_z = mesh.delz
    xi, _, _ = mesh.get_coords() 
    
    total_time = tdis.total_time()   #Total time of simulation
    nper = tdis.nper()            # Number of periods
    perlen, nstp, tsmult = tdis.perioddata(0) # PERLEN, NSTP, TSMULT
    dt = tdis.dt(0)   #delta of time    

    #------------------------------------------------------------------------------------------#
    #---------- Calculando los valores de q para cada nodo y fronteras de la celdas ----------------#
    #------------------------------------------------------------------------------------------#
    
    q_frnts = np.zeros(nx+1)                       #Vector de q en las fronteras (puede tomarse un esquema upwind o TVD, etc 
    q_i = np.zeros(nx)                             #Vector de q en los nodos o centros "i" 
    q_i= qx[0][0][:]/Por    #para tomar el esquema tradicional por diferencias finitas


    #------------------------------------------------------------------------------------------#
    #------ Definiendo vectores y matriz  para el sistema de eq lineales ---------------------#
    #------------------------------------------------------------------------------------------#
    
    TE = np.zeros(nx)                          #Vector de transmisibilidad en i+1/2
    TW = np.zeros(nx)                          #Vector de transmisibilidad en -1/2
    TC = np.zeros(nx)                          #Vector central C
    B = np.zeros(nx)                           #Vector de valores conocidos, lado derecho 
    mtrz_coef = np.zeros((nx, nx))          #Matriz de coeficientes para la solución del sistema de ecuaciones
    c_ini = np.zeros(nx)                       #Vector de condiciones iniciales de c
    c_n = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
    c_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
    solcnsImp_c = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)

    #------------------------------------------------------------------------------------------#
    #---------------------------- Definiendo condiciones iniciales ----------------------------#
    #------------------------------------------------------------------------------------------#
    c_n[:] = c0
    solcnsImp_c[0][:] = c0  

    if verb != 0:
        print("|------------------------------------------------------------------------------|")
        print("|--------------SOLUCIÓN DIFERENCIAS FINITAS TRADICIONALES IMPLÍCITO------------|")
        print("|------------------------------------------------------------------------------|")
    
    #------------------------------------------------------------------------------------------#
    #------------------------ Inicio del ciclo solución en el tiempo --------------------------#
    #------------------------------------------------------------------------------------------#
    tiempoComputo_inicial = time()       
    
    for ti in range(0, nstp):
            
        #--------------------------------------------------------------------------------------#
        #----------------- Calculando transmisibilidades en i, i+1/2 e i-1/2 ------------------#
        #--------------------------------------------------------------------------------------# 
        
        for i in range(0, nx):                                                               #Se calculó la transmisibilidad en todos los nodos ya que se considera el nodo fantasma para las condiciones de frontera, entonces, el nodo en nx+1/2 se utilizará en B
            TE[i] = q_i[i]/(2*delta_x)-Dh/delta_x**2                             #Transmisibilidad en xi+1/2
            TW[i] =-q_i[i]/(2*delta_x)-Dh/delta_x**2                              #Transmisibilidad en xi-1/2    
            TC[i] = 1/dt+(2*Dh)/delta_x**2
            
            # Condicion a la frontera
     
            if i==nx-1:
                TC[i]=TC[i]+TE[i]  #"No flujo" o Neumann 
                
            if i == 0:
                # B[i] =  ((teta/dt)*(c_n[i]))+(r*cr)-2*TW[i]*c_inf              #Condición de frontera izquierda Dirichlet (primera clase)
                B[i] =  c_n[i]/dt-TW[i]*cs
            else:
                B[i] =  c_n[i]/dt
            
        #--------------------------------------------------------------------------------------#
        #------------------------ Llenando la matriz de coeficientes ------------------------#
        #--------------------------------------------------------------------------------------# 
        
        for i in range(0, nx):
            mtrz_coef[i][i] = TC[i]
            
            if i < nx-1:    
                mtrz_coef[i][i+1] = TE[i]
            
            if i > 0:
                mtrz_coef[i][i-1] = TW[i]
        
        #--------------------------------------------------------------------------------------#
        #------------------------ Resolviendo el sistema de ecuaciones ------------------------#
        #--------------------------------------------------------------------------------------#
        
        am = lil_matrix(mtrz_coef)                                                              #Función que pasa de una matriz densa a una matriz tipo lista
        am = am.tocsr()
        solcn_cs = spsolve(am, B)                                                               #Resolviendo el sistema de ecuaciones
        
        for i in range (0, nx):
            c_v[i] = solcn_cs[i]                                                                #Guardando las soluciones en el vector c_v
            
        solcnsImp_c[ti+1][:] = c_v[:]                                                              #Guardando las soluciones en la matriz de c para cada paso de tiempo
        c_n[:] = c_v[:]                                                                         #Guardando el valor de c al tiempo n que se utilizará en el siguiente paso de tiempo

        if verb == 2:
            print("tiempo de simulacion: " +str(round(dt*(ti+1),2))+" segundos" )
        elif verb == 1:
            print("{:>5d}".format(ti), end=(''))
        else:
            print("".format(ti), end=(''))
    
    #------------------------------------------------------------------------------------------#
    #---------------------- Tiempo de ejecución del modelo de simulación ----------------------#
    #------------------------------------------------------------------------------------------# 
    
    tiempoComputo_final=time()                                                                         #Toma el valor de tiempo final de ejecución, para  calcuar el tiempo total de la simulación
    tiempo_ejec_seg=(tiempoComputo_final-tiempoComputo_inicial)                                               #Calcula el tiempo total de simulación, pasado a segundos
    tiempo_ejec_min=(tiempoComputo_final-tiempoComputo_inicial)/60                                            #Calcula el tiempo total de simulación, pasado a minutos

    if verb != 0:
        print("\nTiempo de Ejecución:",tiempo_ejec_seg, "[Segundos]","\t", tiempo_ejec_min, "[Minutos]\n")
        print("|------------------------------------------------------------------------------|")
        print("|------------------ FIN DE LA SIMULACIÓN DIFF TRAD IMP  -------------------------|")
        print("|------------------------------------------------------------------------------|")

    return solcnsImp_c


def plot(xi, solcnsImp_c, obs_data):
    #------------------------------------------------------------------------------------------#
    #--------------------------- Graficando todos los resultados ------------------------------#
    #------------------------------------------------------------------------------------------#
    plt.figure('DiffTrad_Imp')
    plt.style.use('fast')
    plt.minorticks_on()
    plt.title('Perfil de c vs distancia en el tiempo')
    for i in obs_data:
        plt.plot(xi, solcnsImp_c[i][:]) 
    # plt.legend(loc=4)
    plt.xlabel("Distancia x [$cm$]")
    plt.ylabel("Concentraciones [$adim$]")                                                                        
    plt.grid(True,which='major', color='w', linestyle='-')
    plt.grid(True, which='minor', color='w', linestyle='--')
    plt.show()


