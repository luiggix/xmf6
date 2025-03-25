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
    Dh   = ph_par["dispersion_coefficient"]  #!!!! Ya viene dividido por el factor de retardo R
  
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
    q_i = np.zeros(nx)                            #Vector de q en los nodos o centros "i" 
    q_i= qx[0][0][:]            #para tomar descargas especificas del esquema tradicional por diferencias finitas
    v_i= qx[0][0][:]/(Por*R)    #para tomar velocidades del esquema tradicional por diferencias finitas


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
            TE[i] = v_i[i]/(2*delta_x)-Dh/delta_x**2                             #Transmisibilidad en xi+1/2
            TW[i] =-v_i[i]/(2*delta_x)-Dh/delta_x**2                              #Transmisibilidad en xi-1/2    
            TC[i] = 1/dt+(2*Dh)/delta_x**2+Dr
            
            if i==0:  # VCs=vC-dC/dx  CF Robbin 

                v_inf=v_i[0]
                Cte_rhs=v_inf*cs/(Dh/delta_x)   #Parte del rhs
                Cte_coeff=(Dh/delta_x-(v_i[0]))/(Dh/delta_x)   #Parte del sistema de ecuaciones lineales
                TC[i]=TC[i]+Cte_coeff*TW[i]  
            # Condicion a la frontera
     
            if i==nx-1:
                TC[i]=TC[i]+TE[i]  #"Gradiente de concentracion 0" dC/dx=0 
                
            if i == 0:
                B[i] =  c_n[i]/dt-Cte_rhs*TW[i]
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


def upwind_1D(i, h, v):
    
    if(h[i]<=h[i+1]):
        vx_e=v[i+1]
    else:
        vx_e=v[i]
        
    if(h[i-1]<=h[i]):
        vx_w=v[i]
    else:
        vx_w=v[i-1]

    return (vx_e, vx_w)



def solveUpwind(ph_par, mesh, tdis, h, qx, verb = 1):
    sdis = ph_par["specific_discharge"]   # Specific discharge ($cm s^{-1}$
    h    = ph_par["hydraulic_conductivity"]  # Hydraulic conductivity ($cm s^{-1}$)
    cs   = ph_par["source_concentration"]   #dimensionless
    c0   = ph_par["initial_concentration"]  #dimensionless
    Por  = ph_par["porosity"]
    Dl   = ph_par["longitudinal_dispersivity"]
    R    = ph_par["retardation_factor"]
    Dr   = ph_par["decay_rate"]
    Dh   = ph_par["dispersion_coefficient"]  #!!!! Ya viene dividido por el factor de retardo R
  
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
    
    q_i = np.zeros(nx)                            #Vector de q en los nodos o centros "i" 
    q_i= qx[0][0][:]            #para tomar descargas especificas del esquema tradicional por diferencias finitas
    v_i= qx[0][0][:]/(Por*R)    #para tomar velocidades del esquema tradicional por diferencias finitas
    h_i= h[0][0][:]
    v_frnts = np.zeros(nx+1)                       #Vector de velocidades en las fronteras (puede tomarse un esquema upwind o TVD, etc 

    v_inf=v_i[0]  ################ ATENCION ESTE VALOR TIENE QUE ESTAR DADO Y VENIR EN EL DICCIONARIO DE PARAMETROS ################
    v_frnts[0] = v_inf   #Valor de la primera frontera de la celda 
    v_frnts[nx] = v_i[nx-1]  #valor en la ultima frontera
    
    for i in range(1, nx-1):
         #Upwind tradicional
        if (h_[i]<=h[i+1]):
            v_frnts[i] = v[i+1]          #Valor de v en las demás fronteras en i-1/2, i+1/2 
        else:
            v_frnts[i] = v[i]

    
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
        
        for i in range(0, nx):       #Se calculó la transmisibilidad en todos los nodos ya que se considera el nodo fantasma para las condiciones de frontera, entonces, el nodo en nx+1/2 se utilizará en B
            TE[i] = v_i[i]/(2*delta_x)-Dh/delta_x**2                             #Transmisibilidad en xi+1/2
            TW[i] =-v_i[i]/(2*delta_x)-Dh/delta_x**2                              #Transmisibilidad en xi-1/2    
            TC[i] = 1/dt+(2*Dh)/delta_x**2+Dr
            
            if i==0:  # VCs=vC-dC/dx  CF Robbin 

                v_inf=v_i[0]
                Cte_rhs=v_inf*cs/(Dh/delta_x)   #Parte del rhs
                Cte_coeff=(Dh/delta_x-(v_i[0]))/(Dh/delta_x)   #Parte del sistema de ecuaciones lineales
                TC[i]=TC[i]+Cte_coeff*TW[i]  
            # Condicion a la frontera
     
            if i==nx-1:
                TC[i]=TC[i]+TE[i]  #"Gradiente de concentracion 0" dC/dx=0 
                
            if i == 0:
                B[i] =  c_n[i]/dt-Cte_rhs*TW[i]
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
