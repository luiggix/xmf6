
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix 
from scipy.sparse import csr_matrix 
from scipy.sparse.linalg import spsolve  
from time import time  

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

def solveImp(ph_par, mesh, tdis, hx, qx, verb = 1):
    sdis = ph_par["specific_discharge"]   # Specific discharge ($cm s^{-1}$
    h    = ph_par["hydraulic_conductivity"]  # Hydraulic conductivity ($cm s^{-1}$)
    cs   = ph_par["source_concentration"]   #dimensionless
    c0   = ph_par["initial_concentration"]  #dimensionless
    Por  = ph_par["porosity"]
    Dl   = ph_par["longitudinal_dispersivity"]
    rf    = ph_par["retardation_factor"]
    Dr   = ph_par["decay_rate"]
    Dh   = ph_par["dispersion_coefficient"]  #!!!! Ya viene dividido por el factor de retardo rf
  
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

    THETA=1.0    #Esquema de discretización en el tiempo (se toma un esquma de discretización implicito)
    recharge=0.0  #Esto podría ser un vector de recargas a futuro 
    c_inf=cs    #concentración de infiltración es igual a la concentración fuente 
    Dh=Dh*Por #utilizar la ecuacion de descargas hidraulicas
    Dr=Dr*Por 
    #------------------------------------------------------------------------------------------#
    #---------- Calculando los valores de q para cada nodo y fronteras de la celdas ----------------#
    #------------------------------------------------------------------------------------------#
    
     
    q_i = np.zeros(nx)                            #Vector de q en los nodos o centros "i" 
#    q_i= qx[0][0][:]/(rf*Por)                              #para tomar descargas especificas (vienen de modflow o algun simulador de flujo)
    q_i= qx[0][0][:]/rf                              #para tomar descargas especificas (vienen de modflow o algun simulador de flujo)
 
    h_i = np.zeros(nx)          #Vector de cargas hidraulicas en los nodos o centros "i" 
    h_i= hx[0][0][:]            #se toma solo la distribución de cargas hidraulicas en direccion x (vienen de modflow o algun simulador de flujo)
    
    #------------------------------------------------------------------------------------------#
    #---------- Definiendo vectores y matrices para el enfoque de mezclas ---------------------#
    #------------------------------------------------------------------------------------------#
    
    A = np.zeros((nx, nx))          #Matriz de coeficientes A (contiene las proporciones de mezcla o lamnda)
    R = np.identity(nx)*recharge       #Contiene las recargas y descargas 
    Q = np.zeros((nx,nx))          #Contiene los flujos de infiltración
    D = np.identity(nx)*Por           #Contiene la distribución de porosidad del medio poroso (parte acumulativa de la Ecu de AdvDiff 
        
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
        print("|----------WMA SOLUCIÓN DIFERENCIAS FINITAS TRADICIONALES IMPLÍCITO------------|")
        print("|------------------------------------------------------------------------------|")
    
    #------------------------------------------------------------------------------------------#
    #------------------------ Inicio del ciclo solución en el tiempo --------------------------#
    #------------------------------------------------------------------------------------------#
    tiempoComputo_inicial = time()       
    
    q_inf=q_i[0]  ##Este valor se tiene que cambiar si hay una q_inf

    for ti in range(0, nstp):
            
        #--------------------------------------------------------------------------------------#
        #-- Calculando los coeficientes "A" Y las matrices diagonales D, R, Q, --------#
        #--------------------------------------------------------------------------------------# 
        
        for i in range(0, nx):
            
            if ((i>0)and(i<nx-1)):

                qe, qw = upwind_1D(i, h_i, q_i)   #Calclulo de los flujos en las caras (puede tomarse un esquema upwind o TVD, etc)
                
                TE = -qe/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
                TW =  qw/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
                TC = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge

                A[i][i+1]= TE  
                A[i][i]=   TC
                A[i][i-1]= TW
    
            if i==0:
                if(h_i[i]<=h_i[i+1]):  #En la Frontera no sería conveniente utilizar otro esquemás mas que el upwind
                    qe=q_i[i+1]
                else:
                    qe=q_i[i]
                
                qw=q_inf    #solo porque es igual se tendría que cambiar cuando CF tenga algun valor

                TE = -qe/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
                TW =  qw/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
                TC = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge
                
                A[i][i+1]= TE   
                A[i][i]=   TC+TW*(1.-(q_i[i]*delta_x)/Dh)    
                
                Q[0][0]=TW*((q_inf*delta_x)/Dh)
            
            if i==nx-1:
                if(h_i[i-1]<=h_i[i]):     #En la Frontera no sería conveniente utilizar otro cosa que el upwind
                    qw=q_i[i]
                else:
                    qw=q_i[i-1]
                
                qe=q_i[i]

                TE = -qe/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
                TW =  qw/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
                TC = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge

                A[i][i]=   TC+TE      #Condicion de frontera Neumann
                A[i][i-1]= TW
            
        cr=0.0   #Concentración de recarga (por el momento no se está utilizando recarga)
        cr_vec=np.ones(nx)*cr     #vector de concentración de recargas (en caso de tener una distribución en el medio poroso y para poder hacer la operación matricial)
        cinf_vec=np.zeros(nx)  #vector de concentración de infiltración (en caso de tener una distribución en el medio poroso y para poder hacer la operación matricial)
        cinf_vec[0]=c_inf  #vector de concentración de infiltración (en caso de tener una distribución en el medio poroso y para poder hacer la operación 
        
        sumaMat= np.dot((D*(1/dt)+(1-THETA)*A),c_n) +np.dot(R,cr_vec)+np.dot(Q,cinf_vec)   #OPERACIONES MATRICIALES DEL WMA IMPLICITO
        c_v=np.linalg.inv(D*(1/dt)-THETA*A).dot(sumaMat)

        solcnsImp_c[ti+1][:] = c_v[:]                                    #Guardando las soluciones en la matriz de c para cada paso de tiempo
        c_n[:] = c_v[:]                                                  #Guardando el valor de c al tiempo n que se utilizará en el siguiente paso de tiempo

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
        print("|------------------ FIN DE LA SIMULACIÓN WMA IMP  -------------------------|")
        print("|------------------------------------------------------------------------------|")

    return solcnsImp_c





def solveExp(ph_par, mesh, tdis, hx, qx, nx_div, verb = 1):
    sdis = ph_par["specific_discharge"]   # Specific discharge ($cm s^{-1}$
    h    = ph_par["hydraulic_conductivity"]  # Hydraulic conductivity ($cm s^{-1}$)
    cs   = ph_par["source_concentration"]   #dimensionless
    c0   = ph_par["initial_concentration"]  #dimensionless
    Por  = ph_par["porosity"]
    Dl   = ph_par["longitudinal_dispersivity"]
    rf    = ph_par["retardation_factor"]
    Dr   = ph_par["decay_rate"]
    Dh   = ph_par["dispersion_coefficient"]  #!!!! Ya viene dividido por el factor de retardo rf
  
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

    THETA=1.0    #Esquema de discretización en el tiempo (se toma un esquma de discretización implicito)
    recharge=0.0  #Esto podría ser un vector de recargas a futuro 
    c_inf=cs    #concentración de infiltración es igual a la concentración fuente 
    Dh=Dh*Por #utilizar la ecuacion de descargas hidraulicas
    Dr=Dr*Por 
    #------------------------------------------------------------------------------------------#
    #---------- Calculando los valores de q para cada nodo y fronteras de la celdas ----------------#
    #------------------------------------------------------------------------------------------#
    
     
    q_i = np.zeros(nx)                            #Vector de q en los nodos o centros "i" 
#    q_i= qx[0][0][:]/(rf*Por)                              #para tomar descargas especificas (vienen de modflow o algun simulador de flujo)
    q_i= qx[0][0][:]/rf                              #para tomar descargas especificas (vienen de modflow o algun simulador de flujo)
 
    h_i = np.zeros(nx)          #Vector de cargas hidraulicas en los nodos o centros "i" 
    h_i= hx[0][0][:]            #se toma solo la distribución de cargas hidraulicas en direccion x (vienen de modflow o algun simulador de flujo)
    
    #------------------------------------------------------------------------------------------#
    #---------- Definiendo vectores y matrices para el enfoque de mezclas ---------------------#
    #------------------------------------------------------------------------------------------#
    
    A = np.zeros((nx, nx))          #Matriz de coeficientes A (contiene las proporciones de mezcla o lamnda)
    R = np.identity(nx)*recharge       #Contiene las recargas y descargas 
    Q = np.zeros((nx,nx))          #Contiene los flujos de infiltración
    D = np.identity(nx)*Por           #Contiene la distribución de porosidad del medio poroso (parte acumulativa de la Ecu de AdvDiff 
        
    c_ini = np.zeros(nx)                       #Vector de condiciones iniciales de c
    c_n = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
    c_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
    solcnsExp_c = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)

    #------------------------------------------------------------------------------------------#
    #---------------------------- Definiendo condiciones iniciales ----------------------------#
    #------------------------------------------------------------------------------------------#
    c_n[:] = c0
    solcnsExp_c[0][:] = c0  

    if verb != 0:
        print("|------------------------------------------------------------------------------|")
        print("|----------WMA SOLUCIÓN DIFERENCIAS FINITAS TRADICIONALES EXPLÍCITO------------|")
        print("|------------------------------------------------------------------------------|")
    
    #------------------------------------------------------------------------------------------#
    #------------------------ Inicio del ciclo solución en el tiempo --------------------------#
    #------------------------------------------------------------------------------------------#
    tiempoComputo_inicial = time()       
    
    q_inf=q_i[0]  ##Este valor se tiene que cambiar si hay una q_inf

    for ti in range(0, nstp):
            
        #--------------------------------------------------------------------------------------#
        #-- Calculando los coeficientes "A" Y las matrices diagonales D, R, Q, --------#
        #--------------------------------------------------------------------------------------# 
        
        for i in range(0, nx):
            
            if ((i>0)and(i<nx-1)):

                qe, qw = upwind_1D(i, h_i, q_i)   #Calclulo de los flujos en las caras (puede tomarse un esquema upwind o TVD, etc)
                
                TE = -qe/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
                TW =  qw/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
                TC = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge

                A[i][i+1]= TE  
                A[i][i]=   TC
                A[i][i-1]= TW
    
            if i==0:
                if(h_i[i]<=h_i[i+1]):  #En la Frontera no sería conveniente utilizar otro esquemás mas que el upwind
                    qe=q_i[i+1]
                else:
                    qe=q_i[i]
                
                qw=q_inf    #solo porque es igual se tendría que cambiar cuando CF tenga algun valor

                TE = -qe/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
                TW =  qw/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
                TC = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge
                
                A[i][i+1]= TE   
                A[i][i]=   TC+TW*(1.-(q_i[i]*delta_x)/Dh)    
                
                Q[0][0]=TW*((q_inf*delta_x)/Dh)
            
            if i==nx-1:
                if(h_i[i-1]<=h_i[i]):     #En la Frontera no sería conveniente utilizar otro cosa que el upwind
                    qw=q_i[i]
                else:
                    qw=q_i[i-1]
                
                qe=q_i[i]

                TE = -qe/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
                TW =  qw/(2*delta_x)+Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
                TC = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge

                A[i][i]=   TC+TE      #Condicion de frontera Neumann
                A[i][i-1]= TW
            
        cr=0.0   #Concentración de recarga (por el momento no se está utilizando recarga)
        cr_vec=np.ones(nx)*cr     #vector de concentración de recargas (en caso de tener una distribución en el medio poroso y para poder hacer la operación matricial)
        cinf_vec=np.zeros(nx)  #vector de concentración de infiltración (en caso de tener una distribución en el medio poroso y para poder hacer la operación matricial)
        cinf_vec[0]=c_inf  #vector de concentración de infiltración (en caso de tener una distribución en el medio poroso y para poder hacer la operación 
        
        sumaMat= np.dot(A,c_n) +np.dot(R,cr_vec)+np.dot(Q,cinf_vec)   #OPERACIONES MATRICIALES DEL WMA EXPLICITO
        c_v=c_n+dt*np.linalg.inv(D).dot(sumaMat)

        solcnsExp_c[ti+1][:] = c_v[:]                                    #Guardando las soluciones en la matriz de c para cada paso de tiempo
        c_n[:] = c_v[:]                                                  #Guardando el valor de c al tiempo n que se utilizará en el siguiente paso de tiempo

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
        print("|------------------ FIN DE LA SIMULACIÓN WMA EXP  -------------------------|")
        print("|------------------------------------------------------------------------------|")

    return solcnsExp_c

####Qué regresará #### Solo las lamdas o toda 
def ratesMixing(ph_par, mesh, tdis, hx, qx, nx_div, verb = 1):




    return()


    
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

