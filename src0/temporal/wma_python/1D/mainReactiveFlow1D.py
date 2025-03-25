# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:49:25 2024

@author: leo_teja
"""

import numpy as np
import matplotlib.pyplot as plt
import interpolationSchemes as inter
import pandas as pd
import resultsFortran as rF

from PIL import Image              

from time import time 
import sys

"""
Este codigo resuelve la ecuacion de transporte reactivo (binario) unidimensional  
mediante un pulso de la especie C2 que se mezcla con la especie C1

                                                            dC1/dx=0
   |--o--|--o--|--o--|--o--|--o--|--o--|--o--|--o--|--o--|--o--|
   0                    x0                                   
   
   <----------------------- Lx --------------------------------->   

La técnica de discretación es diferencias finitas centrales y
un Euler regresivo en el tiempo con un esquema implícito en el tiempo

"""

###Data 

#### physical variables #####

ph_par = dict(
    specific_discharge = 0.1,  # Specific discharge ($cm s^{-1}$)
    hydraulic_conductivity = 0.01,  # Hydraulic conductivity ($cm s^{-1}$)
    source_concentration = 1.0,  # Source concentration (unitless)
    porosity = 0.5,  # Porosity of mobile domain (unitless)
    initial_concentration = 1.0,  # Initial concentration (unitless)
)

long_disp=1.0
reta_fact=0.0
deca_rate=0.0

ph_par["longitudinal_dispersivity"] = long_disp 
ph_par["retardation_factor"] = reta_fact
ph_par["decay_rate"] =  deca_rate
ph_par["dispersion_coefficient"] = 0.2


######## Discretización espacial y temporal
Lx=30.0  #m 
nx=30    #number of cells on axis x 
dx=Lx/nx    #length of cells 
xc=np.linspace(0.5*dx,30-0.5*dx, nx)   #centers of cells 



sdis = ph_par["specific_discharge"]   # Specific discharge ($cm s^{-1}$
h    = ph_par["hydraulic_conductivity"]  # Hydraulic conductivity ($cm s^{-1}$)
cs   = ph_par["source_concentration"]   #dimensionless
c0   = ph_par["initial_concentration"]  #dimensionless
Por  = ph_par["porosity"]
Dl   = ph_par["longitudinal_dispersivity"]
rf    = ph_par["retardation_factor"]
Dr   = ph_par["decay_rate"]
Dh   = ph_par["dispersion_coefficient"]  #!!!! Ya viene dividido por el factor de retardo rf
  
Lx = 30.0
Ly = 1.0
Lz = 1.0
nx = 30
ny = 1
nz = 1
delta_x = Lx/nx
delta_y = 1.0
delta_z = 1.0
xi=np.linspace(0.5*dx,30-0.5*dx, nx)   #centers of cells 

pulseM=200.0    ####¡?? Unidades 
x0=10.5     #Celda donde entra el pulso de la especie "2" !!!
k=3.29E-05   #Constante de equilibrio para reacción Ca2^+2 + SO4^-2 <----> Ca2SO4


total_time = 90.0    #Total time of simulation days
nper = 1            # Number of periods
#perlen, nstp, tsmult = tdis.perioddata(0) # PERLEN, NSTP, TSMULT
dt = 1.0   #delta of time    
nstp=int(total_time/dt)


THETA=1.0     #Esquema de discretización en el tiempo (se toma un esquma de discretización implicito)
recharge=0.0  #Esto podría ser un vector de recargas a futuro 
c_inf=cs       #concentración de infiltración es igual a la concentración fuente 
# Dh=Dh*Por      #utilizar la ecuacion de descargas hidraulicas
Dh=Dh      #utilizar la ecuacion de descargas hidraulicas
Dr=Dr*Por 
#------------------------------------------------------------------------------------------#
#---------- Calculando los valores de q para cada nodo y fronteras de la celdas ----------------#
#------------------------------------------------------------------------------------------#

q_i=np.ones(nx)*0.2          #specific discharge 
v_i=q_i/ph_par["porosity"]    #velocity 

 
# h_i = np.zeros(nx)          #Vector de cargas hidraulicas en los nodos o centros "i" 
# h_i= hx[0][0][:]            #se toma solo la distribución de cargas hidraulicas en direccion x (vienen de modflow o algun simulador de flujo)
h_i=np.linspace(10, 5, nx)

#------------------------------------------------------------------------------------------#
#---------- Definiendo vectores y matrices para el enfoque de mezclas ---------------------#
#------------------------------------------------------------------------------------------#

A = np.zeros((nx, nx))          #Matriz de coeficientes A (contiene las proporciones de mezcla o lamnda)
B = np.zeros(nx)                #Rhs del sistema de ecuaciones de las U
R = np.identity(nx)*recharge       #Contiene las recargas y descargas 
Q = np.zeros((nx,nx))          #Contiene los flujos de infiltración
D = np.identity(nx)*Por           #Contiene la distribución de porosidad del medio poroso (parte acumulativa de la Ecu de AdvDiff 

TW = np.zeros(nx)
TE = np.zeros(nx)
TC = np.zeros(nx)

#------------------------------------------------------------------------------------------#
#---------------------------- Definiendo condiciones iniciales ----------------------------#
#------------------------------------------------------------------------------------------#

    
c1_ini = np.ones(nx)*c0                       #Vector de condiciones iniciales de c1
c2_ini = np.ones(nx)*(k/c0)                       #Vector de condiciones iniciales de c2

for i in range (0, nx):
    if abs(x0-xi[i])<1e-03:
        c1_ini[i]=pulseM
        c2_ini[i]=(k/c1_ini[i])
        
c1_n = np.copy(c1_ini)                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
c1_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
solcnsImp_c1 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)


#sys.exit()

c2_n = np.copy(c2_ini)                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
c2_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
solcnsImp_c2 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)


u1_ini = c1_ini-c2_ini                      #Vector de condiciones iniciales de c
u1_n = np.copy(u1_ini)                         #Vector de valor de c a un paso de tiempo n (necesario para la solución del sistema de ecuaciones)
u1_v = np.zeros(nx)                         #Vector de valor de c a un paso de tiempo n+1
solcnsImp_u1 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
solcnsImp_u1[0][:] = u1_n 

solcnsImp_c1[0][:] = c1_ini
solcnsImp_c2[0][:] = c2_ini  


r1 = np.zeros(nx) 
solcnsImp_r1 = np.zeros((nstp, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)

# if verb != 0:
#     print("|------------------------------------------------------------------------------|")
#     print("|----------WMA SOLUCIÓN DIFERENCIAS FINITAS TRADICIONALES IMPLÍCITO------------|")
#     print("|------------------------------------------------------------------------------|")



#------------------------------------------------------------------------------------------#
#------------------------ Inicio del ciclo solución en el tiempo --------------------------#
#------------------------------------------------------------------------------------------#
framesU1=[]
framesC1=[]  #Lista vacia para salvar las imagenes (se hará tan grande como tantas imagenes guardemos en ella)
framesC2=[]
framesR1=[]



tiempoComputo_inicial = time()       

q_inf=q_i[0]  ##Este valor se tiene que cambiar si hay una q_inf

for ti in range(0, nstp):
        
    #--------------------------------------------------------------------------------------#
    #--------- Calculando los coeficientes "A" Y las matrices diagonales            -------#
    #--------------------------------------------------------------------------------------# 
    
    
    for i in range(0, nx):

        B[i]=Por*u1_n[i]/dt
        
        if ((i>0)and(i<nx-1)):

            qe, qw = inter.upwind_1D(i, h_i, q_i)   #Calclulo de los flujos en las caras (puede tomarse un esquema upwind o TVD, etc)
            
            TE[i] =  qe/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            TW[i] = -qw/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            TC[i] = -(qe-qw)/(2*delta_x)+(2*Dh)/(delta_x**2)+Por/dt+ Dr+recharge

            A[i][i+1]= TE[i]  
            A[i][i]=   TC[i]
            A[i][i-1]= TW[i]

        if i==0:
            if(h_i[i]<=h_i[i+1]):  #En la Frontera no sería conveniente utilizar otro esquemás mas que el upwind
                qe=q_i[i+1]
            else:
                qe=q_i[i]
            
            qw=q_inf    #solo porque es igual se tendría que cambiar cuando CF tenga algun valor

            TE[i] =  qe/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            TW[i] = -qw/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            TC[i] = -(qe-qw)/(2*delta_x)+(2*Dh)/(delta_x**2)+Por/dt+Dr+recharge
            
            A[i][i+1]= TE[i]   
            A[i][i]=   TC[i]+TW[i]*(1.-(q_i[i]*delta_x)/Dh)    
            
            B[i]=Por*u1_n[i]/dt
            B[0]=B[0]-TW[i]*((q_inf*delta_x)/Dh)*(c_inf-k/c_inf)
        
        if i==nx-1:
            if(h_i[i-1]<=h_i[i]):     #En la Frontera no sería conveniente utilizar otro cosa que el upwind
                qw=q_i[i]
            else:
                qw=q_i[i-1]
            
            qe=q_i[i]

            TE[i] =  qe/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            TW[i] = -qw/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            TC[i] = -(qe-qw)/(2*delta_x)+(2*Dh)/(delta_x**2)+Por/dt+Dr+recharge

            A[i][i]=   TC[i]+TE[i]      #Condicion de frontera Neumann
            A[i][i-1]= TW[i]
        
    u1_v=np.linalg.inv(A).dot(B)
    solcnsImp_u1[ti+1][:] = u1_v[:]                                    #Guardando las soluciones en la matriz de c para cada paso de tiempo
    u1_n = np.copy(u1_v)                                               #Guardando el valor de c al tiempo n que se utilizará en el siguiente paso de tiempo

    ### Calculo de C1 y C2 mediante la constante de equilibrio y la u y velocidades de reaccion mediantes C1 o puede ser mediante C2
    
    for i in range(0, nx):
        c1_v[i]=(u1_v[i]+np.sqrt(u1_v[i]**2+4*k))/2
        c2_v[i]=(-u1_v[i]+np.sqrt(u1_v[i]**2+4*k))/2
        
        if ((i>0)and(i<nx-1)):
            r1[i]=TW[i]*c1_v[i-1]+TC[i]*c1_v[i]+TE[i]*c1_v[i+1]-Por*c1_n[i]/dt
        if i==0:
            r1[i]=(TC[i]+TW[i]*(1.-(q_i[i]*delta_x)/Dh))*c1_v[i]+TE[i]*c1_v[i+1]-Por*c1_n[i]/dt+TW[i]*((q_inf*delta_x)/Dh)*(c_inf)
        if i==nx-1:
            r1[i]=TW[i]*c1_v[i-1]+(TC[i]+TE[i])*c1_v[i]-Por*c1_n[i]/dt
            
            
        # u1_n[i]=c1_v[i]-c2_v[i]
        
    
    solcnsImp_c1[ti+1][:] = c1_v[:]                                    #Guardando las soluciones en la matriz de c para cada paso de tiempo
    solcnsImp_c2[ti+1][:] = c2_v[:]                                    #Guardando las soluciones en la matriz de c para cada paso de tiempo
    solcnsImp_r1[ti][:] = r1[:]
    
    c1_n=np.copy(c1_v)
    c2_n=np.copy(c2_v)
        
    # if verb == 2:
    #     print("tiempo de simulacion: " +str(round(dt*(ti+1),2))+" segundos" )
    # elif verb == 1:
    #     print("{:>5d}".format(ti), end=(''))
    # else:
    #     print("".format(ti), end=(''))
    
    print("tiempo de simulacion: " +str(round(dt*(ti+1),2))+" dias" )
    

#------------------------------------------------------------------------------------------#
#---------------------- Tiempo de ejecución del modelo de simulación ----------------------#
#------------------------------------------------------------------------------------------# 

tiempoComputo_final=time()                                                                         #Toma el valor de tiempo final de ejecución, para  calcuar el tiempo total de la simulación
tiempo_ejec_seg=(tiempoComputo_final-tiempoComputo_inicial)                                               #Calcula el tiempo total de simulación, pasado a segundos
tiempo_ejec_min=(tiempoComputo_final-tiempoComputo_inicial)/60                                            #Calcula el tiempo total de simulación, pasado a minutos

print("\nTiempo de Ejecución:",tiempo_ejec_seg, "[Segundos]","\t", tiempo_ejec_min, "[Minutos]\n")
print("|------------------------------------------------------------------------------|")
print("|------------------ FIN DE LA SIMULACIÓN WMA IMP  -------------------------|")
print("|------------------------------------------------------------------------------|")



#------------------------------------------------------------------------------------------#
#--------------------------- Graficando todos los resultados ------------------------------#
#------------------------------------------------------------------------------------------#

plt.figure('Graf_u1',figsize=(9,8))
plt.style.use('fast')
plt.title('Perfil de $U_1$ vs distancia')
# plt.plot(xi, solcnsImp_u1[0][:], '-r', linewidth=3.0)
for i in range(0,nstp+1):
    
    plt.plot(xi, solcnsImp_u1[i][:])
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("u [$mgr/cm^{3}$]")

    # stringTimes=round(dt*(i+1),2)
    # stringU1Save = 'U1 %s dias.png' %str(stringTimes)
    # plt.savefig(stringU1Save, dpi=200)
    # imgU1 = Image.open(stringU1Save)   #Se abre el archivo de imagen de la presion recien guardado
    # framesU1.append(imgU1)   #El archivo abierto de imagen de anexa a la lista de frames

                                                                   
plt.minorticks_on()
plt.grid(True,which='major', color='w', linestyle='-')
plt.grid(True, which='minor', color='w', linestyle='--')


plt.figure('Graf_c1',figsize=(9,8))
plt.style.use('fast')
plt.minorticks_on()
plt.title('Perfil de $c_1$ vs distancia en el tiempo')
for i in range(0, nstp+1):
    
    plt.plot(xi, solcnsImp_c1[i][:]) 
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("Concentraciones [$mgr/cm^{3}$]") 

    # stringTimes=round(dt*(i+1),2)
    # stringC1Save = 'C1 %s dias.png' %str(stringTimes)
    # plt.savefig(stringC1Save, dpi=200)
    # imgC1 = Image.open(stringC1Save)   #Se abre el archivo de imagen de la presion recien guardado
    # framesC1.append(imgC1)   #El archivo abierto de imagen de anexa a la lista de frames
                                                                  

plt.grid(True,which='major', color='w', linestyle='-')
plt.grid(True, which='minor', color='w', linestyle='--')


plt.figure('Graf_c2',figsize=(9,8))
plt.style.use('fast')
plt.minorticks_on()
plt.title('Perfil de $c_2$ vs distancia en el tiempo')
for i in range(0, nstp+1):
    
    plt.plot(xi, solcnsImp_c2[i][:]) 
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("Concentraciones [$mgr/cm^{3}$]")

    # stringTimes=round(dt*(i+1),2)
    # stringC2Save = 'C2 %s dias.png' %str(stringTimes)
    # plt.savefig(stringC2Save, dpi=200)
    # imgC2 = Image.open(stringC2Save)   #Se abre el archivo de imagen de la presion recien guardado
    # framesC2.append(imgC2)   #El archivo abierto de imagen de anexa a la lista de frames

                                                                   
plt.grid(True,which='major', color='w', linestyle='-')
plt.grid(True, which='minor', color='w', linestyle='--')


plt.figure('Graf_r1',figsize=(9,8))
plt.style.use('fast')
plt.minorticks_on()
plt.title('Perfil de $r$ vs distancia en el tiempo')
for i in range(0, nstp):
    
    plt.plot(xi, solcnsImp_r1[i][:]) 
    plt.xlabel("Distancia x [$m$]")
    plt.ylabel("Velocidad de reaccion [$mgr/cm^{3}/day$]")

    # stringTimes=round(dt*(i+1),2)
    # stringR1Save = 'R1 %s dias.png' %str(stringTimes)
    # plt.savefig(stringR1Save, dpi=200)
    # imgR1 = Image.open(stringR1Save)   #Se abre el archivo de imagen de la presion recien guardado
    # framesR1.append(imgR1)   #El archivo abierto de imagen de anexa a la lista de frames

                                                                   
plt.grid(True,which='major', color='w', linestyle='-')
plt.grid(True, which='minor', color='w', linestyle='--')





#--------------------------------------------------------------------------------------------------#
#------------------------ Graficando un solo resultado y comparandolo vs WMAI ----------------------#
#---------------------------------------------------------------------------------......---------#
tsel=10   #tiempo seleccionado de hoja excel


plt.figure('comparativa_C1',figsize=(9,8))
WMA_I=pd.read_excel('comparativa_02.xlsx', sheet_name='c_1')     #Crea un data frame llamado WMA_I a partir de la tabla de excel
data_set_C1 = np.transpose(WMA_I.iloc[11:,5:95].to_numpy())
# data_set = WMA_I.iloc[11:,5:95].to_numpy()
#WMA_I
plt.plot(xi, solcnsImp_c1[0][:], "-r"  , label="Condicion inicial")
plt.plot(xi, solcnsImp_c1[tsel][:], "-b*", label="Tiempo de simulacion "+str(round(tsel,2))+" dias este código")
plt.plot(xi, data_set_C1[tsel][:], "-.m", label="Tiempo de simulacion "+str(round(tsel,2))+" dias WMA_I" )
plt.xlabel("Distancia x [$m$]"), plt.ylabel("$c_1 [mgr/cm^{3}]$")
plt.legend(loc=0); plt.minorticks_on()
plt.grid(True,which='major', color='w', linestyle='-')
plt.grid(True, which='minor', color='g', linestyle='--')
plt.savefig("Comparativa C1.png", dpi=300)


Res_contrsns_ini = []
Res_contrsns_fin = []
file_name='gypsum_eq.out'
Res_contrsns_ini, Res_contrsns_fin = rF.import_results(file_name)

plt.figure('comparativa_C2',figsize=(9,8))
WMA_I=pd.read_excel('comparativa_02.xlsx', sheet_name='c_2')     #Crea un data frame llamado WMA_I a partir de la tabla de excel
data_set_C2 = np.transpose(WMA_I.iloc[11:,5:95].to_numpy())
plt.plot(xi, solcnsImp_c2[0][:], "-r"  , label="Condicion inicial")
plt.plot(xi, solcnsImp_c2[tsel][:], "-b*", label="Tiempo de simulacion "+str(round(tsel,2))+" dias este código")
plt.plot(xi, data_set_C2[tsel][:], "-.m", label="Tiempo de simulacion "+str(round(tsel,2))+" dias WMA_I" )
plt.plot(xi, Res_contrsns_fin[1][:], "g--", label="Fortrab so4-2 ")
plt.xlabel("Distancia x [$m$]"), plt.ylabel("$c_2 [mgr/cm^{3}]$")
plt.legend(loc=0); plt.minorticks_on()
plt.grid(True,which='major', color='w', linestyle='-')
plt.grid(True, which='minor', color='g', linestyle='--')
plt.savefig("Comparativa C2.png", dpi=300)

# framesU1[0].save('U1.gif', format='GIF', append_images=framesU1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
# framesC1[0].save('C1.gif', format='GIF', append_images=framesC1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
# framesC2[0].save('C2.gif', format='GIF', append_images=framesC2[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
# framesR1[0].save('R1.gif', format='GIF', append_images=framesR1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames

