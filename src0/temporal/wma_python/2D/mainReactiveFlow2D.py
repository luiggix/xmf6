# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:49:25 2024

@author: leo_teja
"""

import numpy as np
import matplotlib.pyplot as plt
import interpolationSchemes as inter
import pandas as pd
from scipy.sparse import lil_matrix  #Es una lista de matrices de scipy 
from scipy.sparse.linalg import spsolve   #Descomposición LU optimizada

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
nx=60    #number of cells on axis x 
delta_x=Lx/nx    #length of cells 
xc=np.linspace(0.5*delta_x,Lx-0.5*delta_x, nx)   #centers of cells 

Ly=10.0  #m 
ny=20    #number of cells on axis x 
delta_y=Ly/ny    #length of cells 
yc=np.linspace(0.5*delta_y,Ly-0.5*delta_y, ny)   #centers of cells 

Lz = 1.0
nz = 1

xg, yg = np.meshgrid(xc, yc)   


sdis = ph_par["specific_discharge"]   # Specific discharge ($cm s^{-1}$
h    = ph_par["hydraulic_conductivity"]  # Hydraulic conductivity ($cm s^{-1}$)
cs   = ph_par["source_concentration"]   #dimensionless
c0   = ph_par["initial_concentration"]  #dimensionless
Por  = ph_par["porosity"]
Dl   = ph_par["longitudinal_dispersivity"]
rf    = ph_par["retardation_factor"]
Dr   = ph_par["decay_rate"]
Dh   = ph_par["dispersion_coefficient"]  #!!!! Ya viene dividido por el factor de retardo rf
  

pulse01=200.0    ####¡?? Unidades 
pulse02=100.0
pulse03=150.0
pulse04=500.0

cx1=30; cy1=10
cx2=5; cy2=3
cx3=30; cy3=12
cx4=50; cy4=15


#x0=10.5     #Celda donde entra el pulso de la especie "2" !!!
keq=3.29E-05   #Constante de equilibrio para reacción Ca2^+2 + SO4^-2 <----> Ca2SO4


total_time = 90.0    #Total time of simulation days
nper = 1            # Number of periods
#perlen, nstp, tsmult = tdis.perioddata(0) # PERLEN, NSTP, TSMULT
dt = 1.0   #delta of time    
nstp=int(total_time/dt)

delta_save=2.0   #Paso de tiempo para salvar imagenes en días
num_save=int(total_time/delta_save)  #Calculo del numero de veces que salvamos o guardamos imagenes 

THETA=1.0     #Esquema de discretización en el tiempo (se toma un esquma de discretización implicito)
recharge=0.0  #Esto podría ser un vector de recargas a futuro 
c_inf=cs       #concentración de infiltración es igual a la concentración fuente 
# Dh=Dh*Por      #utilizar la ecuacion de descargas hidraulicas
Dh=Dh      #utilizar la ecuacion de descargas hidraulicas
Dr=Dr*Por 

#------------------------------------------------------------------------------------------#
#---------- Calculando los valores de q para cada nodo y fronteras de la celdas ----------------#
#------------------------------------------------------------------------------------------#

qx=np.ones((nx,ny))*0.2          #specific discharge 
ux=qx/ph_par["porosity"]    #velocity 

qy=np.ones((nx,ny))*0.0          #specific discharge 
vy=qy/ph_par["porosity"]    #velocity 

 
# h_i = np.zeros(nx)          #Vector de cargas hidraulicas en los nodos o centros "i" 
# h_i= hx[0][0][:]            #se toma solo la distribución de cargas hidraulicas en direccion x (vienen de modflow o algun simulador de flujo)
head=np.ones((nx,ny))

#------------------------------------------------------------------------------------------#
#---------- Definiendo vectores y matrices para el enfoque de mezclas ---------------------#
#------------------------------------------------------------------------------------------#
n=nx*ny
A = np.zeros((n,n))          #Matriz de coeficientes A (contiene las proporciones de mezcla o lamnda)
b = np.zeros(n)                #Rhs del sistema de ecuaciones de las U

TW = np.zeros((nx,ny))
TE = np.zeros((nx,ny))
TC = np.zeros((nx,ny))
TN = np.zeros((nx,ny))
TS = np.zeros((nx,ny))
B = np.zeros((nx,ny))


#------------------------------------------------------------------------------------------#
#---------------------------- Definiendo condiciones iniciales ----------------------------#
#------------------------------------------------------------------------------------------#

    
c1_n = np.ones((nx,ny))*c0                       #Vector de condiciones iniciales de c1
c2_n = np.ones((nx,ny))*(keq/c0)                       #Vector de condiciones iniciales de c2


for j in range (0,ny):
    for i in range (0,nx):
    
        if ((i==cx1)and(j==cy1)):   
            c1_n[i][j]=pulse01
            c2_n[i][j]=(keq/c1_n[i][j])

        # if ((i==cx2)and(j==cy2)):
        #     c1_n[i][j]=pulse01
        #     c2_n[i][j]=(k/c1_n[i][j])

        # if ((i==cx3)and(j==cy3)):
        #     c1_n[i][j]=pulse01
        #     c2_n[i][j]=(k/c1_n[i][j])

        # if ((i==cx4)and(j==cy4)):   
        #     c1_n[i][j]=pulse01
        #     c2_n[i][j]=(k/c1_n[i][j])

        
c1_v = np.copy(c1_n)                         #Vector de valor de c a un paso de tiempo n+1
solcnsImp_c1 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)

c2_v = np.copy(c2_n)                         #Vector de valor de c a un paso de tiempo n+1
solcnsImp_c2 = np.zeros((nstp+1, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)

u1_n = c1_n-c2_n                      #Vector de condiciones iniciales de c
u1_v = np.copy(u1_n)                         #Vector de valor de c a un paso de tiempo n+1
# solcnsImp_u1 = np.zeros((nstp+1, n))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)
# solcnsImp_u1[0][:] = u1_n 

solcnsImp_c1[0][:] = c1_n[:,cy1]
solcnsImp_c2[0][:] = c2_n[:,cy1]


r1 = np.zeros((nx,ny)) 
solcnsImp_r1 = np.zeros((nstp, nx))            #Matriz de guardado de las c para cada paso de tiempo y en toda la malla 1D (incluyendo las condiciones iniciales al tiempo cero)


# if verb != 0:
#     print("|------------------------------------------------------------------------------|")
#     print("|----------WMA SOLUCIÓN DIFERENCIAS FINITAS TRADICIONALES IMPLÍCITO------------|")
#     print("|------------------------------------------------------------------------------|")



#------------------------------------------------------------------------------------------#
#------------------------ Inicio del ciclo solución en el tiempo --------------------------#
#------------------------------------------------------------------------------------------#

#framesU1=[]
framesC1=[]  #Lista vacia para salvar las imagenes (se hará tan grande como tantas imagenes guardemos en ella)
framesC2=[]
framesR1=[]



tiempoComputo_inicial = time()       

q_inf=qx[0][0]  ##Este valor se tiene que cambiar si hay una q_inf
iSave=0
count_save=1    #Contador para salvar imagenes

for ti in range(0, nstp):
        
    #--------------------------------------------------------------------------------------#
    #--------- Calculando los coeficientes "A" Y las matrices diagonales            -------#
    #--------------------------------------------------------------------------------------# 
    
    
    for j in range (0, ny):
        for i in range (0, nx):
            
            k=i+j*nx    #contador para ensamblar las diagonales de la matriz y el vector v
            B[i][j]=Por*u1_n[i][j]/dt
            
            qe, qw, qn, qs = inter.upwind_2D(nx, ny, i, j, head, qx, qy)   #Calclulo de los flujos en las caras (puede tomarse un esquema upwind o TVD, etc)
            # qe, qw, qn, qs = 0.2, 0.2, 0.2, 0.2   #Calclulo de los flujos en las caras (puede tomarse un esquema upwind o TVD, etc)
                
            TE[i][j] =  qe/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi+1/2
            TW[i][j] = -qw/(2*delta_x)-Dh/(delta_x**2)        #Transmisibilidad en xi-1/2    
            TN[i][j] =  qn/(2*delta_y)-Dh/(delta_y**2)        #Transmisibilidad en j+1/2
            TS[i][j] = -qs/(2*delta_y)-Dh/(delta_y**2)        #Transmisibilidad en j-1/2  
            TC[i][j] = -(qe-qw)/(2*delta_x)-(qn-qs)/(2*delta_y)+(2*Dh)/(delta_x**2)+(2*Dh)/(delta_y**2)+Por/dt+Dr+recharge

        
            if ((i>0)and(i<nx-1)and(j>0)and(j<ny-1)):   #Celdas internas no tocan ninguna frontera
            
    
                A[k][k+1]  = TE[i][j]  
                A[k][k-1]  = TW[i][j]
                A[k][k+nx] = TN[i][j]  
                A[k][k-nx] = TS[i][j]
                A[k][k]    = TC[i][j]
    
            
            if((i==0)and(j>0)and(j<ny-1)): #Volumenes o celdas frontera oeste
                           
                # qw=q_inf    #solo porque es igual se tendría que cambiar cuando CF tenga algun valor                
                # TC[i][j] = -(qe-qw)/(2*delta_x)-(qn-qs)/(2*delta_y)+(2*Dh)/(delta_x**2)+(2*Dh)/(delta_y**2)+Por/dt+Dr+recharge
                
                A[k][i+1]  = TE[i][j]   
                A[k][k+nx] = TN[i][j]  
                A[k][k-nx] = TS[i][j]
                A[k][k]    = TC[i][j]+TW[i][j]*(1.-(qx[i][j]*delta_x)/Dh)
                
                B[i][j]=B[i][j]-TW[i][j]*((q_inf*delta_x)/Dh)*(c_inf-keq/c_inf)
            
            
            
            if((i==nx-1)and(j>0)and(j<ny-1)):  #Volumenes en frontera Este
            
                A[k][k-1]  = TW[i][j]
                A[k][k+nx] = TN[i][j]  
                A[k][k-nx] = TS[i][j]
                A[k][k]    = TC[i][j]+TE[i][j]   

            if((i>0)and(i<nx-1)and(j==0)): #Volumenes en frontera sur S
            
                A[k][k+1]  = TE[i][j]  
                A[k][k-1]  = TW[i][j]
                A[k][k+nx] = TN[i][j]  
                A[k][k]    = TC[i][j]+TS[i][j]
            
            if((i>0)and(i<nx-1)and(j==ny-1)): #Volumenes en frontera N
    
                A[k][k+1]  = TE[i][j]  
                A[k][k-1]  = TW[i][j]
                A[k][k-nx] = TS[i][j]
                A[k][k]    = TC[i][j]+TN[i][j]
                
            if((i==0)and(j==0)):    #volumen origen coincide con 2 fronteras 
        
                # qw=q_inf    #solo porque es igual se tendría que cambiar cuando CF tenga algun valor                
                # TC[i][j] = -(qe-qw)/(2*delta_x)-(qn-qs)/(2*delta_y)+(2*Dh)/(delta_x**2)+(2*Dh)/(delta_y**2)+Por/dt+Dr+recharge
                
                A[k][i+1]  = TE[i][j]   
                A[k][k+nx] = TN[i][j]  
                A[k][k]    = TC[i][j]+TS[i][j]+TW[i][j]*(1.-(qx[i][j]*delta_x)/Dh)
                
                B[i][j]=B[i][j]-TW[i][j]*((q_inf*delta_x)/Dh)*(c_inf-keq/c_inf)
            
            
            if((i==nx-1)and(j==0)):   #Celda o volumen de esquina (nx-1,0) 
            
                A[k][k-1]  = TW[i][j]
                A[k][k+nx] = TN[i][j]  
                A[k][k]    = TC[i][j]+TE[i][j]+TS[i][j]   
            
            if((i==0)and(j==ny-1)):   #Celda esquina (0,ny-1)
                
                # qw=q_inf    #solo porque es igual se tendría que cambiar cuando CF tenga algun valor                
                # TC[i][j] = -(qe-qw)/(2*delta_x)-(qn-qs)/(2*delta_y)+(2*Dh)/(delta_x**2)+(2*Dh)/(delta_y**2)+Por/dt+Dr+recharge
                
                A[k][i+1]  = TE[i][j]    
                A[k][k-nx] = TS[i][j]
                A[k][k]    = TC[i][j]+TN[i][j]+TW[i][j]*(1.-(qx[i][j]*delta_x)/Dh)
                
                B[i][j]=B[i][j]-TW[i][j]*((q_inf*delta_x)/Dh)*(c_inf-keq/c_inf)
                
            if((i==nx-1)and(j==ny-1)):   #Nodo de esquina (nx-1,ny-1) solo se hacen las comparaciones con los nodos internos

                A[k][k-1]  = TW[i][j]
                A[k][k-nx] = TS[i][j]
                A[k][k]    = TC[i][j]+TE[i][j]+TN[i][j]
            
            
            b[k]=B[i][j]   #Asignación del RHS
    
    
    A1 = lil_matrix(A)
    A1 = A1.tocsr()   #Cambia a formato compressed row storage y no almacena los ceros 
    U1 = spsolve(A1, b)  #scipy.sparse.linalg.gmres(A, b, x0=None, tol=1e-05, restart=None, maxiter=None, xtype=None, M=None, callback=None, restrt=None)[source]¶


    for j in range (0,ny):
        for i in range (0,nx):
            k=i+j*nx
            u1_v[i][j]=U1[k]    #Cuando se utilizan spsolve
    
    u1_n = np.copy(u1_v)
    
    # solcnsImp_u1[ti+1][:] = u1_v[:]                                    #Guardando las soluciones en la matriz de c para cada paso de tiempo
                                               #Guardando el valor de c al tiempo n que se utilizará en el siguiente paso de tiempo

    # ### Calculo de C1 y C2 mediante la constante de equilibrio y la u y velocidades de reaccion mediantes C1 o puede ser mediante C2
 
    for j in range (0,ny):
        for i in range (0,nx):

            c1_v[i][j]=( u1_v[i][j]+np.sqrt(u1_v[i][j]**2+4*keq))/2
            c2_v[i][j]=(-u1_v[i][j]+np.sqrt(u1_v[i][j]**2+4*keq))/2


    for j in range (0,ny):
        for i in range (0,nx):
            
            if ((i>0)and(i<nx-1)and(j>0)and(j<ny-1)):   #Celdas internas no tocan ninguna frontera
                r1[i][j]=TS[i][j]*c1_v[i][j-1]+TW[i][j]*c1_v[i-1][j]+TC[i][j]*c1_v[i][j]+TE[i][j]*c1_v[i+1][j]+TN[i][j]*c1_v[i][j+1]-Por*c1_n[i][j]/dt

            if((i==0)and(j>0)and(j<ny-1)): #Volumenes o celdas frontera oeste
                qw=q_inf  
                cdummy=(q_inf*delta_x*c_inf)/Dh+(1.-(qw*delta_x)/Dh)*c1_v[i][j]
                r1[i][j]=TS[i][j]*c1_v[i][j-1]+TW[i][j]*cdummy+TC[i][j]*c1_v[i][j]+TE[i][j]*c1_v[i+1][j]+TN[i][j]*c1_v[i][j+1]-Por*c1_n[i][j]/dt

            if((i==nx-1)and(j>0)and(j<ny-1)):  #Volumenes en frontera Este
                r1[i][j]=TS[i][j]*c1_v[i][j-1]+TW[i][j]*c1_v[i-1][j]+(TC[i][j]+TE[i][j])*c1_v[i][j]+TN[i][j]*c1_v[i][j+1]-Por*c1_n[i][j]/dt
            
            if((i>0)and(i<nx-1)and(j==0)): #Volumenes en frontera sur S
                r1[i][j]=TW[i][j]*c1_v[i-1][j]+(TC[i][j]+TS[i][j])*c1_v[i][j]+TE[i][j]*c1_v[i+1][j]+TN[i][j]*c1_v[i][j+1]-Por*c1_n[i][j]/dt
            
            if((i>0)and(i<nx-1)and(j==ny-1)): #Volumenes en frontera N
                r1[i][j]=TS[i][j]*c1_v[i][j-1]+TW[i][j]*c1_v[i-1][j]+(TC[i][j]+TN[i][j])*c1_v[i][j]+TE[i][j]*c1_v[i+1][j]-Por*c1_n[i][j]/dt

            if((i==0)and(j==0)):    #volumen origen coincide con 2 fronteras 
                qw=q_inf  
                cdummy=(q_inf*delta_x*c_inf)/Dh+(1.-(qw*delta_x)/Dh)*c1_v[i][j]
                r1[i][j]=TW[i][j]*cdummy+(TC[i][j]+TS[i][j])*c1_v[i][j]+TE[i][j]*c1_v[i+1][j]+TN[i][j]*c1_v[i][j+1]-Por*c1_n[i][j]/dt
                               
            if((i==nx-1)and(j==0)):   #Celda o volumen de esquina (nx-1,0) 
                r1[i][j]=TW[i][j]*c1_v[i-1][j]+(TC[i][j]+TE[i][j]+TS[i][j])*c1_v[i][j]+TN[i][j]*c1_v[i][j+1]-Por*c1_n[i][j]/dt
            
            if((i==0)and(j==ny-1)):   #Celda esquina (0,ny-1)
                qw=q_inf  
                cdummy=(q_inf*delta_x*c_inf)/Dh+(1.-(qw*delta_x)/Dh)*c1_v[i][j]
                r1[i][j]=TS[i][j]*c1_v[i][j-1]+TW[i][j]*cdummy+(TC[i][j]+TN[i][j])*c1_v[i][j]+TE[i][j]*c1_v[i+1][j]-Por*c1_n[i][j]/dt
                
            if((i==nx-1)and(j==ny-1)):   #Nodo de esquina (nx-1,ny-1) solo se hacen las comparaciones con los nodos internos
                r1[i][j]=TS[i][j]*c1_v[i][j-1]+TW[i][j]*c1_v[i-1][j]+(TC[i][j]+TE[i][j]+TN[i][j])*c1_v[i][j]-Por*c1_n[i][j]/dt



    #     if i==0:
    #         r1[i]=(TC[i]+TW[i]*(1.-(q_i[i]*delta_x)/Dh))*c1_v[i]+TE[i]*c1_v[i+1]-Por*c1_n[i]/dt+TW[i]*((q_inf*delta_x)/Dh)*(c_inf)
    #     if i==nx-1:
    #         r1[i]=TW[i]*c1_v[i-1]+(TC[i]+TE[i])*c1_v[i]-Por*c1_n[i]/dt
            
            
        # u1_n[i]=c1_v[i]-c2_v[i]
        
    
    solcnsImp_c1[ti+1][:] = c1_v[:,cy1]                                    #Guardando las soluciones en la matriz de c para cada paso de tiempo
    solcnsImp_c2[ti+1][:] = c2_v[:,cy1]                                    #Guardando las soluciones en la matriz de c para cada paso de tiempo
    solcnsImp_r1[ti][:] = r1[:,cy1]
    
    c1_n=np.copy(c1_v)
    c2_n=np.copy(c2_v)


    iSave+=1
    
    tiempo=dt*(ti+1)
  #--------------------------------------------------------------------#
  #--------- GUARDADO DE ARCHIVOS Y GRAFICAS CADA "DELTA_SAVE"---------#
  #--------------------------------------------------------------------#

      
    if((abs(tiempo-delta_save*count_save)<1e-05)and(count_save<=num_save)):    #Condicional para el salvado de graficas (se puede utilizar también para salvar los resultados en tablas)
    #Variables tipo cadena para salvar las graficas de saturación
      stringTimes=round(tiempo,2)      #Variable que redondea el tiempo actual en dos digitos despues del punto
      stringC1 = 'c1 %s dias' %str(stringTimes)   #Variable tipo cadena que indica la saturación y los días de simulación se le agrega stringtimes 
      stringC1Save = 'c1 %s dias.png' %str(stringTimes) #Variable tipo cadena que indica la saturación y los días de simulación se le agrega stringtimes servirá para salvar las imagenes en formato png
      stringC2 = 'c2 %s dias' %str(stringTimes)   #Variable tipo cadena que indica la saturación y los días de simulación se le agrega stringtimes 
      stringC2Save = 'c2 %s dias.png' %str(stringTimes) #Variable tipo cadena que indica la saturación y los días de simulación se le agrega stringtimes servirá para salvar las imagenes en formato png
      stringR = 'R %s dias' %str(stringTimes)   #Variable tipo cadena que indica la saturación y los días de simulación se le agrega stringtimes 
      stringRSave = 'R %s dias.png' %str(stringTimes) #Variable tipo cadena que indica la saturación y los días de simulación se le agrega stringtimes servirá para salvar las imagenes en formato png
      
      plt.figure('c1', figsize=(12,4))     #Instrucciones para graficar el perfil de presión vs centros de las celdas
      plt.title(stringC1, fontsize=16)
      plt.contourf(xg,yg, c1_v.transpose(),200,alpha=.75)
      plt.colorbar()
      #C = pl.contour(xg,yg,Sw.transpose(),10, colors='black', linewidths =1.0)
      #pl.clabel(C,inline=1,fontsize=8)
      plt.xlabel('$ x(m) $'); plt.ylabel('$ y(m)$'); plt.grid()
      plt.savefig(stringC1Save, dpi=300)
      
      imgC1 = Image.open(stringC1Save)    #Se abre el archivo de imagen de la saturación recien guardado
      framesC1.append(imgC1)            #El archivo abierto de imagen de anexa a la lista de frames

   
      plt.figure('c2', figsize=(12,4))     #Instrucciones para graficar el perfil de presión vs centros de las celdas
      plt.title(stringC2, fontsize=16)
      plt.contourf(xg,yg, c2_v.transpose(),200,alpha=.75)
      plt.colorbar()
      #C = pl.contour(xg,yg,Sw.transpose(),10, colors='black', linewidths =1.0)
      #pl.clabel(C,inline=1,fontsize=8)
      plt.xlabel('$ x(m) $'); plt.ylabel('$ y(m)$'); plt.grid()
      plt.savefig(stringC2Save, dpi=300)
      
      imgC2 = Image.open(stringC2Save)    #Se abre el archivo de imagen de la saturación recien guardado
      framesC2.append(imgC2)            #El archivo abierto de imagen de anexa a la lista de frames


      plt.figure('R', figsize=(12,4))     #Instrucciones para graficar el perfil de presión vs centros de las celdas
      plt.title(stringR, fontsize=16)
      plt.contourf(xg,yg, r1.transpose(),200,alpha=.75)
      plt.colorbar()
      #C = pl.contour(xg,yg,Sw.transpose(),10, colors='black', linewidths =1.0)
      #pl.clabel(C,inline=1,fontsize=8)
      plt.xlabel('$ x(m) $'); plt.ylabel('$ y(m)$'); plt.grid()
      plt.savefig(stringRSave, dpi=300)
      
      imgR = Image.open(stringRSave)    #Se abre el archivo de imagen de la saturación recien guardado
      framesR1.append(imgR)            #El archivo abierto de imagen de anexa a la lista de frames
      
      plt.show()



      count_save+=1



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



framesC1[0].save('c1_2D.gif', format='GIF', append_images=framesC1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
framesC2[0].save('c2_2D.gif', format='GIF', append_images=framesC2[1:], save_all=True, duration=200, loop=0)   #Crea el gif de la saturación  con las imagenes guardadas en la lista de frames
framesR1[0].save('R_2D.gif', format='GIF', append_images=framesR1[1:], save_all=True, duration=200, loop=0)   #Crea el gif de la saturación  con las imagenes guardadas en la lista de frames





#------------------------------------------------------------------------------------------#
#--------------------------- Graficando todos los resultados ------------------------------#
#------------------------------------------------------------------------------------------#

                                                                   
plt.figure('Graf_c1',figsize=(9,8))
plt.style.use('fast')
plt.minorticks_on()
plt.title('Perfil de $c_1$ vs distancia en el tiempo')
for i in range(0, nstp+1):
    
    plt.plot(xc, solcnsImp_c1[i][:]) 
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
    
    plt.plot(xc, solcnsImp_c2[i][:]) 
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
    
    plt.plot(xc, solcnsImp_r1[i][:]) 
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
# tsel=10   #tiempo seleccionado de hoja excel


# plt.figure('comparativa_C1',figsize=(9,8))
# WMA_I=pd.read_excel('comparativa_02.xlsx', sheet_name='c_1')     #Crea un data frame llamado WMA_I a partir de la tabla de excel
# data_set_C1 = np.transpose(WMA_I.iloc[11:,5:95].to_numpy())
# # data_set = WMA_I.iloc[11:,5:95].to_numpy()
# #WMA_I
# plt.plot(xc, solcnsImp_c1[0][:], "-r"  , label="Condicion inicial")
# plt.plot(xc, solcnsImp_c1[tsel][:], "-b*", label="Tiempo de simulacion "+str(round(tsel,2))+" dias este código")
# plt.plot(xc, data_set_C1[tsel][:], "-.m", label="Tiempo de simulacion "+str(round(tsel,2))+" dias WMA_I" )
# plt.xlabel("Distancia x [$m$]"), plt.ylabel("$c_1 [mgr/cm^{3}]$")
# plt.legend(loc=0); plt.minorticks_on()
# plt.grid(True,which='major', color='w', linestyle='-')
# plt.grid(True, which='minor', color='g', linestyle='--')


# Res_contrsns_ini = []
# Res_contrsns_fin = []
# file_name='gypsum_eq.out'
# Res_contrsns_ini, Res_contrsns_fin = rF.import_results(file_name)

# plt.figure('comparativa_C2',figsize=(9,8))
# WMA_I=pd.read_excel('comparativa_02.xlsx', sheet_name='c_2')     #Crea un data frame llamado WMA_I a partir de la tabla de excel
# data_set_C2 = np.transpose(WMA_I.iloc[11:,5:95].to_numpy())
# plt.plot(xc, solcnsImp_c2[0][:], "-r"  , label="Condicion inicial")
# plt.plot(xc, solcnsImp_c2[tsel][:], "-b*", label="Tiempo de simulacion "+str(round(tsel,2))+" dias este código")
# plt.plot(xc, data_set_C2[tsel][:], "-.m", label="Tiempo de simulacion "+str(round(tsel,2))+" dias WMA_I" )
# plt.plot(xc, Res_contrsns_fin[1][:], "g--", label="Fortrab so4-2 ")
# plt.xlabel("Distancia x [$m$]"), plt.ylabel("$c_2 [mgr/cm^{3}]$")
# plt.legend(loc=0); plt.minorticks_on()
# plt.grid(True,which='major', color='w', linestyle='-')
# plt.grid(True, which='minor', color='g', linestyle='--')


# framesU1[0].save('U1.gif', format='GIF', append_images=framesU1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
# framesC1[0].save('C1.gif', format='GIF', append_images=framesC1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
# framesC2[0].save('C2.gif', format='GIF', append_images=framesC2[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames
# framesR1[0].save('R1.gif', format='GIF', append_images=framesR1[1:], save_all=True, duration=200, loop=0)  #Crea el gif de la presión con las imagenes guardadas en la lista de frames

