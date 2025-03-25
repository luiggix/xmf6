# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:23:03 2024

@author: USUARIO FINAL
"""

#------------------------------------------------------------------------------------------#
#---------------------------------- Importing libraries -----------------------------------#
#------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

#------------------------------------------------------------------------------------------#
#------------------------------------ Importing results -----------------------------------#
#------------------------------------------------------------------------------------------#

# .out file path

def import_results(file_name):

    # Initializing variables
    integration_method = ""
    number_of_targets = 0
    time_step = 0.0
    final_time = 0.0
    matrix_data = []
    mtrz_contrsns_ini = []
    mtrz_contrsns_fin = []
    
    # Reading the .out file and importing data
    with open(file_name, 'r') as file:
        for line in file:
            
            # Clearing blank lines before and after text
            clean_line = line.strip()
            
            # Searching for specific information
            if "Integration method" in clean_line:
                integration_method = clean_line.split(":")[-1].strip()
            elif "Number of targets" in clean_line:
                number_of_targets = int(clean_line.split(":")[-1].strip())
            elif "Time step" in clean_line:
                next(file)
                time_step = float(next(file).strip())
            elif "Final time" in clean_line:
                next(file)
                final_time = float(next(file).strip())
            elif "Dimension + Mixing ratios (by rows)" in clean_line:
                next(file)
                
                # Starting to save the results matrix
                for line in file:
                    
                    # If the line is not empty
                    if line.strip():  
                        numbers = list(map(float, line.split()))
                        matrix_data.append(numbers)
                    
                    # Exit loop if an empty line is found
                    else:
                        break
                    
            elif "Initial concentration of aqueous species" in clean_line:
                next(file)
                
                # Starting to save the results matrix
                for line in file:
                    
                    # If the line is not empty
                    if line.strip():  
                        numbers = list(map(float, line.split()))
                        mtrz_contrsns_ini.append(numbers)
                    
                    # Exit loop if an empty line is found
                    else:
                        break
                    
            elif "Final concentration of aqueous species" in clean_line:
                next(file)
                
                # Starting to save the results matrix
                for line in file:
                    
                    # If the line is not empty
                    if line.strip():  
                        numbers = list(map(float, line.split()))
                        mtrz_contrsns_fin.append(numbers)
                    
                    # Exit loop if an empty line is found
                    else:
                        break
    return (mtrz_contrsns_ini, mtrz_contrsns_fin)

# #------------------------------------------------------------------------------------------#
# #------------------------------------ Plotting results ------------------------------------#
# #------------------------------------------------------------------------------------------#

# # Creating the vector in “x”
# x_len = int(matrix_data[0][0])
# x = np.linspace(1, x_len, x_len)

# # Creating a temporary folder to save images at each time step
# if not os.path.exists('temp_images'):
#     os.makedirs('temp_images')

# # Saving images at each time step
# for i in range(number_of_targets):
#     plt.figure(figsize=(14, 14))
#     plt.style.use('fast')
#     plt.minorticks_on()
#     plt.title(f"Results for time step {i + 1}", fontsize=27)
#     plt.ylim(0, 0.7)
#     plt.plot(x, matrix_data[i][1:], "o-", label=f'Time step {i + 1}')
#     plt.xlabel("length [m]", fontsize=21)
#     plt.ylabel('Concentrations []', fontsize=21)
#     plt.xticks(fontsize=21)
#     plt.yticks(fontsize=21)
#     plt.ticklabel_format(style="plain")
#     plt.grid(True, which='major', color='k', linestyle='-', alpha=0.2)
#     plt.grid(True, which='minor', color='k', linestyle='--', alpha=0.2)
#     plt.legend(fontsize=18)

#     # Saving the graphic as a .jpg file
#     plt.savefig(f'temp_images/Resultados_{i + 1}.jpg', format='jpg', dpi=100)
#     plt.close()

# # Creating a .gif file
# images = []
# for i in range(number_of_targets):
#     img = Image.open(f'temp_images/Resultados_{i + 1}.jpg')
#     images.append(img)

# # Saving the .gif file
# images[0].save('Resultados.gif',
#                 save_all=True,
#                 append_images=images[1:],
#                 duration=500,
#                 loop=0)

# # Deleting temporary images
# for i in range(number_of_targets):
#     os.remove(f'temp_images/Resultados_{i + 1}.jpg')
# os.rmdir('temp_images')

# "Graficando propiedades inicales del SO4-2"
# x_len = number_of_targets
# x = np.linspace(1, x_len, x_len)
# plt.figure('Fig_1',figsize=(14, 14))
# plt.style.use('fast')
# plt.minorticks_on()
# plt.title(f"Resultados iniciales", fontsize=27)
# # plt.ylim(0, 0.7)
# # for i in range(0, 3):
# #     if i == 0:
# #         labl = "ca+2"
# #     elif i == 1:
# #         labl = "so4-2 "
# #     elif i == 2:
# #         labl = "h2o "
# #     plt.plot(x, mtrz_contrsns_ini[i][:], "o-", label=labl)
# plt.plot(x, mtrz_contrsns_ini[1][:], "o-", label="so4-2 ")
# plt.xlabel("length [m]", fontsize=21)
# plt.ylabel('Concentrations []', fontsize=21)
# plt.xticks(fontsize=21)
# plt.yticks(fontsize=21)
# plt.ticklabel_format(style="plain")
# plt.grid(True, which='major', color='k', linestyle='-', alpha=0.2)
# plt.grid(True, which='minor', color='k', linestyle='--', alpha=0.2)
# plt.legend(fontsize=18)

# "Graficando propiedades finales del SO4-2"
# x_len = number_of_targets
# x = np.linspace(1, x_len, x_len)
# plt.figure('Fig_2',figsize=(14, 14))
# plt.style.use('fast')
# plt.minorticks_on()
# plt.title(f"Resultados finales", fontsize=27)
# # plt.ylim(0, 0.7)
# # for i in range(0, 3):
# #     if i == 0:
# #         labl = "ca+2"
# #     elif i == 1:
# #         labl = "so4-2 "
# #     elif i == 2:
# #         labl = "h2o "
# #     plt.plot(x, mtrz_contrsns_fin[i][:], "o-", label=labl)
# plt.plot(x, mtrz_contrsns_fin[1][:], "o-", label="so4-2 ")
# plt.xlabel("length [m]", fontsize=21)
# plt.ylabel('Concentrations []', fontsize=21)
# plt.xticks(fontsize=21)
# plt.yticks(fontsize=21)
# plt.ticklabel_format(style="plain")
# plt.grid(True, which='major', color='k', linestyle='-', alpha=0.2)
# plt.grid(True, which='minor', color='k', linestyle='--', alpha=0.2)
# plt.legend(fontsize=18)
