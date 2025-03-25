import os
from colorama import Fore, Style, Back
import flopy
#from xmf6.mesh import MeshDis
#from xmf6.tdis import TDis

def nice_print(data, message = ''):
    print(Fore.BLUE)
    print(message)
    print('{:^30}'.format(30*'-') + Style.RESET_ALL)

    if isinstance(data, dict):
        for k,v in data.items():
            print('{:>20} = {:<10}'.format(k, v))
    else: #if not isinstance(data, dict):# or isinstance(data, TDis):
        data.print()

class OFiles():
    def __init__(self, os_par, oc_par):
        self.os_par = os_par
        self.oc_par = oc_par

    def get_head(self):
        return flopy.utils.HeadFile(
            os.path.join(self.os_par["ws"], 
                         self.oc_par["head_file"])).get_data()

    def get_bud(self):    
        return flopy.utils.CellBudgetFile(
            os.path.join(self.os_par["ws"], 
                         self.oc_par["fbudget_file"]),
            precision='double')

    def get_spdis(self):
        bud = self.get_bud()
        return bud.get_data(text='DATA-SPDIS')[0]

    def get_q(self, gwf):
        spdis = self.get_spdis()
        return flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    def get_concentration(self, sim, t):
        ucnobj_mf6 = sim.transport.output.concentration()
        simconc = ucnobj_mf6.get_data(totim=t).flatten()
        return simconc

if __name__ == '__main__':
    import mesh
    malla = mesh.MeshDis(
        nrow = 1,    # Number of rows
        ncol = 120,  # Number of columns
        nlay = 1,    # Number of layers
        row_length = 12.0,   # Length of rows
        column_length = 0.1, # Length of columns
        top = 1.0,   # Top of the model
        bottom = 0,  # Layer bottom elevation 
    )
    nice_print(malla.get_dict(), 'Space discretization')

    

