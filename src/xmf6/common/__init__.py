# Desde el .[nombre del archivo] importa [lista de funciones] 
from .common import init_sim, set_par, set_obs

# que nombres se exportan si se usa from xmf6.gwf import * 
__all__ = ["init_sim", "set_par", "set_obs"]