# Desde el .[nombre del archivo] importa [lista de funciones] 
from .gwf import init_sim, set_packages, get_head, get_specific_discharge

# que nombres se exportan si se usa from xmf6.gwf import * 
__all__ = ["init_sim", "set_packages", "get_head", "get_specific_discharge"]