# Desde el .[nombre del archivo] importa [lista de funciones] 
from .gwf import initialize, build, get_head, get_specific_discharge

# que nombres se exportan si se usa from macti.visual import * 
__all__ = ["initialize", "build", "get_head", "get_specific_discharge"]