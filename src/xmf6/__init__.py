# Módulos que se van a integrar en xmf6
from .mesh import MeshDis
from .osys import nice_print, info_array, OFiles, OSPar
from .physpar import PhysPar
from .tdis import TDis
from .vis import plot, scatter
import xmf6.common
import xmf6.gwf
import xmf6.gwt
import xmf6.api

__all__ = ["MeshDis", 
           "nice_print", "info_array", "OFiles", "OSPar",
           "PhysPar", "TDis", "plot", "scatter",
           "common", "gwf", "gwt", "api"
           ]