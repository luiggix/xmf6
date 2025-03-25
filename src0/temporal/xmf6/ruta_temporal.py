import os, sys, platform
c_path = os.getcwd()

sep = "\\" if platform.system() == "Windows" else "/"

l_path = c_path.split(sep=sep)
i_wma = l_path.index('SNN_WMA')
a_path = sep.join(l_path[:i_wma])
src_path = sep.join(['','SNN_WMA','src'])
if not(src_path in sys.path[0]):
    sys.path.insert(0, os.path.abspath(a_path + src_path)) 
