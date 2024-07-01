import sys, os
import pdb
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
pos='snap'
CURRENT_WP=os.getcwd()
if pos=='snap':
    EXP_PATH = CURRENT_WP
else:
    raise ValueError("Please specify the position of the project directory")