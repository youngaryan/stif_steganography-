"""
helper.py — COM31006 - Aryan Golbaghi
##################################################
provide two helper function for the rest of the assignments
1-make_dir(path_n_fldr:str,base:str,typ:str,ext:str)->str: which generate a result folder and return a string for file names based on type,base name and extention
2-binarise(img: np.ndarray)->np.ndarray: convert a grayscale image to black and white
"""
import numpy as np
from pathlib import Path


#### HELPER FUNCTIONS####
def make_dir(path_n_fldr:str="res",base:str="che",typ:str="modifed",ext:str=".png")->str:
    '''
    generate folder if it doesn't exsist
    will be used to generate file names depends on and thier base name, type, and extention.
    '''
    ndir=Path(path_n_fldr)
    ndir.mkdir(exist_ok=True)
    file_name=f"{base}_{typ}{ext}"
    
    return str(ndir/file_name)

def binarise(img: np.ndarray)->np.ndarray:
    '''convert a grayscale image to black and white'''
    
    return (img>127).astype(np.uint8)