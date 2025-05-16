from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple,List

def show_image(image, title="image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def watermark_to_segements(water_mark:np.ndarray=None, segment_szie:int=5) -> Tuple[List[np.ndarray], int, int]:
    '''return lest of segements, and a tuple of row size and clounm size'''
    segments=[]
    height,width =water_mark.shape
    for y in range(0, height, segment_szie):
        for x in range(0, width, segment_szie):
            segment = water_mark[y :y+segment_szie,x: x + segment_szie]
            if segment.shape == (segment_szie, segment_szie):
                segments.append(segment)
    return segments,water_mark.shape[0]//segment_szie,water_mark.shape[1]//segment_szie

def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1
