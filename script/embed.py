"""
embed.py — COM31006 - Aryan Golbaghi
##################################################
emmder class for embedding an watermark to a carrier image.
uses Scale-Invariant Feature Transform (SIFT) algorthim to detect keypoints
embeds watermark using Least Significant Bit (LSB) into strongest non-overlapping keypoints
embeding is done in random colour channel of the carrier image
"""

import numpy as np
import cv2 as cv

from typing import List, Tuple, Dict
import json
from pathlib import Path

from script.helper import binarise, make_dir


###Embeder class####
class Embedder:
    '''embeds a watermark into a carrier image.'''
    def __init__(self,carrier:str,watermark:str,max_pts:int=200,seg_size:int=9):
        self.carrier=carrier
        self.watermark=watermark
        self.max=max_pts
        self.sift=cv.SIFT_create()
        self.seg_size=seg_size if seg_size%2!=0 else seg_size+1
    
    def _points(self, gray_carrier: np.ndarray)->Tuple[List[cv.KeyPoint],np.ndarray]:
        '''returns the strongest non over-lapping SIFT heypoints with their descriptiors'''
        kps,desc=self.sift.detectAndCompute(gray_carrier,None)
        if kps is None or desc is None: return [],np.array([])
        
        pairs=sorted(zip(kps,desc),key=lambda k:-k[0].response)
        occup = np.zeros(gray_carrier.shape,dtype=bool)
        selc,selc_descs =[],[]
        h=self.seg_size//2
        
        for kp, de in pairs:
            x,y = map(int,map(round,kp.pt))
            if x-h<0 or y-h<0 or x+h>=gray_carrier.shape[1] or y+h>=gray_carrier.shape[0]:continue
            if occup[y-h:y+h+1,x-h:x+h+1].any():continue
            
            occup[y-h:y+h+1,x-h:x+h+1]=True
            selc.append(kp)
            selc_descs.append(de)
            if len(selc)>=self.max:break
        
        return selc, np.array(selc_descs)
    
    def embed(self)->Dict[str,str]:
        '''
        embed the black and white watermark into the least significant bits (LSBs) of the carrier image at selected SIFT keypoints.
        returns the output file names of the modifed image and the metadata.json.'''
        col=cv.imread(self.carrier)
        if col is None:raise FileNotFoundError(self.carrier)
        
        gray_carrier=cv.cvtColor(col,cv.COLOR_BGR2GRAY)
        kps,desc=self._points(gray_carrier)
        segment=cv.resize(binarise(cv.imread(self.watermark,cv.IMREAD_GRAYSCALE)),(self.seg_size,self.seg_size),cv.INTER_NEAREST)
        half=self.seg_size//2
        meta={"segment":segment.tolist(),"keypoints":[],"seg_size":self.seg_size}
        out=col.copy()
        
        for kp, de in zip(kps,desc):
            ch=np.random.choice(3)
            x,y=map(int,map(round,kp.pt))
            for dy in range(-half,half+1):
                for dx in range(-half,half+1):
                    if dy+half>=segment.shape[1] or dx+half>=segment.shape[0] or dy+half<0 or dx+half<0:continue
                    bit=int(segment[dy+half,dx+half])
                    px=out[y+dy,x+dx,ch]#choosing random channel colour to avoid single channel attack
                    out[y+dy,x+dx,ch]=(px&~1)|bit
            
            meta["keypoints"].append({"pt":[float(kp.pt[0]),float(kp.pt[1])],'channel':ch,"desc":de.astype(np.float32).tolist(),})
        
        base=Path(self.carrier).stem
        img=make_dir(base=base,typ="modified",ext=".png")
        m=make_dir(base=base,typ="meta",ext=".json")
        cv.imwrite(img,out)
        json.dump(meta,open(m,'w'),indent=2)
        
        return {"img":img,"meta":m}