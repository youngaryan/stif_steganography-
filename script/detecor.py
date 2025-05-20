"""
detecor.py — COM31006 - Aryan Golbaghi
##################################################
Detect tamper:bool +  calculates the number of mismatches on the suspected image + overlay temperd part of the suspected image(if any msimatches).
"""

from pathlib import Path
from typing import Dict, Any
import cv2 as cv

from verify import Verifier
from helper import make_dir

###Detector class####
class Detector:
    '''
    detects mismatches on suspected carrier, if mismatches exist it draws a red circle around them. 
    '''
    def __init__(self,suspect:str,meta:str):
        self.suspect=suspect
        self.meta=meta
    def detect(self)->Dict[str,Any]:
        '''
        returns:
        tampered":not auth,
        "mismatches":len(mism)
        ,"inlier":round(inl,3),
        "overlay":str(overlay_path) 
        '''
        auth,mism,inl=Verifier(self.suspect,self.meta).verify()
        overlay_path=""
        if len(mism)>0:
            img=cv.imread(self.suspect)
            [cv.circle(img,(x,y),8,(0,0,255),2) for x,y in mism]
            base=Path(self.suspect).stem
            overlay_path=make_dir(base=base,typ="overlay",ext=".png")
            cv.imwrite(str(overlay_path),img)
        return {"tampered":not auth,"mismatches":len(mism),"inlier":round(inl,3),"overlay":str(overlay_path)}