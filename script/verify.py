"""
verify.py — COM31006 - Aryan Golbaghi
##################################################
-verifys the authenticity of a suspected image by matching SIFT descriptors with those stored in metadata.
- estimates geometric consistency using homography estimation and inlier ratios.
- Detectes tampered regions by comparing embedded watermark bits against an expected segment.
- Recovers the original watermark using a majority vote across valid matches.
"""

import json
from typing import List, Tuple
import cv2 as cv
import numpy as np


###Golbal VArables###
FLANN_INDEX_KDTREE=1 #opencv code for kd-tree in FLANN

###Verifier class####
class Verifier:
    '''verify suspected carrier image of carrying a watermark '''
    def __init__(self,suspect:str,meta:str,error_tolerance :float=0.1):
        self.suspect=suspect
        self.meta=json.load(open(meta));
        self.error_tolerance =error_tolerance 
        self.segment=np.array(self.meta['segment'],np.uint8)
        self.seg_size:int=self.meta['seg_size']
        self.sift=cv.SIFT_create()
        self._auth_result=None

    def _match_desc(self,ref:np.ndarray,sus:np.ndarray)->List[int|None]:
        if sus is None or len(sus)<1:
            return [None]*len(ref)
        
        matcher=cv.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5),dict(checks=32))
        if ref.shape[1]!=sus.shape[1]:raise RuntimeError(f"reference size: {ref.shape[1]}, sus size: {sus.shape[1]}, don't match")
        matches=matcher.knnMatch(ref,sus,k=2)
        res:List[int|None]=[]

        for p in matches:
            if len(p)<2:
                res.append(None)
                continue
            m,n=p
            res.append(m.trainIdx if m.distance<0.7*n.distance else None)

        return res
    
    def verify(self)->Tuple[bool,List[Tuple[int,int]],float]:
        '''
        check how many embedded watermark segments match the original bits.
        calculate how well the image has been geometrically preserved, e.g > 60% is ok.
        decide whether the watermark is still valid or likely tampered with.
        if no refefrence point is found will return tmepered true, mismatche=-1, and inl=-1
        '''
        col=cv.imread(self.suspect)
        if col is None:raise FileNotFoundError(self.suspect)
        
        gray_carrier=cv.cvtColor(col,cv.COLOR_BGR2GRAY)
        kps,desc=self.sift.detectAndCompute(gray_carrier,None)
        
        if kps is None or desc is None:kps,desc=[],None
        
        ref_pts=np.array([kp['pt']for kp in self.meta["keypoints"]],dtype=np.float32)
        ref_desc=np.array([kp["desc"] for kp in self.meta["keypoints"]],dtype=np.float32)
        ref_channels=[kp["channel"] for kp in self.meta["keypoints"]]
        match_res=self._match_desc(ref=ref_desc,sus=desc)
        mism:List[Tuple[int,int]]=[]
        src,dst=[],[]
        h=self.seg_size//2
        
        for ref_p,chanel,matc_idx in zip(ref_pts,ref_channels,match_res):
            if matc_idx is None:continue #the point is lost after editing
            
            kp=kps[matc_idx]
            x,y=map(int,map(round,kp.pt))
            if x-h<0 or y-h<0 or x+h>=col.shape[1] or y+h>=col.shape[0]:continue
            
            patch=(col[y-h:y+h+1,x-h:x+h+1,chanel]&1)
            if np.count_nonzero(patch^self.segment)/(self.seg_size*self.seg_size)>self.error_tolerance:mism.append((x,y))
            
            src.append(ref_p)
            dst.append(kp.pt)
        inl=1.0
        
        if len(src)>=4:
            _,mask=cv.findHomography(np.array(src),np.array(dst),cv.RANSAC,ransacReprojThreshold=5.0)
            inl=mask.sum()/mask.size if mask is not None else 0
        elif len(src)==0:return True,-1,-1
        self._auth_result=len(mism)<=int(len(src)*self.error_tolerance ) and inl>0.6
        
        return self._auth_result,mism,inl
    
    def extract_watermark(self) -> np.ndarray|None:
        """
        reconstruct the embedded 9x9 watermark pattern by majority-voting
        across all matched key-points.
        """
        if self._auth_result is None:self.verify()
        if not self._auth_result:raise RuntimeError(f"{self.suspect} is not verifed to have the specific watermark asosiated with the chosen metadata.")
        
        col=cv.imread(self.suspect)
        if col is None: raise FileNotFoundError(self.suspect)
        gray=cv.cvtColor(col, cv.COLOR_BGR2GRAY)
        kps,desc=self.sift.detectAndCompute(gray, None)
        
        if kps is None or desc is None:return None
        
        ref_desc=np.array([kp["desc"] for kp in self.meta["keypoints"]],dtype=np.float32)
        ref_channels=[kp["channel"] for kp in self.meta["keypoints"]]
        match_res=self._match_desc(ref=ref_desc,sus=desc)
        half=self.seg_size//2
        patches:List[np.ndarray]=[]
        
        for chanel,matc_idx in zip(ref_channels,match_res):
            if matc_idx is None:continue #the point is lost after editing
           
            kp=kps[matc_idx]
            x,y=map(int,map(round,kp.pt))
            if (x-half<0 or y-half<0 or
                x+half>=col.shape[1] or y+half>=col.shape[0]):continue
            
            patch=(col[y-half:y+half+1,x-half:x+half+1,chanel]&1)
            patches.append(patch)
        if not patches:return None
        
        stack=np.stack(patches,axis=0)
        votes=(stack.sum(axis=0)>(len(patches)/2))
        recovered=votes.astype(np.uint8)*255
        
        if self.seg_size<20:
            recovered=cv.resize(recovered,(self.seg_size*3,self.seg_size*3),cv.INTER_NEAREST)
        
        return recovered