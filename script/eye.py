"""
eye.py — COM31006 - Aryan Golbaghi
##################################################
Embed 5x5 binary segment at non-overlapping SIFT points (blue channel LSB).
Verify bit-pattern & homography;
Detect tamper + overlay image.
*Simple Tk GUI.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2 as cv
import numpy as np
import os, sys

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


SEG_SIZE=5
CHANNEL= 0 # blue
META_DATA = None #to store metadata if none will load json

def binarise(img: np.ndarray) -> np.ndarray:
    '''convert a grayscale image to black and white'''
    return (img > 127).astype(np.uint8)

class Embedder:
    '''embeds a watermark into a carrier image.'''
    def __init__(self, carrier: str, watermark: str, max_pts: int = 400):
        self.carrier=carrier
        self.watermark=watermark
        self.max=max_pts
        self.sift=cv.SIFT_create()

    def _points(self, gray_carrier: np.ndarray) -> List[cv.KeyPoint]:
        '''returns the strongest non over-lapping SIFT heypoints'''
        kps, _=self.sift.detectAndCompute(gray_carrier, None)
        occup = np.zeros(gray_carrier.shape,dtype=bool)
        selc=[]
        h=SEG_SIZE//2
        for kp in sorted(kps,key=lambda k:-k.response):
            x,y = map(int,map(round,kp.pt))
            if x-h<0 or y-h<0 or x+h>=gray_carrier.shape[1] or y+h>=gray_carrier.shape[0]: continue
            if occup[y-h:y+h+1,x-h:x+h+1].any(): continue
            occup[y-h:y+h+1,x-h:x+h+1]=True
            selc.append(kp)
            if len(selc)>=self.max: break
        return selc

    def embed(self)->Dict[str,str]:
        '''
        embed the black and white watermark into the least significant bits (LSBs) of the carrier image at selected SIFT keypoints.
        returns the output file names of the modifed image and the metadata.json.'''
        col=cv.imread(self.carrier)
        gray_carrier=cv.cvtColor(col,cv.COLOR_BGR2GRAY)
        kps=self._points(gray_carrier)
        segment=cv.resize(binarise(cv.imread(self.watermark,cv.IMREAD_GRAYSCALE)),(SEG_SIZE,SEG_SIZE),cv.INTER_NEAREST)
        half=SEG_SIZE//2
        meta={"channel":CHANNEL,"segment":segment.tolist(),"keypoints":[]}
        out=col.copy()
        for kp in kps:
            x,y=map(int,map(round,kp.pt))
            for dy in range(-half,half+1):
                for dx in range(-half,half+1):
                    bit=int(segment[dy+half,dx+half])
                    px=out[y+dy,x+dx,CHANNEL]
                    out[y+dy,x+dx,CHANNEL]=(px&~1)|bit
            meta["keypoints"].append({"pt":[float(kp.pt[0]),float(kp.pt[1])],"size":kp.size,"angle":kp.angle})
        base=Path(self.carrier).with_suffix("")
        img=f"{base}_wm.png"
        m=f"{base}_meta.json"
        cv.imwrite(img,out)
        json.dump(meta,open(m,'w'),indent=2)
        global META_DATA
        META_DATA=meta
        return {"img":img,"meta":m}
    
class Verifier:
    '''verify suspected carrier image of carrying a watermark '''
    def __init__(self,suspect:str,meta:str,error_tolerance :float=0.2):
        self.suspect=suspect
        self.meta=json.load(open(meta)) if META_DATA is None else META_DATA;
        self.error_tolerance =error_tolerance 
        self.segment=np.array(self.meta['segment'],np.uint8)
        self.color_chan=self.meta['channel']    
        self.sift=cv.SIFT_create()
    def verify(self)->Tuple[bool,List[Tuple[int,int]],float]:
        '''
        check how many embedded watermark segments match the original bits.
        calculate how well the image has been geometrically preserved, e.g > 60% is ok.
        decide whether the watermark is still valid or likely tampered with.
        '''
        col=cv.imread(self.suspect)
        gray_carrier=cv.cvtColor(col,cv.COLOR_BGR2GRAY)
        kps,_=self.sift.detectAndCompute(gray_carrier,None)
        mism,src,dst=[],[],[]
        h=SEG_SIZE//2
        for ref in self.meta['keypoints']:
            ref_pt=np.array(ref['pt'])
            idx=int(np.argmin([np.linalg.norm(np.array(k.pt)-ref_pt) for k in kps])) if kps else -1
            if idx==-1: continue
            kp=kps[idx]
            x,y=map(int,map(round,kp.pt))
            if x-h<0 or y-h<0 or x+h>=col.shape[1] or y+h>=col.shape[0]: continue
            patch=(col[y-h:y+h+1,x-h:x+h+1,self.color_chan]&1)
            if np.count_nonzero(patch ^ self.segment)/(SEG_SIZE*SEG_SIZE)>self.error_tolerance : mism.append((x,y))
            src.append(ref_pt)
            dst.append(np.array(kp.pt))
        inl=1.0
        if len(src)>=4:
            _,mask=cv.findHomography(np.array(src),np.array(dst),cv.RANSAC,ransacReprojThreshold=5.0)
            if mask is not None: inl=mask.sum()/mask.size
        auth=len(mism)<=int(len(src)*self.error_tolerance ) and inl>0.6
        return auth,mism,inl
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
        if mism:
            img=cv.imread(self.suspect)
            [cv.circle(img,(x,y),8,(0,0,255),2) for x,y in mism]
            overlay_path = Path(self.suspect).with_name(f"{Path(self.suspect).stem}_overlay{Path(self.suspect).suffix}")
            cv.imwrite(str(overlay_path),img)
        return {"tampered":not auth,"mismatches":len(mism),"inlier":round(inl,3),"overlay":str(overlay_path)}


class InterFace:
    '''simple GUI for with three buttons'''
    def __init__(self,root:tk.Tk):
        self.root=root
        root.title('Detecot Eye')
        self.root.minsize(1000,600)
        self._in_photo=None
        self._out_photo=None
        prv=tk.Frame(root)
        prv.pack(fill=tk.BOTH,expand=True,padx=20,pady=20)
        self.in_label=tk.Label(prv,text='Selected image\n(appears here)',compound=tk.TOP,justify=tk.CENTER)
        self.out_label=tk.Label(prv,text='Generated image\n(appears here)',compound=tk.TOP,justify=tk.CENTER)
        self.in_label.grid(row=0,column=0,sticky='nsew',padx=10)
        self.out_label.grid(row=0, column=1,sticky='nsew',padx=10)
        prv.columnconfigure(0, weight=1)
        prv.columnconfigure(1, weight=1)
        prv.rowconfigure(0,  weight=1)
        for text,fun in [('Embed',self.embed),('Verify',self.verify),('Detect',self.detect)]:
            tk.Button(root,text=text,width=22,command=fun).pack(padx=130,side=tk.LEFT)
    def pick(self,title,typ='Image'):
        path=filedialog.askopenfilename(title=title,filetypes=[(f'{typ} files','*.png;*.tif')]) #only tif and png files are allowed
        if path and typ=='Image':self._show_image(path,type='in')
        return path
    def meta(self):
        return filedialog.askopenfilename(title='Meta JSON',filetypes=[('JSON','*.json')]) if META_DATA is None else META_DATA #use json if metatdata is not preseverd in the memory
    def embed(self):
        carrier=self.pick('Carrier')
        watermark=self.pick('Watermark')#
        if carrier and watermark:
            try:
                emb=Embedder(carrier,watermark).embed()
                self._show_image(emb['img'],type='out')
                messagebox.showinfo('Embed',f"Watermarked: {emb['img']}\nMeta: {emb['meta']}")
            except Exception as e:
                messagebox.showerror('Error',str(e))
    def verify(self):
        sus=self.pick('Suspect')
        meta=self.meta()
        if sus and meta:
            try:
                auth,_,inl=Verifier(sus,meta).verify()
                messagebox.showinfo('Verify','AUTHENTIC' if auth else 'TAMPERED'+f"\nInliers:{inl:.2f}")
            except Exception as e:
                messagebox.showerror('Error',str(e))
    def detect(self):
        sus=self.pick('Suspect')
        meta=self.meta()
        if sus and meta:
            try: 
                res=Detector(sus,meta).detect()
                self._show_image(res['overlay'],type='out')
                messagebox.showinfo('Detect',json.dumps(res,indent=2))
            except Exception as e:
                messagebox.showerror('Error',str(e))
    def _show_image(self,src:str,type:str='in')->None:
        try:
            img=Image.open(src)
            img.thumbnail((600,600))
            photo=ImageTk.PhotoImage(img)
            if type=='in':
                self._in_photo=photo
                self.in_label.config(image=photo,text="")
            else:
                self._out_photo=photo
                self.out_label.config(image=photo,text="")
        except Exception as e:
            messagebox.showerror('Error',str(e))
    
    def run(self):
        self.root.mainloop()


if __name__=='__main__':
    InterFace(tk.Tk()).run()
    