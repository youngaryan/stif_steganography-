"""
eye.py — COM31006 - Aryan Golbaghi
##################################################
Embed 9x9 binary segment at non-overlapping SIFT points (blue channel LSB).
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
from tkinter import filedialog,messagebox,ttk
from PIL import Image,ImageTk
###Golbal VArables###
SEG_SIZE=9 #segment size for the watermark
FLANN_INDEX_KDTREE=1 #opencv code for kd-tree in FLANN
#### HELPER FUNCTIONS####
def make_dir(path_n_fldr:str="res",base:str="che",typ="modifed",ext:str=".png")->str:
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
###Embeder class####
class Embedder:
    '''embeds a watermark into a carrier image.'''
    def __init__(self,carrier:str,watermark:str,max_pts:int=400,seg_size:int=SEG_SIZE):
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
        self._auth_result=len(mism)<=int(len(src)*self.error_tolerance ) and inl>0.6
        return self._auth_result,mism,inl
    def extract_watermark(self, upscale: bool = True) -> np.ndarray|None:
        """
        reconstruct the embedded 9x9 watermark pattern by majority-voting
        across all matched key-points.
        """
        if self._auth_result is None:self.verify()
        if not self._auth_result:raise RuntimeError(f"{self.suspect} is not verifed to have the specific watermark asosiated with the chosen metadata.")
        col=cv.imread(self.suspect)
        if col is None:
            raise FileNotFoundError(self.suspect)
        gray=cv.cvtColor(col, cv.COLOR_BGR2GRAY)
        kps,desc=self.sift.detectAndCompute(gray, None)
        if kps is None or desc is None:return None
        ref_desc=np.array([kp["desc"] for kp in self.meta["keypoints"]],dtype=np.float32)
        ref_channels=[kp["channel"] for kp in self.meta["keypoints"]]
        match_res=self._match_desc(ref=ref_desc,sus=desc)
        half=self.seg_size//2
        patches:List[np.ndarray]=[]
        for chanel,matc_idx,in zip(ref_channels,match_res):
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
        if upscale:
            recovered=cv.resize(recovered,(self.seg_size*3,self.seg_size*3),cv.INTER_NEAREST)
        return recovered
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
###GUI class####
class InterFace:
    '''simple GUI for with three buttons'''
    def __init__(self,root:tk.Tk):
        self.root=root
        root.title('Detecot Eye')
        self.root.minsize(1000,600)
        self.progress=ttk.Progressbar(root,orient="horizontal",mode="determinate",length=400)
        self.progress.pack(pady=(5,2))
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
        for text,fun in [('Embed',self.embed),('Verify',self.verify),('Detect',self.detect),('Recover',self.recover)]:
            ttk.Button(root,text=text,width=20,command=fun,).pack(padx=65,side=tk.LEFT,)
        self.seg_var=tk.IntVar(value=SEG_SIZE)
        seg_frame=tk.Frame(self.root)
        seg_frame.pack(pady=10)

        tk.Label(seg_frame, text="Segment size (px)").grid(row=0, column=0, sticky="w")
        tk.Spinbox(seg_frame, from_=1, to=21, increment=2, textvariable=self.seg_var, width=4).grid(row=0, column=1)#avoid even seg numbers

    def pick(self,title,typ='Image'):
        path=filedialog.askopenfilename(title=title,filetypes=[(f'{typ} files','*.png;*.tif')]) #only tif and png files are allowed
        if path and typ=='Image':self._show_image(path,type='in')
        return path
    def meta(self):
        return filedialog.askopenfilename(title='Meta JSON',filetypes=[('JSON','*.json')])
    def embed(self):
        carrier=self.pick('Carrier')
        watermark=self.pick('Watermark')#
        if carrier and watermark:
            try:
                self.set_progress(10,"Starting embedding...")
                emb=Embedder(carrier,watermark,seg_size=int(self.seg_var.get())).embed()
                self.set_progress(60,"Watermark embedded...")
                self._show_image(emb['img'],type='out')
                self.set_progress(100,"Completed.")
                messagebox.showinfo('Embed',f"Watermarked: {emb['img']}\nMeta: {emb['meta']}")
            except Exception as e:
                messagebox.showerror('Error',str(e))
            self.root.after(1000,lambda:self.set_progress(0))
    def verify(self):
        sus=self.pick('Suspect')
        meta=self.meta()
        if sus and meta:
            try:
                self.set_progress(20,"Starting verifying...")
                auth,_,inl=Verifier(sus,meta).verify()
                self.set_progress(100,"Verification complete.")
                messagebox.showinfo('Verify','AUTHENTIC' if auth else 'TAMPERED'+f"\nInliers:{inl:.2f}")
            except Exception as e:
                messagebox.showerror('Error',str(e))
            self.root.after(1000,lambda:self.set_progress(0))
    def detect(self):
        sus=self.pick('Suspect')
        meta=self.meta()
        if sus and meta:
            try:
                self.set_progress(10,"Analyzing image...")
                res=Detector(sus,meta).detect()
                self.set_progress(80,"Rendering overlay...")
                self._show_image(res['overlay'],type='out')
                self.set_progress(100,"Detection complete.")
                messagebox.showinfo('Detect',json.dumps(res,indent=2))
            except Exception as e:
                messagebox.showerror('Error',str(e))
            self.root.after(1000,lambda:self.set_progress(0))
    def recover(self):
        sus=self.pick('Suspect')
        meta=self.meta()
        if sus and meta:
            try:
                self.set_progress(10,"Recovering watermark...")
                ver=Verifier(sus,meta)
                wm=ver.extract_watermark(upscale=True)
                if wm is None:
                    self.set_progress(0)    
                    messagebox.showwarning('Recover','No watermark patches could be recovered.')
                    return
                base=Path(sus).stem
                tmp=make_dir(base=base,typ="recovered_wm",ext=".png")
                cv.imwrite(str(tmp),wm)
                self.set_progress(100, "Watermark recovered.")
                self._show_image(str(tmp),type='out')
                messagebox.showinfo('Recover',f"Recovered watermark saved to:\n{tmp}")
            except Exception as e:
                messagebox.showerror('Error',str(e))
            self.root.after(1000, lambda: self.set_progress(0))
    def _show_image(self,src:str,type:str='in')->None:
        if src is None or src=='':
            return
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
    def set_progress(self,value:int,text:str=""):
        self.progress['value']=value
        self.root.update_idletasks()
    def run(self):
        self.root.mainloop()
###MAIN class####
if __name__=='__main__':
    InterFace(tk.Tk()).run()