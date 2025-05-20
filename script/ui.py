"""
ui.py — COM31006 - Aryan Golbaghi
##################################################
Simple gui built with tkinter.
provide interface for embeding, verifying, temper detection, and recovery watermark
provide utility tools such as progress bar and segment value inputer 
"""


import json
from pathlib import Path
import cv2 as cv

import tkinter as tk
from tkinter import filedialog,messagebox,ttk
from PIL import Image,ImageTk

from embed import Embedder
from verify import Verifier
from detecor import Detector
from helper import make_dir


###Golbal VArables###
DEFAULT_SEG_SIZE=9 #segment size for the watermark

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
        self.seg_var=tk.IntVar(value=DEFAULT_SEG_SIZE)
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
        if carrier is None or carrier=='':return
        watermark=self.pick('Watermark')
        if watermark is None or watermark=='':return
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
        if sus is None or sus=='':return
        meta=self.meta()
        if meta is None or meta=='':return
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
        if sus is None or sus=='':return
        meta=self.meta()
        if meta is None or meta=='':return
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
        if sus is None or sus=='':return
        meta=self.meta()
        if meta is None or meta=='':return
        try:
            self.set_progress(10,"Recovering watermark...")
            ver=Verifier(sus,meta)
            wm=ver.extract_watermark()
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