import tkinter as tk
from tkinter import filedialog
from embeder import WatermarkEmbedder
from temper_detector import TemperDetector


class InterFace(tk.Tk):
    def __init__(self, screenName = None, baseName = None, className = "Tk", useTk = True, sync = False, use = None):
        super().__init__(screenName, baseName, className, useTk, sync, use)
        self.title("watermark embeder (COM3001)")
        tk.Button(self,text="Embed", command=self.embed).grid(row=0,column=0)
        tk.Button(self,text="verify", command=self.verify).grid(row=0,column=1)
        tk.Button(self,text="detect temper", command=self.temper_detect).grid(row=0,column=2)
        self.embeder:WatermarkEmbedder=None

    
    def _load_paths(self):
        carrier =filedialog.askopenfilename(title="choose carrier image")
        wm= filedialog.askopenfilename(title="choose watermark image ")
        return carrier, wm
    
    def embed(self):
        carrier, wm = self._load_paths()
        #TODO need to change embeder class maybe to only apply the watermark when the function is called
        self.embeder = WatermarkEmbedder(carrier_image_path=carrier, watermark_image_path=wm, segment_size=5)
        self.embeder.embed()
        tk.messagebox.showinfo("Done", f"Saved at {self.embeder.modified_carrier_image_path}")

    def verify(self):
        if not self.embeder:
            tk.messagebox.showinfo("Error", "you haven't embeded any image yet, use the embed button")
            return
        path_to_var = filedialog.askopenfilename()

        check = self.embeder.varify_watermark(img_path=path_to_var)
        print("varifed" if check else "not varifed!")


    def temper_detect(self):
        if not self.embeder:
            tk.messagebox.showinfo("Error","you haven't embeded any image yet, use the embed button")
            return
        
        path_to_var = filedialog.askopenfilename()

        td = TemperDetector(embeder=self.embeder)
        check = td.detect_temper(path_to_var)
        print("temperd!" if check["tampered"] else "not tampered")


