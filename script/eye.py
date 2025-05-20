"""
eye.py — COM31006 - Aryan Golbaghi
##################################################
Embed 9x9 binary segment (could be changed via gui) at non-overlapping SIFT points (random channel LSB).
Verify bit-pattern & homography;
Detect tamper + overlay image.
*Simple Tk GUI.
"""
import tkinter as tk

from ui import InterFace

###MAIN class####
if __name__=='__main__':
    InterFace(tk.Tk()).run()