import tkinter as tk
from tkinter import *

class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tip_window = None
        
    def show_tip(self, tip_text):
        if self.tip_window or not tip_text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x= x + self.widget.winfo_rootx()+25
        y = y + cy + self.widget.winfo_rooty()+25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        
        label = tk.Label(tw, text=tip_text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("tahoma","8","normal"))        
        label.pack(ipadx=1)
        
    def hide_tip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

                
