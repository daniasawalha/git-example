# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 08:21:00 2021

@author: Dania
"""
from tkinter import * 
from tkinter import ttk 
root=Tk()
frm=ttk.Frame(root,padding=10)
frm.grid()
ttk.Label(frm,text="hello word").grid(column=0,row=0)
ttk.Button(frm,text="quit",command=root.destroy).grid(column=1,row=1)
root.mainloop()