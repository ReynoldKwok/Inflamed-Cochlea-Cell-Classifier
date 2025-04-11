from datetime import datetime
from tkinter import filedialog, ttk
from typing import Literal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import tkinter as tk
import DataProcessing as dp
import os
import time
import csv


def title(plane=None, r=0, c=0, rspan=1, cspan=3):
    head = tk.Label(plane, text="Inflamed Cochlear Cell Classifier", font=('Arial', 20))
    head.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=5, sticky='nsew')


def frame_config(plane=None, r=0, c=2, w=10, rspan=1, cspan=1):
    frame = tk.Frame(plane, bg='light grey', width=w, name="frame")
    frame.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=(10, 5), sticky='nsew')
    return frame


def content_frame(plane=None, text=None, r=0, c=0, rspan=None, cspan=2, page=0):
    if rspan is None:
        rspan = len(text) + 1
    frame = tk.Frame(plane, name="contentFrame")
    frame.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=(30, 10), pady=(0, 20), sticky='nw')
    for item in text:
        if text.index(item) == page:
            text_config(frame, r=text.index(item), msg=item, fontsize=14, fcolour='red', px=0, py=0,
                        stick='nw', position='left')
        else:
            text_config(frame, r=text.index(item), msg=item, fontsize=14, fcolour='black', px=0, py=0,
                        stick='nw', position='left')


def text_config(plane=None, r=0, msg='', fontsize=18, bcolour=None, fcolour='black', px=10, py=20,
                stick='nsew', position: 'Literal["left", "right", "center"]' = "center",
                w=0, cspan=1):
    text = tk.Label(plane, bg=bcolour, width=w, font=('Arial', fontsize), fg=fcolour, text=msg,
                    anchor="center", justify=position)
    text.grid(row=r, column=0, columnspan=cspan, padx=px, pady=py, sticky=stick)


def tickboxes_frame_config(plane=None, r=1, c=0, rspan=1, cspan=2):
    frame = tk.Frame(plane, name="tickboxesFrame")
    frame.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=0, sticky='nsew')
    text_config(frame, fontsize=12, msg="Display Values:", position="left", r=0, stick="w")
    return frame


def whitespace(plane=None, r=0, fontsize=18, bcolour=None):
    for i in range(r):
        text = tk.Label(plane, bg=bcolour, font=('Arial', fontsize), text='', name="whitespace")
        text.grid(row=i, column=0, padx=10, pady=10, sticky='nsew')


def mes_box(plane=None, msg="", r=1, c=2, rspan=1, cspan=1):
    message = tk.Text(plane, bg='white', height=4, font=('Arial', 12), name="message")
    message.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=(5, 10), sticky='nsew')
    message.insert(tk.END, msg)
    return message


def add_tickboxes(plane=None, list1_var=None, list1_name=None, list2_var=None, list2_name=None, col=1):
    tick_vars1 = []
    for i, option in enumerate(list1_name):
        if i in list1_var:
            tick_var = tk.BooleanVar(value=True)
        else:
            tick_var = tk.BooleanVar()
        tick_vars1.append(tick_var)
        tickbox = tk.Checkbutton(plane, text=option, variable=tick_var)
        tickbox.grid(row=i + 1, column=0, padx=10, sticky='nw')
    if col == 2:
        tick_vars2 = []
        for i, option in enumerate(list2_name):
            if i in list2_var:
                tick_var = tk.BooleanVar(value=True)
            else:
                tick_var = tk.BooleanVar()
            tick_vars2.append(tick_var)
            tickbox = tk.Checkbutton(plane, text=option, variable=tick_var)
            tickbox.grid(row=i + 1, column=1, padx=10, sticky='nw')
    else:
        tick_vars2 = None
    return tick_vars1, tick_vars2


def toolbar(plane=None, frame=None, state=0):
    tool = NavigationToolbar2Tk(plane, frame)
    tool.update()
    tool.pack(side=tk.BOTTOM, fill=tk.X)
    if state == 0:
        tool.pan()
    return tool


def data_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=2, s: Literal["normal", "active", "disabled"] = "normal"):
    btn = tk.Button(plane, text="Dataset Selection", font=('Arial', 16), width=18, state=s,
                    command=lambda: select_dataset(gui), name="data_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=(10, 0))


def sum_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=2, s: Literal["normal", "active", "disabled"] = "normal"):
    btn = tk.Button(plane, text="Data Summary", font=('Arial', 16), width=18, state=s,
                    command=lambda: gui.data_page2(), name="sum_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10)


def vis_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=2, s: Literal["normal", "active", "disabled"] = "normal"):
    btn = tk.Button(plane, text="Visualise Data", font=('Arial', 16), width=18, state=s,
                    command=lambda: gui.vis_page(), name="vis_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10)


def pre_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=2, s: Literal["normal", "active", "disabled"] = "normal"):
    btn = tk.Button(plane, text="Features Selection", font=('Arial', 16), width=18, state=s,
                    command=lambda: gui.pre_page1(), name="pre_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10)


def ml_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=2, s: Literal["normal", "active", "disabled"] = "normal"):
    btn = tk.Button(plane, text="Machine Learning", font=('Arial', 16), width=18, state=s,
                    command=lambda: gui.ml_page1(), name="ml_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10)


def plot_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=2, w=10):
    btn = tk.Button(plane, text="Plot", font=('Arial', 14), width=w, command=lambda: gui.edit_graph(),
                    name="plot_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10, sticky='n')


def feat_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=1, w=6):
    btn = tk.Button(plane, text="Add", font=('Arial', 16), width=w, command=lambda: features(gui),
                    name="feat_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10, sticky='s')


def reset_btn(plane=None, gui=None, r=1, c=1, rspan=1, cspan=1, w=6):
    btn = tk.Button(plane, text="Reset", font=('Arial', 16), width=w, command=lambda: reset_features(gui),
                    name="reset_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10, sticky='s')


def prev_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=1, w=10):
    btn = tk.Button(plane, text="Previous", font=('Arial', 18), width=w, command=lambda: gui.prev_btn_click(),
                    name="prev_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10, sticky='s')


def next_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=1, w=10):
    btn = tk.Button(plane, text="Next", font=('Arial', 18), width=w, command=lambda: gui.next_btn_click(),
                    name="next_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10, sticky='s')


def home_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=1, s: Literal["normal", "active", "disabled"] = "normal"):
    btn = tk.Button(plane, text="Home", font=('Arial', 18), width=8, state=s,
                    command=lambda: gui.home(), name="home_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=(10, 5), pady=10)


def help_btn(plane=None, gui=None, r=1, c=0, rspan=1, cspan=1, s: Literal["normal", "active", "disabled"] = "normal"):
    btn = tk.Button(plane, text="\u2139", font=('Wingdings', 18), width=2, state=s,
                    command=lambda: gui.help(), name="help_btn")
    btn.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, padx=10, pady=10, sticky='w')


def gui_config(plane=None, row=0, col=0):
    for i in range(row + 1):
        plane.rowconfigure(i, weight=1)
    for i in range(col):
        plane.columnconfigure(i, weight=1)
    plane.columnconfigure(col, weight=10)


def clear(plane=None):
    mes_content = ""
    objects = plane.winfo_children()
    for obj in objects:
        if str(obj) == ".message":
            mes_content = obj.get("1.0", "end-1c")
        obj.destroy()
    return mes_content


def ml_btns(gui=None, plane=None, s='normal'):
    btn1 = tk.Button(plane, text="Self Organising Map (SOM)", font=('Arial', 18), width=26, state=s,
                     command=lambda: gui.som_page1())
    btn1.grid(row=0, column=0, padx=10, pady=10)
    btn2 = tk.Button(plane, text="K Nearest Neighbor (KNN)", font=('Arial', 18), width=26, state=s,
                     command=lambda: gui.ml_page4('knn'))
    btn2.grid(row=0, column=1, padx=10, pady=10)
    btn3 = tk.Button(plane, text="Decision Tree", font=('Arial', 18), width=26, state=s,
                     command=lambda: gui.ml_page4('dt'))
    btn3.grid(row=1, column=0, padx=10, pady=10)
    btn4 = tk.Button(plane, text="Random Forest", font=('Arial', 18), width=26, state=s,
                     command=lambda: gui.ml_page4('rf'))
    btn4.grid(row=1, column=1, padx=10, pady=10)
    btn5 = tk.Button(plane, text="XG Boost", font=('Arial', 18), width=26, state=s,
                     command=lambda: gui.ml_page4('xg'))
    btn5.grid(row=2, column=0, padx=10, pady=10)
    btn5 = tk.Button(plane, text="Support Vector Machine (SVM)", font=('Arial', 18), width=26, state=s,
                     command=lambda: gui.ml_page4('svm'))
    btn5.grid(row=2, column=1, padx=10, pady=10)


def active_btn(plane=None, btn_list=None):
    objects = plane.winfo_children()
    for obj in objects:
        if str(obj) in btn_list:
            obj.config(state="normal")


def features(gui):
    start = float(gui.slider_start)
    end = float(gui.slider_end)
    if start == 0:
        start = float(gui.DF.start)
    if end == 0:
        end = float(gui.DF.end)
    if len(gui.features_list) == 0:
        gui.features_list.append([start, end])
    else:
        list1 = [start, end]
        for i, sublist in enumerate(gui.features_list):
            state, start, end = dp.checkOverlap(list1, sublist)
            if state == 1:
                gui.features_list.remove(sublist)
                gui.features_list.insert(i, [start, end])
            elif state == 2:
                if i == len(gui.features_list) - 1:
                    gui.features_list.append([start, end])
        if len(gui.features_list) > 1:
            gui.features_list = dp.sortList(gui.features_list)

    objects = gui.root.winfo_children()
    for obj in objects:
        if str(obj) == ".message":
            obj.insert("1.0", f"Selected ranges for machine learning: {gui.features_list}\n")


def reset_features(gui):
    gui.features_list = []

    objects = gui.root.winfo_children()
    for obj in objects:
        if str(obj) == ".message":
            obj.insert("1.0", f"Selected ranges for machine learning: {gui.features_list}\n")


def select_dataset(gui=None):
    gui.data_page1()
    home_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(home_dir, "Data")
    file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.xlsx")], initialdir=default_dir)
    if file_path:
        os.startfile(file_path)
        time.sleep(2)
        response = tk.messagebox.askyesno('Dataset Confirmation', 'Would you like to proceed with this dataset?')
        if response:
            dataset = file_path.split('/')[-1]
            if len(dataset) > 25:
                dataset = dataset[:21] + " ..."
            gui.DF.read_excel(file_path)
            gui.data_page2(dataset, file_path)
        else:
            gui.home()
    else:
        gui.home()


def scale_win(gui):
    width = gui.root.winfo_screenwidth() - 200
    height = gui.root.winfo_screenheight() - 200
    gui.root.geometry(f"{width}x{height}+{100}+{100}")


def get_model(model=None):
    if model == 'knn':
        return '|--> KNN'
    elif model == 'dt':
        return '|--> Decision Tree'
    elif model == 'rf':
        return '|--> Random Forest'
    elif model == 'xg':
        return '|--> XG Boost'
    elif model == 'svm':
        return '|--> SVM'


def get_para_content(plane=None, model=None, parameter=None, gui=None):
    text_width = 18
    entry_width = 6
    row = 0
    if model == 'som':
        input_rows, input_columns = gui.DF.get_info()
        text = tk.Label(plane, width=text_width, bg='light grey', font=('Arial', 12),
                        text=f"Data Input Sizes:", justify='left')
        text.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        text = tk.Label(plane, width=text_width, bg='light grey', font=('Arial', 12),
                        text=f"{input_rows} rows\n      x\n{input_columns} features", justify='left')
        text.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        row += 3
        text1 = tk.Label(plane, width=text_width, font=('Arial', 12), text="SOM x-sizes:", justify='left')
        text1.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry1 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry1.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry1.insert(0, str(parameter['som'][0]))
        row += 1
        text2 = tk.Label(plane, width=text_width, font=('Arial', 12), text="SOM y-sizes:", justify='left')
        text2.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry2 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry2.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry2.insert(0, str(parameter['som'][1]))
        row += 1
        text3 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Neighbourhood:", justify='left')
        text3.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry3 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry3.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry3.insert(0, str(parameter['som'][2]))
        row += 1
        text4 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Learning rate:", justify='left')
        text4.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry4 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry4.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry4.insert(0, str(parameter['som'][3]))
        row += 1
        text5 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Epoch:", justify='left')
        text5.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry5 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry5.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry5.insert(0, str(parameter['som'][4]))
        row += 2
        btn = tk.Button(plane, text="Train", font=('Arial', 18), width=10,
                        command=lambda: gui.som_train([int(entry1.get()),
                                                       int(entry2.get()),
                                                       float(entry3.get()),
                                                       float(entry4.get()),
                                                       int(entry5.get())], 0))
        btn.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='s')
    elif model == 'knn':
        text1 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Neighbour:", justify='left')
        text1.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry1 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry1.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry1.insert(0, str(parameter[model][0]))
        row += 1
        white = tk.Label(plane, width=text_width, height=2, bg='light grey', font=('Arial', 12), text="",
                         justify='left')
        white.grid(row=row, column=0, rowspan=3, columnspan=2, padx=10, pady=10, sticky='w')
        row += 3
        btn = tk.Button(plane, text="Train", font=('Arial', 14), width=10,
                        command=lambda: ml_train(gui.frame, gui, model, [int(entry1.get())]))
        btn.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='s')
    elif model == 'dt':
        text1 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Tree Depth:", justify='left')
        text1.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry1 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry1.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry1.insert(0, str(parameter[model][0]))
        row += 1
        white = tk.Label(plane, width=text_width, height=2, bg='light grey', font=('Arial', 12), text="",
                         justify='left')
        white.grid(row=row, column=0, rowspan=3, columnspan=2, padx=10, pady=10, sticky='w')
        row += 3
        btn = tk.Button(plane, text="Train", font=('Arial', 14), width=10,
                        command=lambda: ml_train(gui.frame, gui, model, [int(entry1.get())]))
        btn.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='s')
    elif model == 'rf':
        text1 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Number of Tree:", justify='left')
        text1.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry1 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry1.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry1.insert(0, str(parameter[model][0]))
        row += 1
        text2 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Max Tree Depth:", justify='left')
        text2.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry2 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry2.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry2.insert(0, str(parameter[model][1]))
        row += 1
        white = tk.Label(plane, width=text_width, height=2, bg='light grey', font=('Arial', 12), text="",
                         justify='left')
        white.grid(row=row, column=0, rowspan=2, columnspan=2, padx=10, pady=10, sticky='w')
        row += 2
        btn = tk.Button(plane, text="Train", font=('Arial', 14), width=10,
                        command=lambda: ml_train(gui.frame, gui, model, [int(entry1.get()), int(entry2.get())]))
        btn.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='s')
    elif model == 'xg':
        text1 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Number of Tree:", justify='left')
        text1.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry1 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry1.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry1.insert(0, str(parameter[model][0]))
        row += 1
        text2 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Max Depth:", justify='left')
        text2.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry2 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry2.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry2.insert(0, str(parameter[model][1]))
        row += 1
        text3 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Learning Rate:", justify='left')
        text3.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry3 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry3.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry3.insert(0, str(parameter[model][2]))
        row += 1
        white = tk.Label(plane, width=text_width, height=2, bg='light grey', font=('Arial', 12), text="",
                         justify='left')
        white.grid(row=row, column=0, rowspan=2, columnspan=2, padx=10, pady=10, sticky='w')
        row += 2
        btn = tk.Button(plane, text="Train", font=('Arial', 14), width=10,
                        command=lambda: ml_train(gui.frame, gui, model,
                                                 [int(entry1.get()), int(entry2.get()), float(entry3.get())]))
        btn.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='s')
    elif model == 'svm':
        text1 = tk.Label(plane, width=text_width, font=('Arial', 12), text="Regularization:", justify='left')
        text1.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        entry1 = tk.Entry(plane, width=entry_width, font=('Arial', 12))
        entry1.grid(row=row, column=1, padx=10, pady=10, sticky='w')
        entry1.insert(0, str(parameter[model][0]))
        row += 1
        white = tk.Label(plane, width=text_width, height=2, bg='light grey', font=('Arial', 12), text="",
                         justify='left')
        white.grid(row=row, column=0, rowspan=3, columnspan=2, padx=10, pady=10, sticky='w')
        row += 3
        btn = tk.Button(plane, text="Train", font=('Arial', 14), width=10,
                        command=lambda: ml_train(gui.frame, gui, model, [int(entry1.get())]))
        btn.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='s')


def som_draw(gui=None, state=0):
    clear(gui.frame)
    if state == 0:
        plot, header, report, matrix = gui.DF.SOM(gui.hyperparameter['som'], gui.som_state)
    else:
        plot = gui.DF.som_fig
    canvas = draw_figure(plot, gui.frame)
    canvas.draw()
    return canvas


def ml_train(plane=None, gui=None, model=None, parameter=None):
    gui.hyperparameter[model] = parameter
    active_btn(gui.root, ['.result_btn'])
    clear(plane)
    if model == 'knn':
        header, report, matrix = gui.DF.knn(parameter)
    elif model == 'dt':
        header, report, matrix = gui.DF.dt(parameter)
    elif model == 'rf':
        header, report, matrix = gui.DF.rf(parameter)
    elif model == 'xg':
        header, report, matrix = gui.DF.xg(parameter)
    elif model == 'svm':
        header, report, matrix = gui.DF.svm(parameter)
    heading1 = tk.Label(plane, bg='light grey', font=('Arial', 20), text="Result", justify='center',
                        anchor='center')
    heading1.grid(row=0, column=0, columnspan=2, padx=20, pady=10, sticky='ew')
    heading2 = tk.Label(plane, bg='light grey', font=('Arial', 16), text=header[0], justify='left')
    heading2.grid(row=1, column=0, padx=20, pady=10, sticky='w')
    heading3 = tk.Label(plane, bg='light grey', font=('Arial', 16), text=header[1], justify='left')
    heading3.grid(row=2, column=0, padx=20, pady=10, sticky='w')
    text1 = tk.Label(plane, bg='light grey', font=('Arial', 16), text="Accuracy Scores:", justify='left')
    text1.grid(row=3, column=0, padx=20, pady=10, sticky='w')
    text2 = tk.Label(plane, bg='light grey', font=('Arial', 16), text="Confusion Matrix:", justify='left')
    text2.grid(row=3, column=1, padx=20, pady=10, sticky='w')
    score = treeview(plane, report)
    score.grid(row=4, column=0, padx=20, pady=0, sticky='ew')
    cm = treeview(plane, matrix, w2=60, tree_no=1)
    cm.grid(row=4, column=1, padx=20, pady=0, sticky='ew')


def treeview(plane=None, matrix=None, t1=14, t2=16, r=30, w1=90, w2=100, tree_no=0):
    # Create the style
    style = ttk.Style()
    # Set the font size for the table
    style.configure(f"Tree{tree_no}.Treeview", font=("Arial", t1))
    style.configure(f"Tree{tree_no}.Treeview.Heading", font=("Arial", t2))
    style.configure(f"Tree{tree_no}.Treeview", highlightthickness=5, highlightcolor="gray",
                    highlightbackground="gray", rowheight=r)
    style.map(f"Tree{tree_no}.Treeview", background=[("selected", "blue")],
              foreground=[("selected", "white")])
    table = ttk.Treeview(plane, style=f"Tree{tree_no}.Treeview")
    table['columns'] = tuple(matrix.columns)
    table.column("#0", width=w1, minwidth=w1 - 10, stretch=False)
    table.heading("#0", text=matrix.index.name)
    for i, x in enumerate(matrix.columns):
        table.column(f'{x}', width=w2, minwidth=w2 - 10, anchor='center', stretch=False)
        table.heading(f'{x}', text=x, anchor='center')

    for i, x in enumerate(matrix.index):
        table.insert("", "end", text=f'{x}', values=tuple(matrix.iloc[i:i + 1, :].values.flatten()),
                     tags=('center',))

    return table


def draw_figure(fig=None, plane=None, state=0):
    plt.close('all')
    canvas = FigureCanvasTkAgg(fig, master=plane)
    canvas.draw()
    toolbar(canvas, plane, state)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return canvas


def save_ml(gui=None, msg_box=None):
    df = gui.DF.get_ml()
    if len(gui.features_list) == 0:
        gui.features_list = gui.DF.get_features()
    data = [gui.hyperparameter.items(), gui.features_list, df]
    home_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(home_dir, "Trained ML")
    now = datetime.now()
    file_path = os.path.join(default_dir, f"{now.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(file_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        writer.writerow(['Hyperparameters'])
        writer.writerow(data[0])
        # Save features list
        writer.writerow(['Features'])
        writer.writerow(data[1])
        # Save X_train and y_train
        writer.writerow(['Train'])
        df.to_csv(csvfile, index=False)
    msg_box.insert("1.0", "Machine learning metadata has been saved.\n")


def check_folder():
    home_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(home_dir, "Data")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join(home_dir, "Trained ML")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join(home_dir, "ML Result")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def save_result(gui=None):
    file_name = gui.DF.path.split('/')[-1]
    file_name = file_name.split('.xl')[0]
    home_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(home_dir, "ML Result")
    gui.DF.result_df.to_csv(os.path.join(folder_path, f"{file_name}.csv"), index=False)
