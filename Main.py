"""
This is the script of the graphical user interface (GUI). It basically provides an interface for the user
to interact with. After receiving the input from the user, it will call the functions in the Carpark script and
display the result from the script.
"""

from matplotlib.widgets import RangeSlider
from tkinter.ttk import Combobox
from tkinter import filedialog
import tkinter as tk  # Import the tkinter module for building the GUI
import os
import csv
import ast
import pandas as pd
import GUI_Element as GE
import matplotlib
import DataProcessing as dp
matplotlib.use('TkAgg')


# Construct the GUI class
class GUI:

    def __init__(self):
        GE.check_folder()
        # Initiate the class with all the settings:
        self.root = tk.Tk()  # Set the root window
        self.root.title("Predictive Model of Inflamed Cochlear Cell")
        # The root window of the GUI
        GE.scale_win(self)
        GE.gui_config(self.root, 20, 2)
        self.frame = None
        self.DF = dp.Processing()
        self.plot_tickbox1, self.plot_tickbox2, self.pca_tickbox = [], [1], [0, 1, 2, 3, 4]
        self.tick_vars1, self.tick_vars2, self.tick_vars3 = [], [], []
        self.features_list = []
        self.page, self.split, self.ml_state, self.som_state = 0, 0.2, 0, 0
        self.pca, self.slider_start, self.slider_end = 0, 0, 0
        self.dataset = "None"
        self.hyperparameter = {"som": [10, 10, 5.0, 0.5, 5000],
                               "knn": [5],
                               "dt": [5],
                               "rf": [100, 5],
                               "xg": [100, 5, 0.1],
                               "svm": [1],
                               }
        self.home(state=1)

    def home(self, state=0):
        self.page = 0
        if state == 0:
            message_content = GE.clear(self.root)
        else:
            message_content = ''
        self.frame = GE.frame_config(self.root, rspan=16)
        row = 0
        GE.data_btn(self.root, self, row)
        row += 1
        text = ['Selected Dataset:', self.dataset]
        GE.content_frame(self.root, text, row, page=2)
        row = row + len(text) + 1
        if self.dataset == "None":
            btn_state = 'disabled'
        else:
            btn_state = 'normal'
        GE.sum_btn(self.root, self, row, s=btn_state)
        row += 1
        GE.vis_btn(self.root, self, row, s=btn_state)
        row += 1
        GE.pre_btn(self.root, self, row, s=btn_state)
        row += 1
        GE.ml_btn(self.root, self, row, s=btn_state)
        row = self.root.grid_size()[1] - 5
        GE.mes_box(self.root, msg=message_content, r=row, rspan=5)
        row += 4
        GE.help_btn(self.root, self, row)
        text = "Please click a button on the left to proceed!"
        GE.text_config(self.frame, r=0, msg=text, bcolour='light grey', w=len(text), px=(140, 0), py=(50, 0))

    def data_page1(self):
        self.page = 1
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        GE.text_config(self.frame, r=0, msg='Loading ...', bcolour='light grey', w=40, px=(140, 0), py=(50, 0),
                       stick='nse', fontsize=22)
        row = 0
        GE.sum_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['Selected Dataset:', self.dataset]
        GE.content_frame(self.root, text, row, page=2)
        row = self.root.grid_size()[1] - 4
        GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        row += 3
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1, s='disabled')

    def data_page2(self, dataset=None, file_path=None):
        self.page = 2
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        summary, class_df = self.DF.get_summary()
        GE.text_config(self.frame, r=0, msg=summary.split('\n')[0], bcolour='light grey', px=10, fontsize=22)
        GE.text_config(self.frame, r=1, msg=summary[25:], bcolour='light grey', px=10, py=10, position='left',
                       fontsize=16)
        row = 0
        if self.dataset == 'None':
            self.dataset = dataset
        GE.sum_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['Selected Dataset:', self.dataset]
        GE.content_frame(self.root, text, row, page=2)
        row = self.root.grid_size()[1] - 4
        mes_box = GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        if file_path is not None:
            mes_box.insert("1.0",
                           "----------------------------------------------------------------------------------------\n\n")
            mes_box.insert("1.0", f"""Selected file: {file_path.split('/')[-1]}\n""")
        row += 2
        GE.next_btn(self.root, self, row, cspan=2)
        row += 1
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)

    def data_page3(self):
        self.page = 3
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        summary, class_df = self.DF.get_summary()
        GE.text_config(self.frame, r=0, msg="Number of sample groups:", bcolour='light grey', px=10, fontsize=22)
        table = GE.treeview(self.frame, class_df, w1=200, w2=200)
        table.grid(row=1, column=0, padx=10, pady=0, sticky='ew')
        row = 0
        GE.sum_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['Selected Dataset:', self.dataset]
        GE.content_frame(self.root, text, row, page=2)
        row = self.root.grid_size()[1] - 4
        GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        row += 2
        GE.prev_btn(self.root, self, row, cspan=2)
        row += 1
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)

    def vis_page(self):
        self.page = 11
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        row = 0
        GE.vis_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['|--> Graph Visualisation']
        GE.content_frame(self.root, text, r=row, cspan=2, page=0)
        row += len(text)
        list2_name = ['Min', 'Mean', 'Max']
        class_list, label_list = self.DF.get_list()
        if len(self.plot_tickbox1) == 0:
            self.plot_tickbox1 = [x for x in range(len(class_list))]
        tickboxes_frame = GE.tickboxes_frame_config(self.root, row, rspan=len(class_list) + 2)
        self.tick_vars1, self.tick_vars2 = GE.add_tickboxes(tickboxes_frame, self.plot_tickbox1, class_list,
                                                            self.plot_tickbox2, list2_name, col=2)
        row += len(class_list) + 2
        GE.plot_btn(self.root, self, row)
        row = self.root.grid_size()[1] - 4
        mes_box = GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        mes_box.insert("1.0",
                       "----------------------------------------------------------------------------------------\n\n")
        mes_box.insert("1.0", "Pans the graph by holding the left mouse button;\n"
                              "Zooms in/out by holding the right mouse button.\n")
        row += 3
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)
        plot = self.DF.plot(self.plot_tickbox1, self.plot_tickbox2)
        GE.draw_figure(plot, self.frame)

    def pre_page1(self):
        self.page = 21
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        row = 0
        GE.pre_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['|--> PCA: Scree Plot', '|--> PCA: Loading Plot\n     & Data Selection']
        GE.content_frame(self.root, text, r=row, rspan=4, cspan=2, page=0)
        row = self.root.grid_size()[1] - 4
        GE.next_btn(self.root, self, row, cspan=2)
        mes_box = GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        mes_box.insert("1.0",
                       "----------------------------------------------------------------------------------------\n\n")
        mes_box.insert("1.0", "Pans the graph by holding the left mouse button;\n"
                              "Zooms in/out by holding the right mouse button.\n")
        row += 2
        GE.next_btn(self.root, self, row, cspan=2)
        row += 1
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)
        plot, text = self.DF.pca1()
        GE.draw_figure(plot, self.frame)
        mes_box.insert("1.0", text)

    def pre_page2(self):
        self.page = 22
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        row = 0
        GE.pre_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['|--> PCA: Scree Plot', '|--> PCA: Loading Plot\n     & Data Selection']
        GE.content_frame(self.root, text, r=row, rspan=3, cspan=2, page=1)
        row = row + len(text) + 1
        if len(self.pca_tickbox) == 0:
            self.pca_tickbox = [0, 1, 2, 3, 4]
        list1_name = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        tickboxes_frame = GE.tickboxes_frame_config(self.root, row, rspan=len(list1_name))
        self.tick_vars3, tick_vars = GE.add_tickboxes(tickboxes_frame, self.pca_tickbox, list1_name)
        row = row + len(list1_name) + 1
        GE.plot_btn(self.root, self, r=row)
        row += 1
        text = tk.Label(self.root, font=('Arial', 12), text="Features Selection\nMethod:", justify='left')
        text.grid(row=row, column=0, padx=10, pady=10, sticky='w')
        combo = Combobox(self.root, width=12, state='readonly', font=('Arial', 12))
        combo['values'] = ['Range Input', 'Range Slider']
        combo.current(self.pca)
        combo.grid(row=row, column=1, padx=10, pady=10, sticky='e')

        def on_change(event):
            selected = combo.get()
            if selected == 'Range Input':
                self.pca = 0
            elif selected == 'Range Slider':
                self.pca = 1
            self.pre_page2()

        # Bind the on_change function to the Combobox's selected event
        combo.bind('<<ComboboxSelected>>', on_change)

        def update_plot(val):
            if self.pca == 0:
                state1, state2 = 0, 0
                start = float(entry_value1.get())
                end = float(entry_value2.get())
                if self.DF.start - 1 <= start < end:
                    self.slider_start = start
                    state1 = 1
                if self.DF.end + 1 >= end > start:
                    self.slider_end = end
                    state2 = 1
                if state1 == 0 or state2 == 0:
                    tk.messagebox.showerror(title="Error", message="The input range is not correct.")
                    return
            elif self.pca == 1:
                self.slider_start, self.slider_end = slider.val
            ax.set_xlim(self.slider_start, self.slider_end)
            plot.canvas.draw()

        if self.pca == 0:
            row += 1
            text = tk.Label(self.root, font=('Arial', 12), text="Range Start:", justify='left')
            text.grid(row=row, column=0, padx=10, pady=10, sticky='w')
            entry_value1 = tk.StringVar()
            entry1 = tk.Entry(self.root, textvariable=entry_value1, width=8, font=('Arial', 12), name="entry1")
            entry1.grid(row=row, column=1, padx=10, pady=10, sticky='w')
            entry1.insert(0, str(int(self.DF.start)))
            entry1.bind('<Return>', update_plot)
            row += 1
            text = tk.Label(self.root, font=('Arial', 12), text="Range End:", justify='left')
            text.grid(row=row, column=0, padx=10, pady=10, sticky='w')
            entry_value2 = tk.StringVar()
            entry2 = tk.Entry(self.root, textvariable=entry_value2, width=8, font=('Arial', 12), name="entry2")
            entry2.grid(row=row, column=1, padx=10, pady=10, sticky='w')
            entry2.insert(0, str(int(self.DF.end)))
            entry2.bind('<Return>', update_plot)
        row = self.root.grid_size()[1] - 5
        GE.feat_btn(self.root, self, r=row)
        GE.reset_btn(self.root, self, r=row)
        row = self.root.grid_size()[1] - 4
        mes_box = GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        mes_box.insert("1.0",
                       "----------------------------------------------------------------------------------------\n\n")
        if self.pca == 0:
            mes_box.insert("1.0",
                           f"Press \"Enter\" key to confirm the range after changing the values in input boxes.\n")
        mes_box.insert("1.0", f"Press \"Add\" to select data ranges for machine learning\n")
        mes_box.insert("1.0", f"Selected ranges for machine learning: {self.features_list}\n")
        row += 2
        GE.prev_btn(self.root, self, row, cspan=2)
        row += 1
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)
        plot, ax = self.DF.pca2(self.pca_tickbox)
        GE.draw_figure(plot, self.frame, 1)
        if self.pca == 1:
            slider = self.add_slider(plot)
            slider.on_changed(update_plot)

    def ml_page1(self):
        self.page = 31
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        row = 0
        GE.ml_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['|--> ML Selection', '|--> Model Training']
        GE.content_frame(self.root, text, r=row, cspan=2, page=0)
        row = row + len(text) + 1
        btn1 = tk.Button(self.root, text="Train new ML models", font=('Arial', 16), width=18,
                         command=lambda: self.ml_page3(ml_state=0))
        btn1.grid(row=row, column=0, columnspan=2, padx=10, pady=10)
        row += 1
        btn2 = tk.Button(self.root, text="Use trained ML models", font=('Arial', 16), width=18,
                         command=lambda: self.ml_page2())
        btn2.grid(row=row, column=0, columnspan=2, padx=10, pady=10)
        row = self.root.grid_size()[1] - 4
        GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        row += 3
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)
        GE.ml_btns(self, self.frame, 'disabled')

    def ml_page2(self):
        self.features_list = []
        home_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.join(home_dir, "Trained ML")
        file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv")],
                                               initialdir=default_dir)
        if file_path:
            with open(file_path, 'r') as csvfile:
                # Create a CSV reader object
                reader = csv.reader(csvfile)

                # Iterate over the rows in the CSV file
                for i, row in enumerate(reader):
                    # Process the data in each row
                    if i == 0:
                        if row[0] != 'Hyperparameters':
                            tk.messagebox.showerror(title="Error", message="The file format is not correct.")
                            self.ml_page1()
                            return
                    elif i == 1:
                        for x in row:
                            if x:
                                model = ast.literal_eval(x)[0]
                                parameter = ast.literal_eval(x)[1]
                                self.hyperparameter[model] = parameter
                    elif i == 2:
                        if row[0] != 'Features':
                            tk.messagebox.showerror(title="Error", message="The file format is not correct.")
                            self.ml_page1()
                            return
                    elif i == 3:
                        for x in row:
                            if x:
                                self.features_list.append(ast.literal_eval(x))
                    elif i == 4:
                        if row[0] != 'Train':
                            tk.messagebox.showerror(title="Error", message="The file format is not correct.")
                            self.ml_page1()
                            return
                    elif i == 5:
                        df = pd.DataFrame(columns=row)
                    elif i >= 6:
                        if row:
                            df.loc[len(df)] = row
                errorno = self.DF.load_ml(self, self.features_list, df)
                if errorno == 1:
                    tk.messagebox.showerror(title="Error",
                                            message="The trained model does not match the size of the read dataset.")
                    self.ml_page1()
                self.ml_state = 2
                self.ml_page3(self.ml_state)

        else:
            self.home()

    def ml_page3(self, ml_state=None):
        if ml_state is None:
            ml_state = self.ml_state
        elif ml_state == 0:
            self.ml_state = 0
        self.page = 33
        self.som_state = 0
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        self.DF.data_select(self.features_list)
        row = 0
        GE.ml_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['|--> ML Selection', '|--> Model Training']
        GE.content_frame(self.root, text, r=row, cspan=2, page=0)
        row = row + len(text) + 1
        if ml_state <= 1:
            para_frame = tk.Frame(self.root, bg='light grey', width=8)
            para_frame.grid(row=row, column=0, rowspan=4, columnspan=2, padx=20, pady=10, sticky='nsew')
            row1 = 0
            text_width = 12
            entry_width = 6
            text1 = tk.Label(para_frame, width=text_width, font=('Arial', 12), text="Testing Size:", justify='left')
            text1.grid(row=row1, column=0, padx=10, pady=10, sticky='w')
            entry1 = tk.Entry(para_frame, width=entry_width, font=('Arial', 12))
            entry1.grid(row=row1, column=1, padx=10, pady=10, sticky='w')
            entry1.insert(0, str(self.split))
            row1 += 1
            white = tk.Label(para_frame, width=text_width, height=2, bg='light grey', font=('Arial', 12), text="",
                             justify='left')
            white.grid(row=row1, column=0, rowspan=2, columnspan=2, padx=10, pady=10, sticky='w')
            row1 += 2
            btn1 = tk.Button(para_frame, text="Split Data", font=('Arial', 14), width=10,
                             command=lambda: self.DF.split_data(self, float(entry1.get())))
            btn1.grid(row=row1, column=0, columnspan=2, padx=10, pady=10)
            if self.ml_state == 0:
                self.DF.split_data(self, float(entry1.get()))
            row = self.root.grid_size()[1] - 3
            btn2 = tk.Button(self.root, text="Save ML metadata", font=('Arial', 18), width=16,
                             command=lambda: GE.save_ml(self, mes_box))
            btn2.grid(row=row, column=0, columnspan=2, padx=10, pady=10)
        row = self.root.grid_size()[1] - 4
        mes_box = GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        mes_box.insert("1.0",
                       "----------------------------------------------------------------------------------------\n\n")
        if self.ml_state == 0:
            mes_box.insert("1.0", f"Press \'Split Data\' to split the dataset into training data "
                                  f"and testing data with desired portion.\n")
        mes_box.insert("1.0", f"Please select the machine learning method\n")
        row += 2
        GE.prev_btn(self.root, self, row, cspan=2)
        row += 1
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)
        GE.ml_btns(self, self.frame)

    def som_page1(self):
        self.page = 41
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        row = 0
        GE.ml_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['|--> ML Selection', '|--> SOM']
        GE.content_frame(self.root, text, r=row, cspan=2, page=1)
        row = row + len(text) + 1
        para_frame = tk.Frame(self.root, bg='light grey', width=10)
        para_frame.grid(row=row, column=0, rowspan=10, columnspan=2, padx=10, pady=10, sticky='nsew')
        GE.get_para_content(para_frame, 'som', self.hyperparameter, self)
        row = self.root.grid_size()[1] - 4
        mes_box = GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        mes_box.insert("1.0",
                       "----------------------------------------------------------------------------------------\n\n")
        mes_box.insert("1.0", f"Result of the SOM page 1\n")

        row += 2
        GE.prev_btn(self.root, self, row, w=8)
        GE.next_btn(self.root, self, row, c=1, w=8)
        row += 1
        if self.som_state > 0:
            GE.som_draw(self, 1)
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)

    def som_page2(self):
        self.page = 42
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        row = 0
        GE.ml_btn(self.root, self, row, s='disabled')
        row += 1
        text = ['|--> ML Selection', '|--> SOM']
        GE.content_frame(self.root, text, r=row, cspan=2, page=1)
        row = row + len(text) + 1
        para_frame = tk.Frame(self.root, bg='light grey', width=10, height=6)
        para_frame.grid(row=row, column=0, rowspan=10, columnspan=2, padx=10, pady=10, sticky='nsew')
        GE.get_para_content(para_frame, 'som', self.hyperparameter, self)
        row = self.root.grid_size()[1] - 4
        mes_box = GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        mes_box.insert("1.0",
                       "----------------------------------------------------------------------------------------\n\n")
        mes_box.insert("1.0", f"Result of the SOM page 2\n")
        btn1 = tk.Button(self.root, text="Save Result", font=('Arial', 18), width=10, name='result_btn',
                         command=lambda: GE.save_result())
        btn1.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='s')
        row += 2
        GE.prev_btn(self.root, self, row, cspan=2)
        row += 1
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)
        if self.som_state > 0:
            heading1 = tk.Label(self.frame, bg='light grey', font=('Arial', 20), text="Result", justify='center',
                                anchor='center')
            heading1.grid(row=0, column=0, columnspan=2, padx=20, pady=10, sticky='ew')
            heading2 = tk.Label(self.frame, bg='light grey', font=('Arial', 16), text=self.DF.som_header[0],
                                justify='left')
            heading2.grid(row=1, column=0, padx=20, pady=10, sticky='w')
            heading3 = tk.Label(self.frame, bg='light grey', font=('Arial', 16), text=self.DF.som_header[1],
                                justify='left')
            heading3.grid(row=2, column=0, padx=20, pady=10, sticky='w')
            text1 = tk.Label(self.frame, bg='light grey', font=('Arial', 16), text="Accuracy Scores:", justify='left')
            text1.grid(row=3, column=0, padx=10, pady=10, sticky='w')
            text2 = tk.Label(self.frame, bg='light grey', font=('Arial', 16), text="Confusion Matrix:", justify='left')
            text2.grid(row=3, column=1, padx=10, pady=10, sticky='w')
            score = GE.treeview(self.frame, self.DF.som_report)
            score.grid(row=4, column=0, padx=10, pady=0, sticky='ew')
            cm = GE.treeview(self.frame, self.DF.som_matrix, w2=60, tree_no=1)
            cm.grid(row=4, column=1, padx=10, pady=0, sticky='ew')

    def ml_page4(self, model=None):
        self.page = 34
        message_content = GE.clear(self.root)
        self.frame = GE.frame_config(self.root, rspan=17)
        white = tk.Label(self.frame, width=100, height=2, bg='light grey', font=('Arial', 12), text="",
                         justify='left')
        white.grid(row=0, column=0, rowspan=3, columnspan=2, padx=10, pady=10, sticky='w')
        row = 0
        GE.ml_btn(self.root, self, row, s='disabled')
        row += 1
        text1 = GE.get_model(model)
        text = ['|--> ML Selection', text1]
        GE.content_frame(self.root, text, r=row, page=1)
        row = row + len(text) + 1
        para_frame = tk.Frame(self.root, bg='light grey')
        para_frame.grid(row=row, column=0, rowspan=10, columnspan=2, padx=10, pady=10, sticky='nsew')
        GE.get_para_content(para_frame, model, self.hyperparameter, self)
        row = self.root.grid_size()[1] - 4
        mes_box = GE.mes_box(self.root, msg=message_content, r=row, rspan=4)
        mes_box.insert("1.0",
                       "----------------------------------------------------------------------------------------\n\n")
        mes_box.insert("1.0", f"Result of the {text1.split('>')[1][1:]}\n")
        btn1 = tk.Button(self.root, text="Save Result", font=('Arial', 18), width=10, state='disabled',
                         name='result_btn', command=lambda: GE.save_result(self))
        btn1.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='s')
        row += 2
        GE.prev_btn(self.root, self, row, cspan=2)
        row += 1
        GE.help_btn(self.root, self, row)
        GE.home_btn(self.root, self, row, 1)

    def edit_graph(self):
        list1 = [value.get() for value in self.tick_vars1]
        list2 = [value.get() for value in self.tick_vars2]
        list3 = [value.get() for value in self.tick_vars3]
        self.plot_tickbox1 = []
        self.plot_tickbox2 = []
        self.pca_tickbox = []
        for i, value in enumerate(list1):
            if value:
                self.plot_tickbox1.append(i)
        for i, value in enumerate(list2):
            if value:
                self.plot_tickbox2.append(i)
        for i, value in enumerate(list3):
            if value:
                self.pca_tickbox.append(i)
        GE.clear(self.frame)
        if self.page == 11:
            plot = self.DF.plot(self.plot_tickbox1, self.plot_tickbox2)
            GE.draw_figure(plot, self.frame)
        elif self.page == 22:
            plot, ax = self.DF.pca2(self.pca_tickbox)
            object = self.root.winfo_children()
            for obj in object:
                if str(obj) == '.entry1':
                    entry1 = obj
                elif str(obj) == '.entry2':
                    entry2 = obj
            # Update the plot based on the range slider
            def update_plot(val):
                if self.pca == 0:
                    state1, state2 = 0, 0
                    start = float(entry1.get())
                    end = float(entry2.get())
                    if self.DF.start - 1 <= start < end:
                        self.slider_start = start
                        state1 = 1
                    if self.DF.end + 1 >= end > start:
                        self.slider_end = end
                        state2 = 1
                    if state1 == 0 or state2 == 0:
                        tk.messagebox.showerror(title="Error", message="The input range is not correct.")
                        return
                elif self.pca == 1:
                    self.slider_start, self.slider_end = slider.val
                ax.set_xlim(self.slider_start, self.slider_end)
                plot.canvas.draw()

            if self.pca == 0:
                entry1.bind('<Return>', update_plot)
                entry2.bind('<Return>', update_plot)
            elif self.pca == 1:
                slider = self.add_slider(plot)
                slider.on_changed(update_plot)

            GE.draw_figure(plot, self.frame, 1)

    def next_btn_click(self):
        if self.page == 2:
            self.data_page3()
        elif self.page == 21:
            self.pre_page2()
        elif self.page == 22:
            self.ml_page1()
        elif self.page == 41:
            self.som_page2()

    def prev_btn_click(self):
        if self.page == 3:
            self.data_page2()
        elif self.page == 22:
            self.pre_page1()
        elif self.page == 33:
            self.ml_page1()
        elif self.page == 42:
            self.som_page1()
        else:
            self.ml_page3()

    def add_slider(self, fig=None):
        fig.subplots_adjust(bottom=0.2)
        ax_slider = fig.add_axes((0.1, 0.05, 0.75, 0.03))
        self.slider_start, self.slider_end = self.DF.get_range()
        slider = RangeSlider(ax_slider, "Range:", self.slider_start, self.slider_end,
                             valinit=(self.slider_start, self.slider_end), valstep=0.1)
        return slider

    def som_train(self, sublist=None, state=1):
        if self.page == 42:
            self.som_page1()
        if state == 0:
            self.hyperparameter['som'] = sublist
            self.som_state = sublist[4] // 5
        GE.som_draw(self, 0)
        if self.som_state < self.hyperparameter['som'][4]:
            self.root.after(100, self.som_train)
            self.som_state += (self.hyperparameter['som'][4] // 5)


def on_closing():
    """Function to be called when the close button is clicked"""
    # Add any cleanup or saving operations here
    window.root.quit()  # Quit the main event loop
    window.root.destroy()  # Destroy the Tkinter window


window = GUI()  # Initiate the class GUI
# Set the close button behavior
window.root.protocol("WM_DELETE_WINDOW", on_closing)

# The main loop of the program and react when an error arise
try:
    window.root.mainloop()
except BaseException as e:
    tk.messagebox.showerror("Error Type:", type(e).__name__)
    # print("\nError Type:", type(e).__name__)
