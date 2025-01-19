import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication,QAction, QLineEdit, QDialog, QMainWindow, QPushButton, QFileDialog, QSpinBox , QMenu, QGroupBox, QComboBox, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import scipy.stats as stats
import matplotlib.gridspec as gridspec
import os
from tkinter import filedialog
from tqdm import tqdm
import warnings
import math
import logging
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import copy
from joblib import Parallel, delayed
from filelock import FileLock
from scipy.stats import kstest
from io import BytesIO
from PIL import Image

BIAS_CATEGORIES = list(range(-50, 51,1))
process_column_name = "Results"
date_column_name = "Sampling Time"

global lambda_value


file_path = 'Analytes Indices.xlsx'  
# The variable 'file_path' refers to the uploaded Excel file containing key metrics
# for various analytes. These metrics include:
# - RCVg values (Optimum, Desirable, Minimum): Used to assess the reference change value.
# - TEa values (Optimum, Desirable, Minimum): Essential for defining allowable error ranges.
# - Measurable Lower and Upper Limits: Used to determine the valid measurement range.
# The data is crucial for statistical and analytical evaluations in this application.

df = pd.read_excel(file_path)

Analytes_List = df.set_index('Analyte').T.to_dict()

# The following lists are created to distinguish specific clinic types based on patient data
# extracted from each clinic's LIS (Laboratory Information System). Since the clinical data
# includes names in Turkish, the lists are written in Turkish to accurately identify clinics:
# - icu_name_list: Contains terms used to identify Intensive Care Units (ICU), including variations 
#   such as 'YOĞUN', 'YOĞ.', 'YOG.', and 'yoğun' in Turkish.
# - inpatient_name_list: Includes terms for inpatient services such as 'Servis', 'SERVİS', 'Klinik',
#   'KLİNİĞİ', 'KLİN', 'Klin', 'ONKO.', 'ONKO', 'PALYATİ', and 'TRAVMA'  in Turkish.
# - emergency_name_list: Designed to identify emergency units and related areas, including terms like 
#   'ACİL', 'Acil', 'ALAN', 'Sarı', 'Yeşil', 'Kırmızı', and their variations  in Turkish.
#
# Additional variables are defined for column names in the patient data:
# - gender_column_name: Represents the column containing gender information ('Cinsiyet'  in Turkish).
# - age_column_name: Represents the column for age data ('Yaş'  in Turkish).
# - sampling_time_column_name: Represents the column for the sample collection timestamp ('Kayıt Tar.'  in Turkish).
# - clinics_column_name: Represents the column indicating the clinic name ('Servis Adı'  in Turkish).
#
# These lists and column names are in Turkish to ensure accurate categorization of patient data
# specific to the clinic types (e.g., inpatient, emergency, ICU).

icu_name_list = ['YOĞUN', 'YOĞ.', 'YOG.', "yoğun"]
inpatient_name_list = ['Servis', 'SERVİS', "Klinik", "KLİNİĞİ", "KLİN", "Klin", 'ONKO.', 'ONKO', "PALYATİ", "TRAVMA"]
emergency_name_list = ['ACİL', 'Acil', 'ALAN', 'Alan', 'alan', 'Sarı', 'sarı', 'Yeşil', 'yeşil','Kırmızı','kırmızı']
gender_column_name = "Cinsiyet"
age_column_name = "Yaş" 
sampling_time_column_name = "Kayıt Tar." 
clinics_column_name = "Servis Adı"

def process_data_file(patient_category, allocation_ratio, measurement_lower_limit, measurement_upper_limit,  
                     gender_column_name, age_column_name, sampling_time_column_name, clinics_column_name,
                     icu_name_list, inpatient_name_list, emergency_name_list, process_column_name, selected_analyte):

    filepath = filedialog.askopenfilename(title="Select data for batch monitoring")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("File format not supported. Please select a CSV or Excel file.")
    

    if 'Gender' and 'Age' and 'Sampling Time' and 'Results'  in df.columns:
        print(f"\n{filepath} dosyası zaten mevcut, işlenmiş veriler yükleniyor.\n")
        df['Sampling Time'] = pd.to_datetime(df['Sampling Time'], errors='coerce', infer_datetime_format=True)
        df = df.sort_values(by='Sampling Time')
        final_df = df.copy()
    else:
        print("DataFrame sütun isimleri:", df.columns)

        if sampling_time_column_name not in df.columns:
            raise KeyError(f"'{sampling_time_column_name}' sütunu bulunamadı. Mevcut sütunlar: {df.columns}")

        result_column = df.columns[-1]
        df[process_column_name] = pd.to_numeric(df[result_column], errors='coerce')
        df = df.dropna(subset=[process_column_name])

        if (isinstance(measurement_lower_limit, (int, float)) and not math.isnan(measurement_lower_limit) and
            isinstance(measurement_upper_limit, (int, float)) and not math.isnan(measurement_upper_limit)):
            df = df[(df[process_column_name] <= measurement_upper_limit) & (df[process_column_name] >= measurement_lower_limit)]

        df[gender_column_name] = df[gender_column_name].replace({'Erkek': 'Male', 'Kadın': 'Female'})
        df[age_column_name] = df[age_column_name].replace({'Gün': 'Day', 'Ay': 'Month'}, regex=True)
        df[age_column_name] = pd.to_numeric(df[age_column_name], errors='coerce')
     
 
        if clinics_column_name in df.columns:
            df['All Patients'] = df.apply(
                lambda row: '+' , axis=1)
            df['ICU'] = df.apply(lambda row: '+' if isinstance(row[clinics_column_name], str) 
                     and any(keyword in row[clinics_column_name] for keyword in icu_name_list) 
                     else '', axis=1)

            df['Inpatient clinics'] = df.apply(lambda row: '+' if isinstance(row[clinics_column_name], str) 
                                   and any(keyword in row[clinics_column_name] for keyword in inpatient_name_list) 
                                   else '', axis=1)

            df['Emergency'] = df.apply(lambda row: '+' if isinstance(row[clinics_column_name], str) 
                           and any(keyword in row[clinics_column_name] for keyword in emergency_name_list) 
                           else '', axis=1)

            df['Outpatient clinics'] = df.apply(
                lambda row: '+' if (row['ICU'] == '' and row['Inpatient clinics'] == '' and row['Emergency'] == '' 
                                    and isinstance(row[clinics_column_name], str) 
                                    and all(keyword not in row[clinics_column_name] for keyword in inpatient_name_list) 
                                    ) else '', axis=1)
        
        
        new_df = df[['All Patients', 'ICU', 'Inpatient clinics', 'Emergency', 'Outpatient clinics', clinics_column_name,  age_column_name, gender_column_name, sampling_time_column_name, process_column_name]].copy()

        new_df[sampling_time_column_name] = pd.to_datetime(new_df[sampling_time_column_name], errors='coerce', infer_datetime_format=True)
        new_df = new_df.sort_values(by=sampling_time_column_name)
        edited_new_df = new_df

        original_columns = edited_new_df.columns.tolist()
        rename_columns = {
            gender_column_name: 'Gender',
            age_column_name: 'Age',
            sampling_time_column_name: 'Sampling Time',
            process_column_name: 'Results'
        }

        final_df = edited_new_df[original_columns].copy()
        final_df.columns = [rename_columns.get(col, col) for col in final_df.columns]

        patient_filtered_df = final_df.copy()
        if filepath.endswith('.csv'):
            csv_filepath = filepath.replace('.csv', '_converted.xlsx')
        elif filepath.endswith('.xlsx'):
            csv_filepath = filepath.replace('.xlsx', '_converted.csv')
         
        patient_filtered_df.to_csv(csv_filepath, index=False)
    
    
    if patient_category == 'All Patients':
        patient_filtered_df = final_df[final_df['All Patients'] == '+']
    else:
        patient_filtered_df = final_df[final_df[patient_category] == '+']
    
    patient_filtered_df.reset_index(inplace=True)
    print(patient_filtered_df)

    min_date = patient_filtered_df['Sampling Time'].min()
    max_date = patient_filtered_df['Sampling Time'].max()


    total_days = (max_date - min_date).days + 1

    # Find out how many weeks there are and calculate these weeks in 5 days
    total_weeks = total_days / 7
    effective_workdays = total_weeks * 5

    # Calculate the total number of patients
    total_patients = len(patient_filtered_df)
    patient_per_day = round(total_patients / effective_workdays)
    


    split_index = total_patients * allocation_ratio

    if split_index <= 0 or split_index >= len(patient_filtered_df):
        raise ValueError("Split index is out of range. Check allocation ratio and DataFrame size.")

    print(f"allocation_ratio : {allocation_ratio}")
    print(f"Split Index : {split_index}")

    first_half_df = patient_filtered_df.iloc[:(total_patients - int(split_index))]
    second_half_df = patient_filtered_df.iloc[int(split_index):]

    # Calculate Male and Female ratios for the first half
    gender_counts_first_half = first_half_df['Gender'].dropna().value_counts()
    total_first_half = gender_counts_first_half.sum()
    male_ratio_first_half = gender_counts_first_half.get('Male', 0) / total_first_half * 100
    female_ratio_first_half = gender_counts_first_half.get('Female', 0) / total_first_half * 100

    # Calculate Male and Female ratios for the second half
    gender_counts_second_half = second_half_df['Gender'].dropna().value_counts()
    total_second_half = gender_counts_second_half.sum()
    male_ratio_second_half = gender_counts_second_half.get('Male', 0) / total_second_half * 100
    female_ratio_second_half = gender_counts_second_half.get('Female', 0) / total_second_half * 100
    
    training_data_m_f_ratio = male_ratio_first_half/female_ratio_first_half
    test_data_m_f_ratio = male_ratio_second_half/female_ratio_second_half
    
    age_median_first_half = (first_half_df['Age'].dropna().median())
    age_median_second_half = (second_half_df['Age'].dropna().median())
    

    if selected_analyte == "Iron":
        patient_filtered_df["Results"] = patient_filtered_df["Results"] * 0.179
        

    return patient_filtered_df['Results'], total_patients, patient_per_day, effective_workdays, training_data_m_f_ratio, test_data_m_f_ratio, age_median_first_half, age_median_second_half



class NewPlotWindow(QDialog):
    def __init__(self, original_fig, canvas_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Data")
        self.setGeometry(200, 100, 1600, 1200)
        self.setWindowState(Qt.WindowMaximized)  # Open in full screen

        # Copy the original figure and enlarge it 3 times
        self.fig = copy.deepcopy(original_fig)
        original_size = self.fig.get_size_inches()
        original_dpi = self.fig.get_dpi()
        self.fig.set_size_inches(original_size[0] * 3, original_size[1] * 3)
        self.fig.set_dpi(original_dpi * 3)

        # Adjust spaces between plots
        self.fig.tight_layout(pad=2)  # Add padding to arrange plots
        self.fig.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.1, left=0.1, right=0.9, top=0.9)  # Adjust horizontal, vertical, bottom, and right spaces

        # Create canvas for the new figure
        self.canvas = FigureCanvas(self.fig)
        
        # Add right-click menu
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_context_menu)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_canvas_size()

    def adjust_canvas_size(self):
        if hasattr(self, 'fig'):
            size = self.size()
            dpi = self.fig.get_dpi()
            self.fig.set_size_inches(size.width() / dpi, size.height() / dpi, forward=True)
            
            scale_factor = min(size.width() / 1600, size.height() / 1200)  # Assuming original dimensions are 1600x1200
            self.adjust_text_size(scale_factor)
            self.adjust_marker_size(scale_factor)
            self.adjust_legend_size(scale_factor)
            
            self.canvas.draw()

    def adjust_text_size(self, scale_factor):
        for ax in self.fig.axes:
            ax.title.set_fontsize(5 * scale_factor)
            ax.xaxis.label.set_fontsize(4 * scale_factor)
            ax.yaxis.label.set_fontsize(4 * scale_factor)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(4 * scale_factor)
            for text in ax.texts:
                text.set_fontsize(4 * scale_factor)

    def adjust_marker_size(self, scale_factor):
        for ax in self.fig.axes:
            for line in ax.lines:
                line.set_markersize(2 * scale_factor)
            for collection in ax.collections:
                try:
                    collection.set_sizes([5 * scale_factor])
                except AttributeError:
                    pass

    def adjust_legend_size(self, scale_factor):
        for ax in self.fig.axes:
            legend = ax.get_legend()
            if legend:
                legend.get_title().set_fontsize(4 * scale_factor)
                for text in legend.get_texts():
                    text.set_fontsize(4 * scale_factor)

    def show_context_menu(self, pos):
        context_menu = QMenu(self)
        save_action = QAction('Save Figure', self)
        save_action.triggered.connect(self.save_figure)
        context_menu.addAction(save_action)
        context_menu.exec_(self.canvas.mapToGlobal(pos))

    def save_figure(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "", "PNG Files (*.png);;All Files (*)")
        if file_path:
            self.fig.savefig(file_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PBRTQC - Sadi Konuk Training and Research Hospital")
        self.setGeometry(100, 30, 1800, 960)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.training_data_canvas = Plot_Training_Data()
        self.control_limit_det_canvas = Plot_Control_Limit()
        self.performance_canvas = PlotPatientErrorDetection()
     
        self.setup_ui()
       
        self.df = None
  
        self.transformed_data = None
        self.selected_analyte = self.select_analyte_widget.currentText()
        self.lambda_value = None
        self.training_data_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.training_data_canvas.customContextMenuRequested.connect(lambda pos: self.show_context_menu(pos, self.training_data_canvas, 'training'))

        self.control_limit_det_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.control_limit_det_canvas.customContextMenuRequested.connect(lambda pos_2: self.show_context_menu(pos_2, self.control_limit_det_canvas, 'control'))

        self.performance_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.performance_canvas.customContextMenuRequested.connect(lambda pos_3: self.show_context_menu(pos_3, self.performance_canvas, 'performance'))
        
    def open_new_window(self, canvas_type):
        if canvas_type == 'training':
            self.new_plot_window = NewPlotWindow(self.training_data_canvas.fig, 'training', self)
        elif canvas_type == 'control':
            self.new_plot_window = NewPlotWindow(self.control_limit_det_canvas.fig, 'control', self)
        elif canvas_type == 'performance':
            self.new_plot_window = NewPlotWindow(self.performance_canvas.fig, 'performance', self)
        self.new_plot_window.show()

    def setup_ui(self):
        mainLayout = QHBoxLayout()

        # Training Data Processes Group Box
        groupBox_training_data = QGroupBox('Training Data Processes')
        layout_canvas = QVBoxLayout()

        self.patient_category_widget = QComboBox()
        self.patient_category_widget.addItems(["All Patients","ICU", "Inpatient clinics", "Emergency","Outpatient clinics"])
        index_pat = self.patient_category_widget.findText("All Patients", Qt.MatchFixedString)
        if index_pat >= 0:
            self.patient_category_widget.setCurrentIndex(index_pat)
        layout_canvas.addLayout(self.add_label_to_widget(self.patient_category_widget, "Admission Status"))

        self.selected_performance_parameters_widget = QComboBox()
        self.selected_performance_parameters_widget.addItems(["Desirable", "Minimum", "Optimum"])
        layout_canvas.addLayout(self.add_label_to_widget(self.selected_performance_parameters_widget, "Select Performance Limits"))

        self.allocation_widget = QComboBox()
        self.allocation_widget.addItems(["50:50", "30:70", "40:60"])
        layout_canvas.addLayout(self.add_label_to_widget(self.allocation_widget, "Allocation (Training/Test Dataset)"))
          
        self.patient_per_day_widget = QSpinBox()
        self.patient_per_day_widget.setMaximum(2500)
        self.patient_per_day_widget.setValue(1000)
        layout_canvas.addLayout(self.add_label_to_widget(self.patient_per_day_widget, "Patient Per Day"))
        
        sorted_analytes = sorted(Analytes_List.keys())

      
        self.select_analyte_widget = QComboBox()
        self.select_analyte_widget.addItems(sorted_analytes)

        index = self.select_analyte_widget.findText("AST", Qt.MatchFixedString)
        if index >= 0:
            self.select_analyte_widget.setCurrentIndex(index)

       
        self.select_analyte_widget.currentIndexChanged.connect(self.update_training_data_plot)


        layout_canvas.addLayout(self.add_label_to_widget(self.select_analyte_widget, "Select Analyte"))

        self.auto_pbrtqc_button_widget = QPushButton("Auto PBRTQC")
        self.auto_pbrtqc_button_widget.clicked.connect(self.pick_automatic_PBRTQC)
        layout_canvas.addLayout(self.add_label_to_widget(self.auto_pbrtqc_button_widget, ""))

        self.select_button_widget = QPushButton("Select Excel/CSV File")
        self.select_button_widget.clicked.connect(self.select_csv)
        layout_canvas.addLayout(self.add_label_to_widget(self.select_button_widget, ""))

        self.exclusion_parameter_widget = QComboBox()
        self.exclusion_parameter_widget.addItems(["No Exclusion", "Trimming", "Winsorization"])
        self.exclusion_parameter_widget.currentIndexChanged.connect(self.update_training_data_plot)
        layout_canvas.addLayout(self.add_label_to_widget(self.exclusion_parameter_widget, "Exclusion Parameters"))

        self.truncation_parameter_widget = QComboBox()
        self.truncation_parameter_widget.addItems(["No Truncation", "RCVg",  "1%-99%", "5%-95%",  "3*SD"])
        self.truncation_parameter_widget.currentIndexChanged.connect(self.update_training_data_plot)
        layout_canvas.addLayout(self.add_label_to_widget(self.truncation_parameter_widget, "Truncation Parameters"))

        self.transformation_parameter_widget = QComboBox()
        self.transformation_parameter_widget.addItems(["No Transformation", "Yeo-Johnson",  "Standard Scale", "Min-Max Scale", "Robust Scale", "Square Root", "Square", "Ln", "Log10"])
        self.transformation_parameter_widget.currentIndexChanged.connect(self.update_training_data_plot)
        layout_canvas.addLayout(self.add_label_to_widget(self.transformation_parameter_widget, "Transformation Parameters"))

        self.convert_factor_widget = QLineEdit()
        double_validator = QDoubleValidator()
        double_validator.setNotation(QDoubleValidator.StandardNotation)
        self.convert_factor_widget.setValidator(double_validator)
        self.convert_factor_widget.setText("1")
        self.convert_factor_widget.textChanged.connect(self.update_training_data_plot)
        layout_canvas.addLayout(self.add_label_to_widget(self.convert_factor_widget, "Convert Factor"))

        

        layout_canvas.addWidget(self.training_data_canvas)
        groupBox_training_data.setLayout(layout_canvas)
        mainLayout.addWidget(groupBox_training_data, 1)  

        rightLayout = QVBoxLayout()

        control_limit_data = QGroupBox('Control Limits Determination')
        control_layout_canvas = QVBoxLayout()

        self.block_size_widget = QSpinBox()
        self.block_size_widget.setMaximum(250)
        self.block_size_widget.setValue(20)
        control_layout_canvas.addLayout(self.add_label_to_widget(self.block_size_widget, "Block Size"))

        self.statistical_parameter_widget = QComboBox()
        self.statistical_parameter_widget.addItems(["EWMA-Decay factor","Moving Mean", "Moving Median", "Moving Std"])
        control_layout_canvas.addLayout(self.add_label_to_widget(self.statistical_parameter_widget, "Statistical Parameters"))

        self.control_limit_type_widget = QComboBox()
        self.control_limit_type_widget.addItems([  "Mean of Daily 1th-99th Percentiles", "1st-99th Percentile Values of Entire Data", 
                                                 "5th-95th Percentile Values of Entire Data", "0.25th-99.75th Percentile Values of Entire Data",
                                                  "5th-95th Percentiles of Daily Min-Max Limits",  "New Daily CLs"])
        self.control_limit_type_widget.currentIndexChanged.connect(self.update_control_limit_plot)
        control_layout_canvas.addLayout(self.add_label_to_widget(self.control_limit_type_widget, "Select Control Limit"))
        
        self.create_control_button_widget = QPushButton("Create Control Limits")
        self.create_control_button_widget.clicked.connect(self.update_control_limit_plot)
        control_layout_canvas.addLayout(self.add_label_to_widget(self.create_control_button_widget, ""))

        control_layout_canvas.addWidget(self.control_limit_det_canvas)
        control_limit_data.setLayout(control_layout_canvas)
        rightLayout.addWidget(control_limit_data, 1) 

        # Performance of PBRTQC Group Box
        performance_limit_data = QGroupBox('Performance of PBRTQC')
        performance_layout_canvas = QVBoxLayout()

        self.biased_point_widget = QSpinBox()
        self.biased_point_widget.setMaximum(1000)
        self.biased_point_widget.setValue(100)
        performance_layout_canvas.addLayout(self.add_label_to_widget(self.biased_point_widget, "Bias Adding Point"))
        

        



        self.create_MNPed_button_widget = QPushButton("Create MNPed Plot")
        self.create_MNPed_button_widget.clicked.connect(self.draw_NPed_plot)
        performance_layout_canvas.addLayout(self.add_label_to_widget(self.create_MNPed_button_widget, ""))

        self.create_edited_NPed_button = QPushButton("Compare Performance Plots")
        self.create_edited_NPed_button.clicked.connect(self.draw_performance_plot)
        performance_layout_canvas.addLayout(self.add_label_to_widget(self.create_edited_NPed_button, ""))

        


        performance_layout_canvas.addWidget(self.performance_canvas)
        performance_limit_data.setLayout(performance_layout_canvas)
        rightLayout.addWidget(performance_limit_data, 1) 

        rightContainer = QWidget()
        rightContainer.setLayout(rightLayout)
        mainLayout.addWidget(rightContainer, 2) 

        self.central_widget.setLayout(mainLayout)
        
    def add_label_to_widget(self, widget, label_text):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        layout.addWidget(label)
        layout.addWidget(widget)
        return layout

    def show_context_menu(self, pos, canvas, canvas_type):
        context_menu = QMenu(self)
        save_action = context_menu.addAction("Save Image")
        new_window_action = context_menu.addAction("Open in New Window")
        global_pos = canvas.mapToGlobal(pos)
    
        action = context_menu.exec_(global_pos)

        if action == save_action:
            self.save_image(canvas)
        elif action == new_window_action:
            self.open_new_window(canvas_type)
            
    def save_image(self, canvas):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg *.tiff)", options=options)
        if file_name:
            canvas.fig.savefig(file_name, dpi=600)

    def set_analyte_values(self):
        self.selected_analyte = self.select_analyte_widget.currentText()
        self.selected_performance_parameter = self.selected_performance_parameters_widget.currentText()
        print(self.selected_analyte)
        if self.selected_analyte in Analytes_List:
            analyte_info = Analytes_List[self.selected_analyte]
            self.selected_limit_cvi = analyte_info['CVi']
            self.selected_limit_cvg = analyte_info['CVg']
            self.selected_limit_ii = analyte_info['Index of Individuality']

            if self.selected_performance_parameter == 'Minimum':            
               self.selected_limit_RCVg = analyte_info['RCVg (Minimum)']
               self.selected_limit_TEa = analyte_info['TEa (Minimum)']
               
            if self.selected_performance_parameter == 'Desirable':             
               self.selected_limit_RCVg = analyte_info['RCVg (Desirable)']
               self.selected_limit_TEa = analyte_info['TEa (Desirable)']
               
            if self.selected_performance_parameter == 'Optimum':        
               self.selected_limit_RCVg = analyte_info['RCVg (Optimum)']
               self.selected_limit_TEa = analyte_info['TEa (Optimum)']
            
            self.selected_limit_BVi = analyte_info['CVi']
            self.selected_limit_BVg = analyte_info['CVg']
            
            self.selected_limit_CVa = analyte_info['Imprecision (CVa)']
            self.selected_limit_bias = analyte_info['Bias']
            
            self.measurement_lower_limit = analyte_info['Measurable Lower Limit']
            self.measurement_upper_limit = analyte_info['Measurable Upper Limit']
            self.allowable_bias_array = np.array([-self.selected_limit_bias, self.selected_limit_bias])
        else:
            QMessageBox.critical(self, "Error", "Invalid analyte selected.")
    
    def pick_automatic_PBRTQC(self):
        

        self.set_analyte_values()
        
        current_text = self.allocation_widget.currentText()
        if current_text == "50:50":
            self.allocation_ratio = 0.5
        elif current_text == "30:70":
            self.allocation_ratio = 0.3
        elif current_text == "40:60":
            self.allocation_ratio = 0.4
        else:
            self.allocation_ratio = 0.5
            
        auto_process = automatic_PBRTQC(self.patient_category_widget.currentText(),
                                        self.selected_analyte, date_column_name, process_column_name,self.selected_limit_RCVg, 
                                        self.selected_limit_TEa, self.allowable_bias_array,
                                        self.selected_limit_BVi, self.selected_limit_BVg, self.selected_limit_CVa, 
                                        self.selected_limit_bias, self.measurement_lower_limit, self.measurement_upper_limit,
                                        self.patient_per_day_widget.value(), self.allocation_ratio, self.biased_point_widget.value(), 
                                        float(self.convert_factor_widget.text()), self.control_limit_type_widget.currentText() )  

        
        auto_process.starting_automatic_process()    
    
    def select_csv(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        self.set_analyte_values()

        current_text = self.allocation_widget.currentText()
        if current_text == "50:50":
            self.allocation_ratio = 0.5
        elif current_text == "30:70":
            self.allocation_ratio = 0.3
        elif current_text == "40:60":
            self.allocation_ratio = 0.4
        else:
            self.allocation_ratio = 0.5
            
        self.df,  self.total_patients, _, _, self.training_data_m_f_ratio, self.test_data_m_f_ratio, self.training_data_age_median, self.test_data_age_median  = process_data_file(self.patient_category_widget.currentText(),                                                                                                                                                                            
        self.allocation_ratio, self.measurement_lower_limit,self.measurement_upper_limit, gender_column_name, age_column_name, sampling_time_column_name, clinics_column_name,
        icu_name_list, inpatient_name_list, emergency_name_list, "Results", self.selected_analyte)

                                 
        self.patient_per_day = self.patient_per_day_widget.value() 
        self.total_days = self.total_patients // self.patient_per_day

        
        self.list_day_and_data()
  
    

    def list_day_and_data(self):
        
            
        self.first_half_series, self.second_half_series = df_to_list(self.df, self.allocation_ratio, 1-self.allocation_ratio, self.patient_per_day, return_type='pandas', exclude_last_list=True)
        
        self.first_half_series_copy = self.first_half_series.copy()
        self.second_half_series_copy = self.second_half_series.copy()
        
        print(f"\nTraining Pandas Series (Entire Data: {type(self.first_half_series_copy)}),(List Data: {type(self.first_half_series_copy[0])}):")
        print(self.first_half_series_copy)
        print(f"\nTest Pandas Series (Entire Data: {type(self.second_half_series_copy)}),(List Data: {type(self.second_half_series_copy[0])}):")
        print(self.second_half_series_copy)
        
        self.original_shape_training_data = [len(lst) for lst in self.first_half_series_copy]
        self.original_shape_test_data = [len(lst) for lst in self.second_half_series_copy]
        
        self.flattened_training_data = convert_list_to_array(self.first_half_series_copy, "Training Data Array")
        self.flattened_test_data = convert_list_to_array(self.second_half_series_copy, "Test Data Series Array")
        
        Q1_training_data = np.percentile(self.flattened_training_data, 25)
        Q3_training_data = np.percentile(self.flattened_training_data, 75)
        self.IQR_training_data = Q3_training_data - Q1_training_data
  
        Q1_test_data = np.percentile(self.flattened_test_data, 25)
        Q3_test_data = np.percentile(self.flattened_test_data, 75)
        self.IQR_test_data = Q3_test_data - Q1_test_data
        
        self.update_training_data_plot()
        

    def update_training_data_plot(self):
        
        self.set_analyte_values()
        
        print("---------------START--------------------\n")
        print("------------------------------------------------------------------------------------------------------------------------")
        if self.first_half_series_copy is not None and self.second_half_series_copy is not None:
            
            truncation_of_first_half = Truncation(
                self.flattened_training_data, 
                self.truncation_parameter_widget.currentText(), 
                self.exclusion_parameter_widget.currentText(), 
                self.selected_limit_RCVg)
            
            self.training_lower_trun_limit, self.training_upper_trun_limit, self.z_score_training = truncation_of_first_half.determine_truncation_limit()
            self.truncated_first_half = truncation_of_first_half.truncate_data(
                self.training_lower_trun_limit, 
                self.training_upper_trun_limit, 
                self.block_size_widget.value(), 
                self.z_score_training)

            transformation_of_first_half = Transformation(
                self.selected_analyte, 
                self.truncated_first_half, 
                self.transformation_parameter_widget.currentText(), 
                float(self.convert_factor_widget.text()))
            self.transformed_training_data, self.lambda_value_training = transformation_of_first_half.transform_data()
            
            
            truncation_of_second_half = Truncation(
                self.flattened_test_data, 
                self.truncation_parameter_widget.currentText(), 
                self.exclusion_parameter_widget.currentText(), 
                self.selected_limit_RCVg)
            
            
            self.truncated_second_half = truncation_of_second_half.truncate_data(
                self.training_lower_trun_limit, 
                self.training_upper_trun_limit, 
                self.block_size_widget.value(), 
                self.z_score_training)

            transformation_of_second_half = Transformation(
                self.selected_analyte, 
                self.truncated_second_half, 
                self.transformation_parameter_widget.currentText(), 
                float(self.convert_factor_widget.text()), self.lambda_value_training
            )
            self.transformed_test_data, _ = transformation_of_second_half.transform_data()
            
            median_first_half = np.median(self.transformed_training_data)
            iqr_first_half = np.percentile(self.transformed_training_data, 75) - np.percentile(self.transformed_training_data, 25)
            mad_training_MNPed = np.median(np.abs(self.transformed_training_data - median_first_half))
            # İlk yarı için RNS hesaplayın
            if median_first_half != 0:
                self.rns_first_half = iqr_first_half / median_first_half
                self.mad_training_data = 1.4826 * mad_training_MNPed / median_first_half
            else:
                self.rns_first_half = np.nan
                self.mad_training_data = np.nan


            median_second_half = np.median(self.transformed_test_data)
            iqr_second_half = np.percentile(self.transformed_test_data, 75) - np.percentile(self.transformed_test_data, 25)
            mad_test_MNPed = np.median(np.abs(self.transformed_test_data - median_second_half))

            if median_second_half != 0:
                self.rns_second_half = iqr_second_half / median_second_half
                self.mad_test_data = 1.4826 * mad_test_MNPed / median_second_half
            else:
                self.rns_second_half = np.nan
                self.mad_test_data = np.nan

            
            self.transformed_daily_series_first = convert_array_to_list(self.transformed_training_data, self.original_shape_training_data, 'numpy_list', "self.transformed_daily_series_first")
            self.transformed_daily_series_second = convert_array_to_list(self.transformed_test_data, self.original_shape_test_data, 'numpy_list', "self.transformed_daily_series_second")         
            print(f"Training dataset Min-Max Values : {min(self.transformed_training_data)}-{max(self.transformed_training_data)}")
            print(f"Test dataset Min-Max Values : {min(self.transformed_test_data)}-{max(self.transformed_test_data)}")
            
            print("\nRobust CV IQR for Training Data Array:")
            print(round(self.rns_first_half,2))
            print("Robust CV IQR for Test Data Array:")
            print(round(self.rns_second_half,2))
            
            print("\nRobust CV MAD for Training Data Array:")
            print(round(self.mad_training_data,2))
            print("Robust CV MAD for Test Data Array:")
            print(round(self.mad_test_data,2))
            print(f"training_male_female_ratio : {self.training_data_m_f_ratio:.2f}")
            print(f"test_male_female_ratio : {self.test_data_m_f_ratio:.2f}\n")
            
            print(f"Training Dataset({self.selected_analyte}):\n")
            print(f"{len(self.flattened_training_data)}\n{self.total_days//2}\n{self.training_data_age_median}\n{self.training_data_m_f_ratio:.2f}\n{np.median(self.flattened_training_data):.2f}\n{self.IQR_training_data:.2f}\n{round(self.mad_training_data,2)}\n")
            
            print(f"Test Dataset({self.selected_analyte}):\n")
            print(f"{len(self.flattened_test_data)}\n{self.total_days//2}\n{self.test_data_age_median}\n{self.test_data_m_f_ratio:.2f}\n{np.median(self.flattened_test_data):.2f}\n{self.IQR_test_data:.2f}\n{round(self.mad_test_data,2)}\n")
            
   
            self.training_data_canvas.training_and_test_data_plot(self.selected_analyte, self.transformed_daily_series_first, 
                                                                  self.transformed_daily_series_second, self.transformation_parameter_widget.currentText(), 
                                                                  self.truncation_parameter_widget.currentText(), self.exclusion_parameter_widget.currentText(),
                                                                  self.lambda_value_training, self.mad_training_data, self.mad_test_data,)
          
            
    def update_control_limit_plot(self):
        
        ############################ STATISTICAL DATA CHECK ############################

        statistical_process_for_control_limit = StatisticalProcess(
                    self.transformed_daily_series_first, 
                    self.statistical_parameter_widget.currentText(), 
                    self.block_size_widget.value(), 
                    self.z_score_training, keep_nan_index=True
                )
        daily_statistical_data = statistical_process_for_control_limit.statistical_process_data()              

        daily_statistical_data_without_nan = remove_nan_from_lists(daily_statistical_data, "daily_statistical_data_without_nan cleaning NaN")
                  
        print("Applied statistical process for transformed training data: ")
        for index, day_data in daily_statistical_data_without_nan.head(5).items():
            print(f"Gün {index}: Statistical Liste Uzunluğu = {len(day_data)}")
        
  
        print("------------------------------------------------------------------------------------------------------------------------")
        print("---------------END--------------------\n")

        
        cl_data_class = ControlLimitDetermination(daily_statistical_data_without_nan, self.control_limit_type_widget.currentText())
                            
        (self.daily_data_parameters_list, 
         self.daily_min,  self.daily_max, self.selected_lower_control_limit, self.selected_upper_control_limit, self.total_exceedances,
         self.daily_min_5_percentile_limit, self.daily_max_95_percentile_limit, self.new_daily_lower_limit, self.new_daily_upper_limit,
         self.daily_1_percentile_mean_limit, self.daily_99_percentile_mean_limit, self.whole_data_025_quantile_limit, self.whole_data_975_quantile_limit,
         self.whole_data_1_quantile_limit, self.whole_data_99_quantile_limit, self.whole_data_5_quantile_limit, self.whole_data_95_quantile_limit,
         self.total_exceedances_min_max_quantiles_daily, self.total_exceedances_new_daily, self.total_exceedances_025_975_quantiles_whole, self.total_exceedances_mean_1_99_quantiles_daily, 
         self.total_exceedances_1_99_quantiles_whole, self.total_exceedances_5_95_quantiles_whole, 
         self.cv_lower_third, self.cv_middle_third, self.cv_upper_third) = cl_data_class.calculate_control_limits()
        
        self.control_limit_det_canvas.control_limit_plot(daily_statistical_data_without_nan, self.daily_data_parameters_list, self.selected_analyte, 
                                                         self.training_lower_trun_limit, self.training_upper_trun_limit, 
                                                         self.daily_min,  self.daily_max, self.daily_min_5_percentile_limit, self.daily_max_95_percentile_limit, self.new_daily_lower_limit, self.new_daily_upper_limit,
                                                         self.daily_1_percentile_mean_limit, self.daily_99_percentile_mean_limit, self.whole_data_025_quantile_limit, self.whole_data_975_quantile_limit,
                                                         self.whole_data_1_quantile_limit, self.whole_data_99_quantile_limit, self.whole_data_5_quantile_limit, self.whole_data_95_quantile_limit,
                                                         self.total_exceedances_min_max_quantiles_daily, self.total_exceedances_new_daily, self.total_exceedances_025_975_quantiles_whole, self.total_exceedances_mean_1_99_quantiles_daily, 
                                                         self.total_exceedances_1_99_quantiles_whole, self.total_exceedances_5_95_quantiles_whole,
                                                         self.transformation_parameter_widget.currentText(),self.truncation_parameter_widget.currentText(), 
                                                         self.exclusion_parameter_widget.currentText(), self.statistical_parameter_widget.currentText(), 
                                                         self.block_size_widget.value(), self.patient_per_day_widget.value(), self.training_lower_trun_limit, 
                                                         self.training_upper_trun_limit, self.control_limit_type_widget.currentText()
                                                         )
                                                         
                                                                                                                       
    def draw_NPed_plot(self):

        performance_of_last_half_data = PBRTQC_Biased(len(self.flattened_test_data), self.second_half_series, self.patient_category_widget.currentText(),
                                                      self.allocation_widget.currentText(),
                                                      self.training_data_m_f_ratio, self.test_data_m_f_ratio, self.training_data_age_median, self.test_data_age_median,
                                                      self.training_lower_trun_limit, self.training_upper_trun_limit,
                                                      self.selected_limit_RCVg, self.selected_analyte, self.truncation_parameter_widget.currentText(), 
                                                      self.exclusion_parameter_widget.currentText(), self.transformation_parameter_widget.currentText(),
                                                      self.statistical_parameter_widget.currentText(), self.block_size_widget.value(),
                                                      self.biased_point_widget.value(), self.patient_per_day_widget.value(), 
                                                      self.control_limit_type_widget.currentText(), self.selected_lower_control_limit, self.selected_upper_control_limit,
                                                      self.cv_lower_third, self.cv_middle_third, self.cv_upper_third,
                                                      self.lambda_value_training, float(self.convert_factor_widget.text()), self.z_score_training)
      
        
        
        false_positive_rate, sum_of_MNPed, area_NPed, area_NPed_multiplied, self.data_for_plot_manuel = performance_of_last_half_data.counting_saving_NPed_manuel(self.selected_analyte)
        self.performance_canvas.draw_MNPed_Plot(self.data_for_plot_manuel, false_positive_rate, sum_of_MNPed,
                                                area_NPed, area_NPed_multiplied, 
                                                self.select_analyte_widget.currentText(), self.control_limit_type_widget.currentText(),
                                                self.truncation_parameter_widget.currentText(),self.training_lower_trun_limit, self.training_upper_trun_limit,                                                
                                                self.transformation_parameter_widget.currentText(),
                                                self.statistical_parameter_widget.currentText(), self.block_size_widget.value(), self.selected_lower_control_limit, self.selected_upper_control_limit,
                                                self.mad_test_data,
                                                self.patient_per_day_widget.value(), 
                                                self.biased_point_widget.value(),)


    def draw_performance_plot(self):
        self.set_analyte_values()
        
        file_path = filedialog.askopenfilename(title="Select Performance data to Plot", filetypes=[("Excel Files", "*.xlsx")])
        predata = pd.read_excel(file_path)
         
        
        all_benchmarking = [ "Lowest Sum of MNPed"] #"Highest Sum of ROC AUC", "Lowest Total AUC (x Bias)", "Robust CV MAD",
       
        performance_metric = "MNPed" #"Lowest Total AUC (x Bias)" #, "Highest Sum of Accuracy", "Highest Sum of Sensitivity", "Highest Sum of ROC AUC",  
                   
        for selected_benchmarking in all_benchmarking:
            if selected_benchmarking == "Highest Sum of Accuracy" or selected_benchmarking == "Highest Sum of Accuracy (In Allowable Bias)":
                another_performance_metric = "accuracy_points"
                another_performance_metric_label = "Accuracy Median"
            elif selected_benchmarking == "Highest Sum of Sensitivity":  
                another_performance_metric = "sensitivity_points"
                another_performance_metric_label = "Sensitivity Median"
            else :
                another_performance_metric = "ROC_points"
                another_performance_metric_label = "ROC AUC Median"
            
            another_performance_metric = "accuracy_points"
            another_performance_metric_label = "Accuracy Median"
                       
            performance_plots = GroupedDataPlotter(predata, self.selected_limit_TEa, self.selected_limit_cvi , self.selected_limit_cvg, 
                                                   self.selected_limit_ii, self.selected_limit_RCVg, self.selected_limit_TEa, 
                                                   selected_benchmarking, performance_metric,
                                                   another_performance_metric, another_performance_metric_label ,5, 1, self.control_limit_type_widget.currentText(), reverse=False)        
            performance_plots.plot_grouped_data_from_excel()

        print("All benchmarking groups completed...")
    

# DETERMINATION OF TRUNCATION LIMITS STEP                                         
class Truncation:
    def __init__(self, data, truncation_param, exclusion_param, limit_RCVg_desirable_value):
        self.data = data
        self.truncation_param = truncation_param
        self.exclusion_param = exclusion_param
        self.selected_limit_RCVg = limit_RCVg_desirable_value
         
    def determine_truncation_limit(self):
        lower_limit = None
        upper_limit = None
        mean_data = np.nanmean(self.data)
        std_data = np.nanstd(self.data)
        
        if self.truncation_param == "No Truncation":
            upper_limit = np.nanmax(self.data)
            lower_limit = np.nanmin(self.data)
            z_score = (upper_limit - mean_data) / std_data
            
        elif self.truncation_param == "RCVg":
            factor = mean_data * (self.selected_limit_RCVg / 100)
            upper_limit = mean_data + factor
            lower_limit = mean_data - factor
            z_score = (upper_limit - mean_data) / std_data

        elif self.truncation_param == "5%-95%":
            lower_quantile = 0.05
            upper_quantile = 0.95
            lower_limit = np.nanquantile(self.data, lower_quantile)
            upper_limit = np.nanquantile(self.data, upper_quantile)
            z_score = (upper_limit - mean_data) / std_data
            
        elif self.truncation_param == "1%-99%":
            lower_quantile = 0.01
            upper_quantile = 0.99
            lower_limit = np.nanquantile(self.data, lower_quantile)
            upper_limit = np.nanquantile(self.data, upper_quantile)
            z_score = (upper_limit - mean_data) / std_data
              
        elif self.truncation_param == "3*SD":

            lower_limit = mean_data - 3 * std_data
            upper_limit = mean_data + 3 * std_data
            z_score = (upper_limit - mean_data) / std_data
        

        if lower_limit < 0:
           lower_limit = self.data.min() 
        return lower_limit, upper_limit, z_score

    def truncate_data(self, trun_lower_limit, trun_upper_limit, block_size, n_sigmas):
        truncated_data = np.copy(self.data)
        
        nan_indices = np.isnan(truncated_data)
        
        if self.truncation_param == "No Truncation":
            return truncated_data
        else:
            if self.exclusion_param == "Trimming":
                # Truncation işlemi ile sınırlar dışındaki değerleri NaN yap
                truncation_indices = (truncated_data > trun_upper_limit) | (truncated_data < trun_lower_limit)
                truncated_data[truncation_indices] = np.nan
            elif self.exclusion_param == "Winsorization":
                truncated_data = np.clip(truncated_data, trun_lower_limit, trun_upper_limit)
        
        truncated_data[nan_indices] = np.nan
        
        # Non-NaN data check after truncation
        if np.all(np.isnan(truncated_data)):
            print("Warning: All data became NaN after TRUNCATION.")
        
        return truncated_data   


# DETERMINATION OF TRANSFORMATION STEP 
class Transformation:
    def __init__(self, selected_analyte, data, transformation_param, convert_factor, lambda_value=None):
        self.selected_analyte = selected_analyte
        self.data = data  # Liste olarak veri
        self.transformation_parameter = transformation_param
        self.lambda_value = lambda_value
        self.convert_factor = convert_factor

    def transform_data(self):
        
        data_array = np.array(self.data)
        nan_indices = np.isnan(data_array)
        
        data_no_nan = data_array[~nan_indices] * self.convert_factor
        lambda_values_data = []
        
        if self.transformation_parameter == "No Transformation":
            transformed_data_no_nan = data_no_nan
        elif self.transformation_parameter == "Yeo-Johnson":
            if self.lambda_value is not None:
                lambda_values_data.append(self.lambda_value)
                transformed_data_no_nan = stats.yeojohnson(data_no_nan, lmbda=self.lambda_value)
            else:
                transformed_data_no_nan, self.lambda_value = stats.yeojohnson(data_no_nan)
                lambda_values_data.append(self.lambda_value)
        elif self.transformation_parameter == "Square Root":
            transformed_data_no_nan = np.sqrt(data_no_nan)
        elif self.transformation_parameter == "Log10":
            transformed_data_no_nan = np.log10(data_no_nan)
        else:
            raise ValueError("Invalid transformation parameter")
        
        self.transformed_data = np.full_like(data_array, np.nan, dtype=np.float64)
        self.transformed_data[~nan_indices] = transformed_data_no_nan
        
        return self.transformed_data, self.lambda_value

# DETERMINATION OF MOVING STATISTICAL METHOD STEP             
class StatisticalProcess:
    def __init__(self, data, statistical_param, block_size, z_score, alpha_value=None, keep_nan_index=True):
        self.data = data 
        self.statistical_param = statistical_param
        self.block_size = block_size
        self.alpha_value = alpha_value
        self.keep_nan_index = keep_nan_index

    def remove_nan_and_process(self, array_data):

        nan_indices = np.isnan(array_data)

        data_no_nan = array_data[~nan_indices]
        
        if self.statistical_param == "EWMA-Decay factor":
            processed_data_no_nan = pd.Series(data_no_nan).ewm(span=self.block_size, adjust=False).mean().values
        elif self.statistical_param == "Moving Mean":
            processed_data_no_nan = pd.Series(data_no_nan).rolling(window=self.block_size).mean().values
        elif self.statistical_param == "Moving Median":
            processed_data_no_nan = pd.Series(data_no_nan).rolling(window=self.block_size).median().values
        elif self.statistical_param == "Moving Std":
            processed_data_no_nan = pd.Series(data_no_nan).rolling(window=self.block_size).std().values
        else:
            raise ValueError(f"Unknown statistical parameter: {self.statistical_param}")
        
        if self.keep_nan_index:
            processed_full_data = np.full_like(array_data, np.nan, dtype=np.float64)
            processed_full_data[~nan_indices] = processed_data_no_nan
            return processed_full_data
        else:
            return processed_data_no_nan

    def statistical_process_data(self):
        processed_series = []
        
        for array_data in self.data:
            processed_data = self.remove_nan_and_process(array_data)
            processed_series.append(processed_data)

        return pd.Series(processed_series)

# DETERMINATION OF CONTROL LIMITS FOLLOWING MOVING STATISTICAL METHOD STEP     
class ControlLimitDetermination:
       
    def __init__(self, daily_statistical_data_series, control_limit_type):
        
        self.daily_statistical_data_series = daily_statistical_data_series
        self.control_limit_type = control_limit_type
        self.total_days = len(daily_statistical_data_series)
        
    def calculate_control_limits(self):         

        daily_data_list = []
        
        flattened_array = convert_list_to_array(self.daily_statistical_data_series)
        
        print(flattened_array)
        for i, day_data in enumerate(self.daily_statistical_data_series):
            daily_data_list.append({
                'mean': np.mean(day_data),
                'std': np.std(day_data),
                'min': np.min(day_data),
                'max': np.max(day_data),
                '1_percentile': np.quantile(day_data, 0.01),
                '99_percentile': np.quantile(day_data,0.99),
                '5_percentile': np.quantile(day_data, 0.05),
                '95_percentile': np.quantile(day_data,0.95)                
            })
        
        
        daily_data_parameters = pd.DataFrame(daily_data_list)

        daily_min = daily_data_parameters['min']
        daily_max = daily_data_parameters['max']
        daily_std = daily_data_parameters['std']
        daily_1_percentile = daily_data_parameters['1_percentile']
        daily_99_percentile = daily_data_parameters['99_percentile']
        
        daily_min_5_percentile_limit = daily_min.quantile(0.05)
        daily_max_95_percentile_limit = daily_max.quantile(0.95)

        daily_1_percentile_mean_limit = daily_1_percentile.mean()
        daily_99_percentile_mean_limit = daily_99_percentile.mean()
        
        
        whole_data_mean = flattened_array.mean() 
        whole_data_std = flattened_array.std()
        whole_data_min = flattened_array.min()
        whole_data_max = flattened_array.max()
        whole_data_025_quantile_limit =  np.percentile(flattened_array, 0.25)
        whole_data_975_quantile_limit = np.percentile(flattened_array, 99.75)
        whole_data_1_quantile = np.percentile(flattened_array, 1)
        whole_data_99_quantile = np.percentile(flattened_array, 99)
        whole_data_5_quantile = np.percentile(flattened_array, 5)
        whole_data_95_quantile = np.percentile(flattened_array, 95)
       
        new_daily_lower_limit, new_daily_upper_limit = self.print_new_limits(daily_min_5_percentile_limit,daily_max_95_percentile_limit)        

        all_data = pd.concat([pd.Series(d) for d in self.daily_statistical_data_series]).sort_values()
        lower_third = all_data[:len(all_data) // 3]
        middle_third = all_data[len(all_data) // 3:2 * len(all_data) // 3]
        upper_third = all_data[2 * len(all_data) // 3:]
        
        cv_lower_third = (np.std(lower_third) / np.mean(lower_third)) * 100 if np.mean(lower_third) != 0 else np.nan
        cv_middle_third = (np.std(middle_third) / np.mean(middle_third)) * 100 if np.mean(middle_third) != 0 else np.nan
        cv_upper_third = (np.std(upper_third) / np.mean(upper_third)) * 100 if np.mean(upper_third) != 0 else np.nan
        
        exceedances_daily_min_5_percentile_limit = (daily_min < daily_min_5_percentile_limit).sum()
        exceedances_daily_max_95_percentile_limit = (daily_max > daily_max_95_percentile_limit).sum()
        
        exceedances_daily_new_lower_limit = (daily_min < new_daily_lower_limit).sum()
        exceedances_daily_new_upper_limit = (daily_max > new_daily_upper_limit).sum()
        
        exceedances_daily_mean_1_percentile_limit = (daily_min < daily_1_percentile_mean_limit).sum()
        exceedances_daily_mean_99_percentile_limit = (daily_max > daily_99_percentile_mean_limit).sum()
        
        exceedances_whole_data_025_quantile_limit = (daily_min < whole_data_025_quantile_limit).sum()
        exceedances_whole_data_975_quantile_limit = (daily_max > whole_data_975_quantile_limit).sum()

        exceedances_whole_data_1_quantile_limit = (flattened_array < whole_data_1_quantile).sum()
        exceedances_whole_data_99_quantile_limit = (flattened_array > whole_data_99_quantile).sum()
        
        exceedances_whole_data_5_quantile_limit = (flattened_array < whole_data_5_quantile).sum()
        exceedances_whole_data_95_quantile_limit = (flattened_array > whole_data_95_quantile).sum()

        total_exceedances_min_max_quantiles_daily = ((exceedances_daily_min_5_percentile_limit + exceedances_daily_max_95_percentile_limit) / (self.total_days)) * 100
        total_exceedances_new_daily = ((exceedances_daily_new_lower_limit + exceedances_daily_new_upper_limit) / (self.total_days)) * 100      
        total_exceedances_mean_1_99_quantiles_daily = ((exceedances_daily_mean_1_percentile_limit + exceedances_daily_mean_99_percentile_limit) / (self.total_days)) * 100
        total_exceedances_025_975_quantiles_whole = ((exceedances_whole_data_025_quantile_limit + exceedances_whole_data_975_quantile_limit) / (len(flattened_array))) * 100
        total_exceedances_1_99_quantiles_whole = ((exceedances_whole_data_1_quantile_limit + exceedances_whole_data_99_quantile_limit) / (len(flattened_array))) * 100
        total_exceedances_5_95_quantiles_whole = ((exceedances_whole_data_5_quantile_limit + exceedances_whole_data_95_quantile_limit) / (len(flattened_array))) * 100

        
        print(f'Daily Data SD Column Mean : {daily_std.mean():.2f}\n')
        print(f'Daily Data Min Column Mean : {daily_min.mean():.2f}\n')
        print(f'Daily Data Min Column 5. Quantile Value : {daily_min_5_percentile_limit:.2f}\n')
        print(f'Daily Data Max Column Mean : {daily_max.mean():.2f}\n')
        print(f'Daily Data Max Column 95. Quantile Value : {daily_max_95_percentile_limit:.2f}\n')


        print(f'Daily Data Mean Quantile(%5-%95) Values Exceedances Perc. : {total_exceedances_mean_1_99_quantiles_daily}\n')
        
        print(f'Entire Data Mean : {whole_data_mean}\n')
        print(f'Entire Data SD : {whole_data_std}\n')
        print(f'Entire Data Min : {whole_data_min}\n')
        print(f'Entire Data Max : {whole_data_max}\n')
        
        print(f'Daily Data 1.-99.Percentile Column Mean : [{daily_1_percentile_mean_limit:.2f}-{daily_99_percentile_mean_limit:.2f}](exc:{total_exceedances_min_max_quantiles_daily:.2f})\n')
        print(f'Daily Data New CL : [{new_daily_lower_limit:.2f}-{new_daily_upper_limit:.2f}](exc:{total_exceedances_new_daily:.2f})\n')
        print(f'Daily Data Mean Quantile(%1-%99) : [{daily_1_percentile_mean_limit:.2f}-{daily_99_percentile_mean_limit:.2f}](exc:{total_exceedances_mean_1_99_quantiles_daily:.2f})\n')
        print(f'Entire Data .5-.95 Quantiles : [{whole_data_5_quantile:.2f}-{whole_data_95_quantile:.2f}](exc:{total_exceedances_5_95_quantiles_whole:.2f})\n')
        print(f'Entire Data .1-.99 Quantiles : [{whole_data_1_quantile:.2f}-{whole_data_99_quantile:.2f}](exc:{total_exceedances_1_99_quantiles_whole:.2f})\n')
        print(f'Entire Data .0025-.9975 Quantile Value : [{whole_data_025_quantile_limit:.2f}-{whole_data_975_quantile_limit:.2f}](exc:{total_exceedances_025_975_quantiles_whole:.2f})\n')

        
        if self.control_limit_type == "5th-95th Percentiles of Daily Min-Max Limits":
            lower_control_limit = daily_min_5_percentile_limit
            upper_control_limit = daily_max_95_percentile_limit
            total_exceedances = total_exceedances_min_max_quantiles_daily
        elif self.control_limit_type == "New Daily CLs":
            lower_control_limit = new_daily_lower_limit
            upper_control_limit = new_daily_upper_limit  
            total_exceedances = total_exceedances_new_daily
        elif self.control_limit_type == "Mean of Daily 1th-99th Percentiles":
            lower_control_limit = daily_1_percentile_mean_limit
            upper_control_limit = daily_99_percentile_mean_limit
            total_exceedances = total_exceedances_mean_1_99_quantiles_daily
        elif self.control_limit_type == "5th-95th Percentile Values of Entire Data":
            lower_control_limit = whole_data_5_quantile
            upper_control_limit = whole_data_95_quantile   
            total_exceedances = total_exceedances_5_95_quantiles_whole
        elif self.control_limit_type == "1st-99th Percentile Values of Entire Data":
            lower_control_limit = whole_data_1_quantile
            upper_control_limit = whole_data_99_quantile
            total_exceedances = total_exceedances_1_99_quantiles_whole
        elif self.control_limit_type == "0.25th-99.75th Percentile Values of Entire Data":
            lower_control_limit = whole_data_025_quantile_limit
            upper_control_limit = whole_data_975_quantile_limit
            total_exceedances = total_exceedances_025_975_quantiles_whole
        
            
        
        return (daily_data_list, 
                daily_min, daily_max, lower_control_limit, upper_control_limit, total_exceedances, 
                daily_min_5_percentile_limit, daily_max_95_percentile_limit, new_daily_lower_limit, new_daily_upper_limit,
                daily_1_percentile_mean_limit, daily_99_percentile_mean_limit, whole_data_025_quantile_limit, whole_data_975_quantile_limit,
                whole_data_1_quantile, whole_data_99_quantile, whole_data_5_quantile, whole_data_95_quantile,
                total_exceedances_min_max_quantiles_daily, total_exceedances_new_daily, total_exceedances_025_975_quantiles_whole, total_exceedances_mean_1_99_quantiles_daily,
                total_exceedances_1_99_quantiles_whole, total_exceedances_5_95_quantiles_whole,
                cv_lower_third, cv_middle_third, cv_upper_third)

    # Target to achieve a false day count of 10%
    def optimize_limits(self,initial_lower_limit, initial_upper_limit):
        
        def objective(limits):
            lower_limit, upper_limit = limits
            false_day_count = 0
            total_day_count = len(self.daily_statistical_data_series)

            for daily_data in self.daily_statistical_data_series:
                for value in daily_data:
                    if not (lower_limit <= value <= upper_limit):
                        false_day_count += 1
                        break  
            false_day_percentage = (false_day_count / total_day_count) * 100
            return abs(false_day_percentage - 10) 


        initial_limits = [initial_lower_limit, initial_upper_limit]
        result = minimize(objective, initial_limits, method='Nelder-Mead')
        optimal_limits = result.x

        return optimal_limits
    
    def print_new_limits(self,initial_lower_limit, initial_upper_limit):
        new_lower_limit, new_upper_limit = self.optimize_limits(initial_lower_limit, initial_upper_limit)
        print(f"New Lower Limit: {new_lower_limit:.2f}")
        print(f"New Upper Limit: {new_upper_limit:.2f}")
        return new_lower_limit, new_upper_limit


# DAYS SEPARATION - BIAS ADDITION - TRUNKING OVER THE ENTIRE DATASET - TRANSFORMATION - STATISTICAL METHOD AND COUNTING ERROR DETECTION
class PBRTQC_Biased:
    def __init__(self, whole_total_patient, data, patient_category, allocation_ratio, training_data_m_f_ratio, test_data_m_f_ratio, training_data_age_median, test_data_age_median, training_truncation_lower, training_truncation_upper, 
                 RCVg_value, selected_analyte, truncation_parameter, exclusion_parameter, transformation_parameter, 
                 statistical_parameter, block_size, bias_adding_point, patient_per_day, control_limit_type, lower_control_limit, 
                 upper_control_limit, cv_lower_third, cv_middle_third, cv_upper_third,
                 lambda_value, convert_factor, test_z_score):
        
        self.whole_total_patient = whole_total_patient
        self.data = data
        self.patient_category = patient_category
        self.allocation_ratio = allocation_ratio
        self.training_data_m_f_ratio = training_data_m_f_ratio
        self.test_data_m_f_ratio = test_data_m_f_ratio
        self.training_data_age_median = training_data_age_median
        self.test_data_age_median = test_data_age_median
        self.selected_analyte = selected_analyte
        self.flattened_data = convert_list_to_array(self.data, None)
        self.total_number_of_test_data = len(self.flattened_data)
        self.RCVg_value = RCVg_value
        self.bias_adding_point = int(bias_adding_point)
        self.patient_per_day = patient_per_day
        self.truncation_parameter = truncation_parameter
        self.training_truncation_lower_limit = training_truncation_lower
        self.training_truncation_upper_limit = training_truncation_upper
        self.exclusion_parameter = exclusion_parameter
        self.transformation_parameter = transformation_parameter
        self.statistical_parameter = statistical_parameter
        self.block_size = block_size
        self.control_limit_type = control_limit_type
        self.lower_control_limit = lower_control_limit
        self.upper_control_limit = upper_control_limit
        self.cv_lower_third = cv_lower_third
        self.cv_middle_third = cv_middle_third
        self.cv_upper_third = cv_upper_third
        self.training_lambda_value = lambda_value
        self.test_z_score = test_z_score
        self.columns = {}
        self.convert_factor = convert_factor
        self.bias_positions = []  # Rastgele bias pozisyonlarını saklamak için liste
        self.filtered_days = {bias_category: [] for bias_category in BIAS_CATEGORIES}


    def counting_saving_NPed(self):
        self.columns = {
            'Analyte': [self.selected_analyte],
            'Patient Category' : [self.patient_category],
            'Entire Data Size' : [self.whole_total_patient],
            'Test Data Size' : [],
            'Loss (%)' : [],
            'Patient Per Day': [self.patient_per_day],
            'Bias Adding Point': [self.bias_adding_point],
            
            'Allocation (Training/Test Dataset)' : [f"{self.allocation_ratio}"],
            'Convert Factor': [self.convert_factor],
            'Skewness': [],
            'Kurtosis': [],     
            'Test Data Male to Female Ratio' : [self.test_data_m_f_ratio],
            'Test Data Age Median' : [self.test_data_age_median],  
            'Robust CV IQR' : [],
            'Robust CV MAD' : [],
            'Robust CV IQR (all NPed)' : [],
            'Robust CV Median (all NPed)' : [],           
            'Transformation': [self.transformation_parameter],
            'Lambda Value' : [self.training_lambda_value],
            'Truncation': [f'{self.truncation_parameter} ({self.exclusion_parameter})'],
            'Truncation Lower Limit': [self.training_truncation_lower_limit],
            'Truncation Upper Limit': [self.training_truncation_upper_limit],
            'Control Limit Type': [f"{self.control_limit_type}"],
            'Control Lower Limit': [self.lower_control_limit],
            'Control Upper Limit': [self.upper_control_limit],
            'Exclusion': [self.exclusion_parameter],
            'Statistical Method': [self.statistical_parameter],
            'Block Size': [self.block_size],
            'Sum of MNPed' : [],      
            'AUC (x Bias)' : [],  # ALTERNATIVE PERFORMANCE METHOD. YOU CAN TRY IT !
            'FPR (Bias = 0)': [],
            'AUC' : [],        

        }
        
        MNPed_list = []
        MNPed_list_with_bias_coeff = []
        total_daily_indices = [] 
        NPed_bias = []
        NPed_multiplied_bias = []
        bias_values_for_trapz = []
        all_missed_values = []
        mnped_fpr_included_count = []
        bias_day = 0
        
        self.bias_positions = {bias_category: [] for bias_category in BIAS_CATEGORIES}
        self.filtered_days = {bias_category: [] for bias_category in BIAS_CATEGORIES}
        for bias_category in BIAS_CATEGORIES:
            total_counted_data = []
            bias_day += 1
            self.biased_column_name = f'Biased Results ({bias_category}%)'
            self.biased_median_column_Daily = f'Median({self.biased_column_name})'
            self.biased_min_column_Daily = f'Min({self.biased_column_name})'
            self.biased_max_column_Daily = f'Max({self.biased_column_name})'
            self.biased_accuracy = f'Median Accuracy({self.biased_column_name})'

                 
            self.columns[self.biased_median_column_Daily] = []
            self.columns[self.biased_min_column_Daily] = []
            self.columns[self.biased_max_column_Daily] = []
            self.columns[self.biased_accuracy] = []
   
            
            neat_data_series = self.data.copy()
            original_shape = [len(lst) for lst in neat_data_series]
            
            biased_series = add_bias(neat_data_series, bias_adding_point=self.bias_adding_point, factor=int(bias_category), random_mode=False)
            
            
            self.flattened_biased_data = convert_list_to_array(biased_series, None)
            
            truncation_of_second_half = Truncation(self.flattened_biased_data, self.truncation_parameter,  self.exclusion_parameter,  self.RCVg_value)
            self.truncated_test_data = truncation_of_second_half.truncate_data(self.training_truncation_lower_limit,  self.training_truncation_upper_limit, 
                                                                                 self.block_size, self.test_z_score)
                      
            
            transformation_of_second_half = Transformation(self.selected_analyte, self.truncated_test_data , self.transformation_parameter, self.convert_factor, self.training_lambda_value)
            transformed_data, _ = transformation_of_second_half.transform_data()    
            
                
                
            self.biased_transformed_daily_series = convert_array_to_list(transformed_data, original_shape, 'numpy_list')
            
            statistical_process_for_control_limit = StatisticalProcess(self.biased_transformed_daily_series, self.statistical_parameter, 
                                                   self.block_size, self.test_z_score, keep_nan_index=True)
            statistical_data_series_with_nan = statistical_process_for_control_limit.statistical_process_data()
         
            transformed_series_test_data_without_nan, start_indices= remove_nan_from_lists(statistical_data_series_with_nan,None,self.bias_adding_point)            
            
            
            daily_count_in_a_bias = []

            false_positives_daily_list = []
            true_negatives_daily_list = []
            false_negatives_daily_list = []
            true_positives_daily_list = []
            arranged_daily_count_in_a_bias = []
            missed_values_in_a_bias = []
            count_values_in_a_bias = []
            ratio_NPed = []
            nped_count_ratio = []
            daily_count = 0
 
            day_number = 0
                
            for i, daily_data_list in enumerate(transformed_series_test_data_without_nan):
                    day_number += 1
                    false_positives = 0
                    true_negatives = 0
                    true_positives = 0 
                    false_negatives = 0
                    start_index = start_indices[i]
                    total_daily_data = len(daily_data_list)
          
                    if bias_category != 0:
                        for m in range(0,start_index):
                            if not (self.lower_control_limit <= daily_data_list[m] <= self.upper_control_limit):
                                false_positives += 1  
                            else:
                                true_negatives += 1
                        false_positives_daily_list.append((false_positives + (self.bias_adding_point - start_index))/ self.bias_adding_point)  
                        true_negatives_daily_list.append(true_negatives/ self.bias_adding_point)
                        
                        for n in range(start_index, total_daily_data):
                            if not (self.lower_control_limit <= daily_data_list[n] <= self.upper_control_limit):
                                true_positives += 1  
                            else:
                                false_negatives += 1
                        
                        false_negatives_daily_list.append((false_negatives + abs((self.patient_per_day - self.bias_adding_point) - (total_daily_data - start_index))) / (self.patient_per_day-self.bias_adding_point))  
                        true_positives_daily_list.append(true_positives / (self.patient_per_day-self.bias_adding_point))
                                
                        if total_daily_data > start_index:
                            for j in range(start_index, total_daily_data):
                                if not (self.lower_control_limit <= daily_data_list[j] <= self.upper_control_limit):
                                    daily_count = j - start_index

                                    break
                            else:
                                daily_count = total_daily_data - start_index
                   
                            
                            
                            daily_count_in_a_bias.append(daily_count)         
                            total_daily_indices.append(daily_count)
                            ratio_NPed.append(daily_count/(total_daily_data - start_index))
                            count_values_in_a_bias.append(total_daily_data)
                            dif_length_day_start_index = total_daily_data - start_index
                            if dif_length_day_start_index == 0:
                               dif_length_day_start_index = 1 
                            nped_count_ratio.append(daily_count/(total_daily_data-dif_length_day_start_index))
                    missed_values_in_a_bias.append(self.patient_per_day-total_daily_data)            
                    arranged_daily_count_in_a_bias.append(((self.patient_per_day-self.bias_adding_point)*daily_count)/(total_daily_data-start_index))
                
            fpr_count = 0
            false_day_count = 0
            
            if bias_category == 0:
                
                
                fpr_data = convert_list_to_array(transformed_series_test_data_without_nan, None)
               
                total_fpr_data_points = len(fpr_data)
                self.total_data_size  = total_fpr_data_points
     
                median_second_half = np.median(fpr_data)
                iqr_second_half = np.percentile(fpr_data, 75) - np.percentile(fpr_data, 25)
                mad_test_MNPed = np.median(np.abs(fpr_data - median_second_half))
               
                if median_second_half != 0:
                    self.RCVQ_test_data = iqr_second_half / median_second_half
                    self.RCVM_test_data = 1.4826 * mad_test_MNPed / median_second_half
                else:
                    self.RCVQ_test_data = np.nan
                    self.RCVM_test_data = np.nan
                
                print(self.RCVM_test_data)
                self.columns['Robust CV IQR'] = self.RCVQ_test_data
                self.columns['Robust CV MAD'] = self.RCVM_test_data
                for k in fpr_data:
                    if not (self.lower_control_limit <= k <= self.upper_control_limit):
                        fpr_count += 1
    
                for x_daily_data in transformed_series_test_data_without_nan:
                    max_value = max(x_daily_data)
                    min_value = min(x_daily_data)

                    if max_value > self.upper_control_limit:
                        false_day_count += 1  

                    if min_value < self.lower_control_limit:
                        false_day_count += 1 

                if total_fpr_data_points > 0:
                    fpr_percentage = (fpr_count / total_fpr_data_points) * 100
                    self.columns['%FPR'] = fpr_percentage
                    print(f"FPR Percentage in All Data (Bias 0)-total_fpr_data_points > 0-: {fpr_percentage:.2f}")
                else:
                    fpr_percentage = 0
                    self.columns['%FPR'] = fpr_percentage
                    print(f"FPR Percentage in All Data (Bias 0)--total_fpr_data_points <= 0-: {fpr_percentage:.2f}%")
      
            
            sum_of_missing_value_in_a_bias = sum(missed_values_in_a_bias)
            total_counted_data.append(sum(count_values_in_a_bias))
         

            if daily_count_in_a_bias:
                daily_bias_median = np.median(daily_count_in_a_bias)
                daily_bias_min = np.min(daily_count_in_a_bias)
                daily_bias_max = np.max(daily_count_in_a_bias)


                MNPed_list.append(daily_bias_median)
                MNPed_list_with_bias_coeff.append(daily_bias_median * abs(int(bias_category)))

                self.columns[self.biased_median_column_Daily].append(daily_bias_median)
                self.columns[self.biased_min_column_Daily].append(daily_bias_min)
                self.columns[self.biased_max_column_Daily].append(daily_bias_max)
                
                NPed_bias.append(daily_bias_median)
                NPed_multiplied_bias.append(daily_bias_median * abs(int(bias_category)) if not np.isnan(daily_bias_median) else np.nan)
                bias_values_for_trapz.append(int(bias_category))
                mnped_fpr_included_count.append(np.median(nped_count_ratio))     
                

 
        area_NPed_multiplied = np.trapz([v for v in NPed_multiplied_bias if not np.isnan(v)],
                                            x=[b for i, b in enumerate(bias_values_for_trapz) if not np.isnan(NPed_multiplied_bias[i])])
        
        print(f"\nAUC(*Bias): {area_NPed_multiplied}")
        area_NPed = np.trapz([v for v in NPed_bias if not np.isnan(v)],
                                            x=[b for i, b in enumerate(bias_values_for_trapz) if not np.isnan(NPed_multiplied_bias[i])])
        
        print(f"\nAUC(*Bias): {area_NPed}")

        sum_of_MNPed = sum(MNPed_list)
        self.columns['FPR (Bias = 0)'] = fpr_percentage
        self.columns['Sum of MNPed'] = sum_of_MNPed
        self.columns['AUC'] = area_NPed
        self.columns['AUC (x Bias)'] = area_NPed_multiplied
        self.columns['Test Data Size'] = self.total_data_size
        self.columns['Loss (%)'] = round(((self.whole_total_patient - self.total_data_size)/self.whole_total_patient)*100)
        self.columns['Skewness'] = self.skewnessData
        self.columns['Kurtosis'] = self.kurtosisData
        all_missed_values.append(sum_of_missing_value_in_a_bias) 
        
        filtered_indices = [x for x in total_daily_indices if not np.isnan(x)]

        total_daily_indices_array = np.array(filtered_indices)
        non_nan_array = total_daily_indices_array[~np.isnan(total_daily_indices_array)]
        medyan_MNPed = np.median(non_nan_array)
        mad_MNPed = np.median(np.abs(non_nan_array - medyan_MNPed))
        q1_MNPed = np.percentile(non_nan_array, 25) 
        q3_MNPed = np.percentile(non_nan_array, 75) 
        iqr_MNPed = q3_MNPed - q1_MNPed
        if medyan_MNPed != 0:
            instability_metric = iqr_MNPed / medyan_MNPed
            rcvm = 1.4826 * mad_MNPed / medyan_MNPed
        else:
            instability_metric = np.nan
            rcvm = np.nan
            
        self.columns['Robust CV IQR (all NPed)'] = instability_metric
        self.columns['Robust CV Median (all NPed)'] = rcvm
        
    
        for key, value in self.columns.items():
            if not isinstance(value, list):
                self.columns[key] = [value]

        max_length = max(len(col) for col in self.columns.values())
        for key, col in self.columns.items():
            if len(col) < max_length:
                col.extend([None] * (max_length - len(col)))
                  
        data_for_plot = pd.DataFrame(self.columns)
                     
        return  fpr_percentage, sum_of_MNPed , area_NPed, area_NPed_multiplied, data_for_plot

    def counting_saving_NPed_manuel(self,selected_analyte):   
            
            performance_folder_name = f'Manuel Performance Process/{selected_analyte}'
            if not os.path.exists(performance_folder_name):
                os.makedirs(performance_folder_name)
            performance_file_name = f'{performance_folder_name}/Performance of {selected_analyte}.xlsx'
                                   
            neat_data_series = self.data.copy()

            flattened_data = convert_list_to_array(neat_data_series, None)
            
            truncation_of_second_half = Truncation(flattened_data, self.truncation_parameter,  self.exclusion_parameter,  self.RCVg_value)
            truncated_test_data = truncation_of_second_half.truncate_data(self.training_truncation_lower_limit,  self.training_truncation_upper_limit, 
                                                                                 self.block_size, self.test_z_score)
                  
            
            transformation_of_second_half = Transformation(self.selected_analyte, truncated_test_data , self.transformation_parameter, self.convert_factor, self.training_lambda_value)
            transformed_data, _ = transformation_of_second_half.transform_data()
            
            non_nan_data = transformed_data[~np.isnan(transformed_data)]
            # Skewness ve Kurtosis hesaplayın
            self.skewnessData = stats.skew(non_nan_data)
            self.kurtosisData = stats.kurtosis(non_nan_data)
            fpr_percentage, sum_of_MNPed , area_NPed, area_NPed_multiplied, data_for_plot_manuel = self.counting_saving_NPed()

            if os.path.exists(performance_file_name):
                existing_data = pd.read_excel(performance_file_name)
                combined_data = pd.concat([existing_data, data_for_plot_manuel], ignore_index=True)
            else:
                combined_data = data_for_plot_manuel         
            
            
            combined_data.to_excel(performance_file_name, index=False)
            print(f'Saved File : {performance_file_name}')
            return fpr_percentage, sum_of_MNPed , area_NPed, area_NPed_multiplied, data_for_plot_manuel
        
    def counting_saving_NPed_automatic(self):   

            neat_data_series = self.data.copy()

            flattened_data = convert_list_to_array(neat_data_series, None)
            
            truncation_of_second_half = Truncation(flattened_data, self.truncation_parameter,  self.exclusion_parameter,  self.RCVg_value)
            truncated_test_data = truncation_of_second_half.truncate_data(self.training_truncation_lower_limit,  self.training_truncation_upper_limit, 
                                                                                 self.block_size, self.test_z_score)
            
            
            
            transformation_of_second_half = Transformation(self.selected_analyte, truncated_test_data , self.transformation_parameter, self.convert_factor, self.training_lambda_value)
            transformed_data, _ = transformation_of_second_half.transform_data()
            
            non_nan_data = transformed_data[~np.isnan(transformed_data)]

            self.skewnessData = stats.skew(non_nan_data)
            self.kurtosisData = stats.kurtosis(non_nan_data)
            
            
            if (-0.5 < self.skewnessData < 0.5) and (-0.5 < self.kurtosisData < 0.5):
                
                boolValue = 1
                fpr_percentage, sum_of_MNPed , area_NPed, area_NPed_multiplied, data_for_plot_manuel = self.counting_saving_NPed()                 
            else:
                boolValue = 0
                fpr_percentage = 0
                sum_of_MNPed = 0
                area_NPed = 0
                area_NPed_multiplied = 0
                data_for_plot_manuel = 0
                
            return boolValue, fpr_percentage, sum_of_MNPed , area_NPed, area_NPed_multiplied, data_for_plot_manuel
        
 
class Plot_Training_Data(FigureCanvas):
    
    def __init__(self, parent=None, width=8, height=15, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        gs = gridspec.GridSpec(3, 2, figure=self.fig)

        self.ax_training_boxplot = self.fig.add_subplot(gs[0, 0])
        self.ax_test_boxplot = self.fig.add_subplot(gs[0, 1])
        self.ax_training_hist = self.fig.add_subplot(gs[1, :])
        self.ax_test_hist = self.fig.add_subplot(gs[2, :])

        super(Plot_Training_Data, self).__init__(self.fig)

    def training_and_test_data_plot(self, analyte, training_data_with_nan, test_data_with_nan, transformation_parameter, 
                                    truncation_parameter, exclusion_parameter, lambda_value_training, mad_training_data, mad_test_data, ):
  
        transformed_training_list_without_nan = remove_nan_from_lists(training_data_with_nan)
        transformed_test_list_without_nan = remove_nan_from_lists(test_data_with_nan)
         
        training_data = convert_list_to_array(transformed_training_list_without_nan, "Plot-Transformed Training Data Array without NaN")
        test_data = convert_list_to_array(transformed_test_list_without_nan, "Plot-Transformed Test Data Array without NaN") 
        
        Q1_training_data = np.percentile(training_data, 25)
        Q3_training_data = np.percentile(training_data, 75)
        IQR_training_data = Q3_training_data - Q1_training_data
        lower_bound_training_data = Q1_training_data - 1.5 * IQR_training_data
        upper_bound_training_data = Q3_training_data + 1.5 * IQR_training_data
        outliers_training_data = np.sum((training_data < lower_bound_training_data) | (training_data > upper_bound_training_data))
        number_of_training_data = len(training_data)

 
    
        median_training_data = np.median(training_data)
        if median_training_data != 0:
            rns_training_data = IQR_training_data / median_training_data
        else:
            rns_training_data = np.nan

        
        self.ax_training_boxplot.clear()
        self.ax_training_boxplot.boxplot(training_data)
        y_min_training_data, y_max_training_data = self.ax_training_boxplot.get_ylim()
        self.ax_training_boxplot.set_title(f'{analyte} of Training Data\n[Number of Data({number_of_training_data}), Number of Outliers({outliers_training_data})]', fontsize=6, weight='bold')
        self.ax_training_boxplot.annotate(f'Median: {median_training_data:.2f}\nRNS: {rns_training_data:.2f}\nIQR: {IQR_training_data:.2f}\nMean: {np.mean(training_data):.2f}\nRobust CV MAD: {mad_training_data:.2f}',
                                          xy=(0.95, 0.95), xycoords='axes fraction',
                                          horizontalalignment='right', verticalalignment='top',
                                          color='black', fontsize=6)


        # Test verileri için istatistiksel değerler ve boxplot
        Q1_test_data = np.percentile(test_data, 25)
        Q3_test_data = np.percentile(test_data, 75)
        IQR_test_data = Q3_test_data - Q1_test_data
        lower_bound_test_data = Q1_test_data - 1.5 * IQR_test_data
        upper_bound_test_data = Q3_test_data + 1.5 * IQR_test_data
        outliers_test_data = np.sum((test_data < lower_bound_test_data) | (test_data > upper_bound_test_data))
        number_of_test_data = len(test_data)


        median_test_data = np.median(test_data)
        if median_test_data != 0:
            rns_test_data = IQR_test_data / median_test_data
        else:
            rns_test_data = np.nan
            
        self.ax_test_boxplot.clear()
        self.ax_test_boxplot.boxplot(test_data)
        y_min_test_data, y_max_test_data = self.ax_test_boxplot.get_ylim()
        self.ax_test_boxplot.set_title(f'{analyte} of Test Data\n[Number of Data({number_of_test_data}), Number of Outliers({outliers_test_data})]', fontsize=6, weight='bold')
        self.ax_test_boxplot.annotate(f'Median: {np.median(test_data):.2f}\nRNS: {rns_test_data:.2f}\nIQR: {IQR_test_data:.2f}\nMean: {np.mean(test_data):.2f}\nRobust CV MAD: {mad_test_data:.2f}',
                                      xy=(0.95, 0.9), xycoords='axes fraction',
                                      horizontalalignment='right', verticalalignment='top',
                                      color='black', fontsize=6)

        

        # Eğitim verileri için histogram
        self.ax_training_hist.clear()
        self.ax_training_hist.hist(training_data, bins=50, color='blue', alpha=0.7, density=True)
        self.ax_training_hist.set_title(f'Distribution Plot of Training Data ({transformation_parameter}, {truncation_parameter}, {exclusion_parameter})', fontsize=8, weight='bold')
        self.ax_training_hist.tick_params(axis='x', labelsize=8)
        self.ax_training_hist.tick_params(axis='y', labelsize=8)
        skewness = stats.skew(training_data)
        kurtosis = stats.kurtosis(training_data)
        self.ax_training_hist.annotate(f'S: {skewness:.2f}\nK: {kurtosis:.1f}',
                                       xy=(0.95, 0.95), xycoords='axes fraction',
                                       horizontalalignment='right', verticalalignment='top',
                                       color='black', fontsize=12, fontweight='bold')

        mean_training = np.mean(training_data)
        std_training = np.std(training_data)
        xmin, xmax = self.ax_training_hist.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_training, std_training)
        self.ax_training_hist.plot(x, p, 'r--', linewidth=2)

        kde_training = gaussian_kde(training_data)
        self.ax_training_hist.plot(x, kde_training(x), 'black', linewidth=2)

        # Test verileri için histogram
        self.ax_test_hist.clear()
        self.ax_test_hist.hist(test_data, bins=50, color='blue', alpha=0.7, density=True)
        self.ax_test_hist.set_title(f'Distribution Plot of Test Data({transformation_parameter}, {truncation_parameter}, {exclusion_parameter})', fontsize=8, weight='bold')
        self.ax_test_hist.tick_params(axis='x', labelsize=8)
        self.ax_test_hist.tick_params(axis='y', labelsize=8)
        skewness = stats.skew(test_data)
        kurtosis = stats.kurtosis(test_data)
        self.ax_test_hist.annotate(f'S: {skewness:.2f}\nK: {kurtosis:.1f}',
                                   xy=(0.95, 0.95), xycoords='axes fraction',
                                   horizontalalignment='right', verticalalignment='top',
                                   color='black', fontsize=12, fontweight='bold')
        
        
        if transformation_parameter == "Yeo-Johnson":
            self.ax_training_hist.annotate(f'Yeo-Johnson Lambda: {lambda_value_training:.2f}',
                                              xy=(1, 0.5), xycoords='axes fraction',
                                              horizontalalignment='right', verticalalignment='bottom',
                                              color='red', fontsize=8)

        mean_test = np.mean(test_data)
        std_test = np.std(test_data)
        xmin, xmax = self.ax_test_hist.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_test, std_test)
        self.ax_test_hist.plot(x, p, 'r--', linewidth=2)

        kde_test = gaussian_kde(test_data)
        self.ax_test_hist.plot(x, kde_test(x), 'black', linewidth=2)

        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.95)
        self.draw() 
        
class Plot_Control_Limit(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.ax_control_limit = self.fig.add_subplot()
        super(Plot_Control_Limit, self).__init__(self.fig)

    def control_limit_plot(self, daily_statistical_data_series, daily_data_parameters_list, 
                           analyte, training_lower_trun_limit, training_upper_trun_limit,
                           daily_min,  daily_max, daily_min_5_percentile_limit, daily_max_95_percentile_limit, new_daily_lower_limit, new_daily_upper_limit,
                           daily_1_percentile_mean_limit, daily_99_percentile_mean_limit, whole_data_025_quantile_limit, whole_data_975_quantile_limit,
                           whole_data_1_quantile_limit, whole_data_99_quantile_limit, whole_data_5_quantile_limit, whole_data_95_quantile_limit,
                           total_exceedances_min_max_quantiles_daily, total_exceedances_new_daily, total_exceedances_025_975_quantiles_whole, total_exceedances_mean_1_99_quantiles_daily, 
                           total_exceedances_1_99_quantiles_whole, total_exceedances_5_95_quantiles_whole,
                           transformation_parameter, truncation_parameter, exclusion_parameter, statistical_param, block_size, 
                           patient_per_day, lower_trun_limit, upper_trun_limit, control_limit_type):
        
        daily_statistical_data_series_without_nan = remove_nan_from_lists(daily_statistical_data_series)

        new_total_days = len(daily_statistical_data_series_without_nan)             
        self.ax_control_limit.clear()  

        x_values = np.arange(new_total_days)
                          
        try:
            for i, daily_data in enumerate(daily_statistical_data_series_without_nan):
   
                self.ax_control_limit.scatter(i, daily_data_parameters_list[i]['max'], marker='x', color='black', s=1)
                self.ax_control_limit.scatter(i, daily_data_parameters_list[i]['min'], marker='x', color='black', s=1)
                self.ax_control_limit.errorbar(
                    i, daily_data_parameters_list[i]['mean'],
                    yerr=[[daily_data_parameters_list[i]['mean'] - daily_data_parameters_list[i]['5_percentile']], 
                          [daily_data_parameters_list[i]['95_percentile'] - daily_data_parameters_list[i]['mean']]],
                    fmt='o', color='black', ecolor='lightgray', elinewidth=1, capsize=1, markersize=3
                )

            line_properties = {'linewidth': 3}  

            self.ax_control_limit.hlines([daily_min_5_percentile_limit, daily_max_95_percentile_limit], 
                             xmin=0, xmax=new_total_days-1, colors='blue', linestyles='dashed',
                             label=f"5th-95th Percentiles of Daily Min-Max Limits(Exceedances % of Total Days: {total_exceedances_min_max_quantiles_daily:.2f})",
                             **(line_properties if control_limit_type == "5th-95th Percentiles of Daily Min-Max Limits" else {}))

            self.ax_control_limit.hlines([daily_1_percentile_mean_limit, daily_99_percentile_mean_limit], 
                             xmin=0, xmax=new_total_days-1, colors='cyan', linestyles='dashed',
                             label=f"Mean of Daily 1th-99th Percentiles(Exceedances % of Total Days: {total_exceedances_mean_1_99_quantiles_daily:.2f})",
                             **(line_properties if control_limit_type == "Mean of Daily 1th-99th Percentiles" else {}))

            self.ax_control_limit.hlines([new_daily_lower_limit, new_daily_upper_limit], 
                             xmin=0, xmax=new_total_days-1, colors='black', linestyles='dashed',
                             label=f"New Daily Data Limits (Exceedances % of Total Days: {total_exceedances_new_daily:.2f})",
                             **(line_properties if control_limit_type == "New Daily CLs" else {}))

            self.ax_control_limit.hlines([whole_data_025_quantile_limit, whole_data_975_quantile_limit], 
                             xmin=0, xmax=new_total_days-1, colors='red', linestyles='dashed',
                             label=f"0.25th-99.75th Percentile Values of Entire Data(Exceedances % of Entire Data: {total_exceedances_025_975_quantiles_whole:.2f})",
                             **(line_properties if control_limit_type == "0.25th-99.75th Percentile Values of Entire Data" else {}))

            self.ax_control_limit.hlines([whole_data_1_quantile_limit, whole_data_99_quantile_limit], 
                             xmin=0, xmax=new_total_days-1, colors='orange', linestyles='dashed',
                             label=f"1st-99th Percentile Values of Entire Data(Exceedances % of Entire Data: {total_exceedances_1_99_quantiles_whole:.2f})",
                             **(line_properties if control_limit_type == "1st-99th Percentile Values of Entire Data" else {}))

            self.ax_control_limit.hlines([whole_data_5_quantile_limit, whole_data_95_quantile_limit], 
                             xmin=0, xmax=new_total_days-1, colors='yellow', linestyles='dashed',
                             label=f"5th-95th Percentile Values of Entire Data(Exceedances % of Entire Data: {total_exceedances_5_95_quantiles_whole:.2f})",
                             **(line_properties if control_limit_type == "5th-95th Percentile Values of Entire Data" else {}))


            if control_limit_type == "5th-95th Percentiles of Daily Min-Max Limits":
                lower_control_limit = daily_min_5_percentile_limit
                upper_control_limit = daily_max_95_percentile_limit
            elif control_limit_type == "New Daily CLs":
                lower_control_limit = new_daily_lower_limit
                upper_control_limit = new_daily_upper_limit    
            elif control_limit_type == "Mean of Daily 1th-99th Percentiles":
                lower_control_limit = daily_1_percentile_mean_limit
                upper_control_limit = daily_99_percentile_mean_limit
            elif control_limit_type == "5th-95th Percentile Values of Entire Data":
                lower_control_limit = whole_data_5_quantile_limit
                upper_control_limit = whole_data_95_quantile_limit            
            elif control_limit_type == "1st-99th Percentile Values of Entire Data":
                lower_control_limit = whole_data_1_quantile_limit
                upper_control_limit = whole_data_99_quantile_limit
            elif control_limit_type == "0.25th-99.75th Percentile Values of Entire Data":
                lower_control_limit = whole_data_025_quantile_limit
                upper_control_limit = whole_data_975_quantile_limit        
        except Exception as e:
            logging.error(f"An error occurred: {e}")

        self.ax_control_limit.set_xlim(x_values.min(), new_total_days - 1)

        self.ax_control_limit.set_title(f'Control Limits of {analyte}\n({transformation_parameter},{truncation_parameter}, {exclusion_parameter}, {statistical_param})\nTL[{lower_trun_limit:.2f}-{upper_trun_limit:.2f}], CL[{lower_control_limit:.2f}-{upper_control_limit:.2f}], BS[{block_size}], TD[{new_total_days}], PPD[{patient_per_day}]', fontsize=8, weight="bold")
        self.ax_control_limit.set_xlabel('Day Number', fontsize=8)
        self.ax_control_limit.set_ylabel('Results', fontsize=8)
        legend = self.ax_control_limit.legend(fontsize=6)
        for text in legend.get_texts():
            if control_limit_type in text.get_text():
                text.set_fontweight('bold')

        self.draw()
        
class PlotPatientErrorDetection(FigureCanvas):
    def __init__(self, parent=None, width=12, height=8, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.ax_MNPed = self.fig.add_subplot()
        self.fig.subplots_adjust(wspace=0.5)
        super(PlotPatientErrorDetection, self).__init__(self.fig)

    def clear_axes(self):
        self.ax_MNPed.clear()


    def draw_MNPed_Plot(self, data_for_plot, false_positive_rate, sum_of_MNPed, 
                        area_NPed, area_NPed_multiplied, selected_analyte, control_limit_type, truncation_parameter, 
                        training_lower_trun_limit, training_upper_trun_limit, transformation_parameters, 
                        statistical_parameters, block_size, lower_control_limit, upper_control_limit, 
                        rns_test_data, patient_per_day, biased_point_widget):
        
        self.clear_axes()  


        NPed_values, min_values, max_values, bias_values = [], [], [], []

        for bias_category in tqdm(BIAS_CATEGORIES, desc="Bias processing..."):
            biased_median_column = f'Median(Biased Results ({bias_category}%))'
            biased_min_column = f'Min(Biased Results ({bias_category}%))'
            biased_max_column = f'Max(Biased Results ({bias_category}%))'

            NPed_data = data_for_plot[biased_median_column].tolist()
            min_data = data_for_plot[biased_min_column].tolist()
            max_data = data_for_plot[biased_max_column].tolist()

            filtered_NPed_data = []
            filtered_min_data = []
            filtered_max_data = []
            filtered_bias_values = []

            for i in range(len(NPed_data)):
                value = NPed_data[i]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    filtered_NPed_data.append(value)
                    filtered_min_data.append(min_data[i])
                    filtered_max_data.append(max_data[i])
                    filtered_bias_values.append(bias_category)

            NPed_values.append(filtered_NPed_data)
            min_values.append(filtered_min_data)
            max_values.append(filtered_max_data)
            bias_values.append(filtered_bias_values)

        flattened_bias_values = [item for sublist in bias_values for item in sublist]
        flattened_NPed_values = [item for sublist in NPed_values for item in sublist]
        flattened_min_values = [item for sublist in min_values for item in sublist]
        flattened_max_values = [item for sublist in max_values for item in sublist]

        errorbar_handle = None
        for i, bias in enumerate(flattened_bias_values):
            errorbar_handle = self.ax_MNPed.errorbar(
                bias, flattened_NPed_values[i],
                yerr=[[flattened_NPed_values[i] - flattened_min_values[i]],
                      [flattened_max_values[i] - flattened_NPed_values[i]]],
                fmt='o', capsize=5, color='black', alpha=0.7
            )

        self.ax_MNPed.axvline(x=0, color='black', linestyle='--')
        self.ax_MNPed.set_title(f'{selected_analyte}\nTL: {truncation_parameter}[{training_lower_trun_limit:.2f}-{training_upper_trun_limit:.2f}], {transformation_parameters}\n{statistical_parameters}({block_size}), CL: {control_limit_type}[{lower_control_limit:.2f}-{upper_control_limit:.2f}], CVMAD of Test Data({rns_test_data:.2f})',
                  fontsize=12, fontweight='bold')
        self.ax_MNPed.set_xlabel('Systematic Error (Bias)', fontsize=12, fontweight='bold')
        self.ax_MNPed.set_ylabel("MNPed", fontsize=12, fontweight='bold')

        if errorbar_handle is not None:
            self.ax_MNPed.legend([errorbar_handle], ['Median Values with Min/Max'], loc='lower right')

        self.ax_MNPed.figure.tight_layout()
        self.draw()


# IMPLEMENT ALL THE ALGORITHMS IN THE APPLICATION AT THE FASTEST SPEED. CAUTION: THIS MAY CONSUME THE PERFORMANCE OF ALL CPU PROCESSORS, WHICH MAY SLOW DOWN YOUR USE OF OTHER APPLICATIONS. 
class automatic_PBRTQC:
    def __init__(self,  patient_category, selected_analyte, date_column_name, process_column_name, 
                 selected_limit_RCVg, selected_limit_TEa, allowable_bias_array, selected_limit_BVi, selected_limit_BVg, 
                 selected_limit_CVa, selected_limit_bias, measurement_lower_limit, measurement_upper_limit,
                 patient_per_day, allocation_ratio, biased_adding_point, convert_factor, control_limit_type ):
                                            
        self.selected_analyte = selected_analyte
        self.patient_category = patient_category
        self.Analytes_List = []
        self.date_column_name = date_column_name
        self.process_column_name = process_column_name
        self.selected_limit_RCVg = selected_limit_RCVg 
        self.selected_limit_TEa = selected_limit_TEa
        self.allowable_bias_array = allowable_bias_array
        self.selected_limit_BVi = selected_limit_BVi
        self.selected_limit_BVg = selected_limit_BVg 
        self.selected_limit_CVa = selected_limit_CVa 
        self.selected_limit_bias = selected_limit_bias
        self.measurement_lower_limit = measurement_lower_limit
        self.measurement_upper_limit = measurement_upper_limit  
        self.control_limit_type = control_limit_type
        self.patient_per_day = patient_per_day
        self.allocation_ratio = allocation_ratio
        self.biased_adding_point = biased_adding_point
        self.convert_factor = convert_factor
        self.plot_for_data_folder_name = f'Performances/Supplementary 1/{self.selected_analyte}_{self.patient_per_day}_{self.biased_adding_point}_{self.allocation_ratio}'
        if not os.path.exists(self.plot_for_data_folder_name):
            os.makedirs(self.plot_for_data_folder_name)

                       
    def starting_automatic_process(self):
        warnings.simplefilter("ignore")
        self.select_csv()
        print("-------------")
        print("Analyte Info :\n")
        print(f'{self.selected_analyte}\n')
        print(f'RCVg = {self.selected_limit_RCVg}\n')
        print(f'TEa = {self.selected_limit_TEa}\n')
        print(f'BVi = {self.selected_limit_BVi}\n')
        print(f'BVg = {self.selected_limit_BVg}\n')      
        print(f'CV = {self.selected_limit_CVa}\n') 
        print(f'Bias = {self.selected_limit_bias}\n') 
        print(f'Measurable Lower Limit = {self.measurement_lower_limit}\n')   
        print(f'Measurable Upper Limit = {self.measurement_upper_limit}\n') 
        print("-------------")        
        print("Total Business Days :")
        print(self.total_days)
        print("-------------")
        print("Total Number of Training Data :")
        print(self.total_number_of_training_data)
        print("Total Number of Test Data :")
        print(self.total_number_of_test_data)
        print("-------------")
        print("Patient Per Day :")
        print(self.patient_per_day)
        print("-------------")
        print("Bias Adding Point :")
        print(self.biased_adding_point)
        print("-------------")       
        print("Allocation Ratio :")
        print(f"{self.allocation_ratio}")
        print("-------------")   
        
        print(self.first_half_series)
        print(self.second_half_series)                      
        self.calculation_control_limits()       
            
    def select_csv(self):
                warnings.simplefilter(action='ignore', category=FutureWarning)
        
            
                self.df, self.total_patients, _, _, self.training_data_m_f_ratio, self.test_data_m_f_ratio, self.training_data_age_median, self.test_data_age_median, = process_data_file( self.patient_category,self.allocation_ratio, self.measurement_lower_limit,self.measurement_upper_limit,
                    gender_column_name, age_column_name, sampling_time_column_name, clinics_column_name,
                    icu_name_list, inpatient_name_list, emergency_name_list, "Results",  self.selected_analyte)
                

                self.total_patients = len(self.df)
                self.total_days = self.total_patients // self.patient_per_day
        
                
                self.list_day_and_data()
    
    def list_day_and_data(self):
        
        self.first_half_series, self.second_half_series = df_to_list(self.df, self.allocation_ratio, 1 - self.allocation_ratio, self.patient_per_day, return_type='pandas', exclude_last_list=True)
        self.total_number_of_training_data = len(self.first_half_series)  
        self.total_number_of_test_data = len(self.second_half_series)
        self.first_half_series_copy = self.first_half_series.copy()
        self.second_half_series_copy = self.second_half_series.copy()
        
        
        self.original_shape_training_data = [len(lst) for lst in self.first_half_series_copy]
        self.original_shape_test_data = [len(lst) for lst in self.second_half_series_copy]
        
        self.flattened_training_data = convert_list_to_array(self.first_half_series_copy)
        self.flattened_test_data = convert_list_to_array(self.second_half_series_copy)

            
    def calculation_control_limits(self):

        BLOCK_SIZE = ["20", "30", "50", "75", "100"]
        TRUNCATION_CATEGORIES = ["No Truncation", "RCVg", "1%-99%", "3*SD"] # , ,"RCVg", "1%-99%"
        
        try:
            combinations = []
            
            for block_size in BLOCK_SIZE:
                for truncation_parameter in TRUNCATION_CATEGORIES:
                    if truncation_parameter == "No Truncation":
                            transformation_categories = ["No Transformation", "Yeo-Johnson", "Square Root",  "Log10"] #"Standard Scale", "Min-Max Scale", "Robust Scale", 
                            STATISTICAL_CATEGORIES = ["EWMA-Decay factor", "Moving Mean", "Moving Median", "Moving Std"]#"Moving MAD", 
                    else:
                            STATISTICAL_CATEGORIES = ["EWMA-Decay factor", "Moving Mean", "Moving Median", "Moving Std"]#
                            transformation_categories = ["No Transformation", "Yeo-Johnson", "Square Root", "Log10"]

                    for exclusion_parameter in (["No Exclusion"] if truncation_parameter == "No Truncation" else ["Trimming", "Winsorization"]):
                        for transformation_parameter in transformation_categories:
                            for statistical_parameter in STATISTICAL_CATEGORIES:
                                    combinations.append((block_size, truncation_parameter, exclusion_parameter, transformation_parameter, statistical_parameter))

            update_combinations, existing_files_set = self.get_update_combinations(combinations)
            
            
            if update_combinations:
                batch_index = len(existing_files_set)
                for i in tqdm(range(0, len(update_combinations), 20), desc="Processing combinations..."):
                    batch_combinations = update_combinations[i:i + 20]
        
                    batch_results = Parallel(n_jobs=-1)(delayed(self.process_combination)(*comb) for comb in batch_combinations)

                    filtered_results = [result for result in batch_results if isinstance(result, pd.DataFrame) and not result.empty]

                    if len(filtered_results) > 0:
                        while True:
                            performance_file_name = f'{self.plot_for_data_folder_name}/Performance_of_{self.selected_analyte}_{self.patient_category}_{batch_index + 1}.xlsx'
                            if performance_file_name not in existing_files_set:
                                self.save_to_excel(filtered_results, performance_file_name)
                                existing_files_set.add(performance_file_name)
                                batch_index += 1
                                break
                            else:
                                batch_index += 1

                self.merge_batch_files(self.selected_analyte, self.patient_per_day, self.patient_category)


        except Exception as e:
            print(f'General Error: {e}')

    def get_update_combinations(self, combinations):
        processed_combinations = set()
        existing_files_set = set()


        if os.path.exists(self.plot_for_data_folder_name):
            existing_files = [f for f in os.listdir(self.plot_for_data_folder_name) if f.endswith('.xlsx')]
            existing_files_set = set(existing_files)
            for file in existing_files:
                file_path = os.path.join(self.plot_for_data_folder_name, file)
                df_existing = pd.read_excel(file_path)

                for _, row in df_existing.iterrows():
                    combination = (
                        str(row['Block Size']).strip(), 
                        str(row['Truncation']).strip().lower(),
                        str(row['Exclusion']).strip().lower(),
                        str(row['Transformation']).strip().lower(),
                        str(row['Statistical Method']).strip().lower()
                    )
                    processed_combinations.add(combination)
                    

        update_combinations = [
            comb for comb in combinations
            if (
                str(comb[0]).strip(),
                str(comb[1]).strip().lower(),
                str(comb[2]).strip().lower(),
                str(comb[3]).strip().lower(),
                str(comb[4]).strip().lower()
            ) not in processed_combinations
        ]
        
        return update_combinations, existing_files_set

    def save_to_excel(self, results, performance_file_name):
        temp_file_name = performance_file_name + '.temp.xlsx'

        with FileLock(performance_file_name + '.lock'):
            try:
                combined_data = pd.concat(results)
                combined_data.to_excel(temp_file_name, index=False, engine='openpyxl')
                os.replace(temp_file_name, performance_file_name)
            except Exception as e:
                print(f"Error during Excel file writing: {e}")
            finally:
                if os.path.exists(temp_file_name):
                    try:
                        os.remove(temp_file_name)
                    except Exception as e:
                        print(f"Error deleting temporary file: {e}")

    def merge_batch_files(self, selected_analyte, patient_per_day, patient_category):
        batch_files = [f for f in os.listdir(self.plot_for_data_folder_name) if f.startswith(f'Performance_of_{selected_analyte}_{patient_category}_')]
        
        if not batch_files:
            print(f"Warning: No file to merge was found in the folder '{self.plot_for_data_folder_name}'.")
            return

        combined_data = pd.DataFrame()

        for file in batch_files:
            file_path = os.path.join(self.plot_for_data_folder_name, file)
            print(f"Reading: {file_path}")
            batch_data = pd.read_excel(file_path)
            combined_data = pd.concat([combined_data, batch_data], ignore_index=True)

        combined_file_name = f'{self.plot_for_data_folder_name}/Performance_{selected_analyte}_{patient_category}_combined.xlsx'
        combined_data.to_excel(combined_file_name, index=False)
        print(f'All batch files are merged as {combined_file_name}.')
         
        
            
    def process_combination(self, block_size, truncation_parameter, exclusion_parameter, transformation_parameter, statistical_parameter):
        block_size = int(block_size)

        self.convert_factor = 1  
                    
        truncation_of_first_half = Truncation(
                        self.flattened_training_data,
                        truncation_parameter, 
                        exclusion_parameter, 
                        self.selected_limit_RCVg)
        self.training_lower_trun_limit, self.training_upper_trun_limit, self.z_score_training = truncation_of_first_half.determine_truncation_limit()
        self.truncated_first_half = truncation_of_first_half.truncate_data(
                        self.training_lower_trun_limit, 
                        self.training_upper_trun_limit, 
                        block_size, 
                        self.z_score_training
                    )

        transformation_of_first_half = Transformation(
                self.selected_analyte, 
                self.truncated_first_half, 
                transformation_parameter, 
                self.convert_factor)
        self.transformed_training_data, self.lambda_value_training = transformation_of_first_half.transform_data()
          
                      
        self.biased_transformed_daily_series = convert_array_to_list(self.transformed_training_data, self.original_shape_training_data, 'numpy_list')        
           
           
        statistical_process_for_control_limit = StatisticalProcess(self.biased_transformed_daily_series, statistical_parameter, 
                                               block_size, self.z_score_training, keep_nan_index=True)
        
        statistical_data_series_with_nan = statistical_process_for_control_limit.statistical_process_data()                 
        
        daily_statistical_data_without_nan = remove_nan_from_lists(statistical_data_series_with_nan)    
        
        
        cl_data_class = ControlLimitDetermination(daily_statistical_data_without_nan,self.control_limit_type)
               
        (self.daily_data_parameters_list, 
         self.daily_min,  self.daily_max, self.selected_lower_control_limit, self.selected_upper_control_limit, self.total_exceedances,
         self.daily_min_5_percentile_limit, self.daily_max_95_percentile_limit,self.new_daily_lower_limit,self.new_daily_upper_limit,
         self.daily_1_percentile_mean_limit, self.daily_99_percentile_mean_limit, self.whole_data_025_quantile_limit, self.whole_data_975_quantile_limit,
         self.whole_data_1_quantile_limit, self.whole_data_99_quantile_limit, self.whole_data_5_quantile_limit, self.whole_data_95_quantile_limit,
         self.total_exceedances_min_max_quantiles_daily, self.total_exceedances_new_daily, self.total_exceedances_025_975_quantiles_whole, self.total_exceedances_mean_1_99_quantiles_daily, 
         self.total_exceedances_1_99_quantiles_whole, self.total_exceedances_5_95_quantiles_whole, 
         self.cv_lower_third, self.cv_middle_third, self.cv_upper_third) = cl_data_class.calculate_control_limits()
        
        performance_of_last_half_data = PBRTQC_Biased(len(self.flattened_test_data), self.second_half_series, self.patient_category, 
                                                      self.allocation_ratio, self.training_data_m_f_ratio, self.test_data_m_f_ratio, self.training_data_age_median, self.test_data_age_median,
                                                      self.training_lower_trun_limit, self.training_upper_trun_limit, 
                                                      self.selected_limit_RCVg, self.selected_analyte, truncation_parameter, 
                                                      exclusion_parameter, transformation_parameter, statistical_parameter, block_size,
                                                      self.biased_adding_point, self.patient_per_day, self.control_limit_type, self.selected_lower_control_limit,
                                                      self.selected_upper_control_limit, self.cv_lower_third, self.cv_middle_third, 
                                                      self.cv_upper_third, 
                                                      self.lambda_value_training, self.convert_factor, self.z_score_training)
        
        

        boolValue, false_positive_rate, sum_of_MNPed, area_NPed, area_NPed_multiplied, data_for_plot = performance_of_last_half_data.counting_saving_NPed_automatic()
        if boolValue == 0:
            return boolValue
        else:
            return data_for_plot                                                             
          
class GroupedDataPlotter:
    def __init__(self, predata, max_allowable_bias, cvi_value , cvg_value, ii_value, RCVg, TEa_value, selected_benchmarking, performance_metric, another_performance_metric, another_performance_metric_label, max_sorted_number, list_count, control_limit_type, reverse):
        self.predata = predata
        self.max_allowable_bias = max_allowable_bias
        self.cvi_value = cvi_value
        self.cvg_value = cvg_value
        self.ii_value = ii_value
        self.RCVg = RCVg
        self.TEa_value = TEa_value
        self.cutoffLoss = 20
        self.max_sorted_number = max_sorted_number
        self.list_count = list_count
        self.selected_benchmarking = selected_benchmarking
        self.control_limit_type = control_limit_type
        self.performance_metric = performance_metric
        self.another_performance_metric = another_performance_metric
        self.another_performance_metric_label = another_performance_metric_label
        self.reverse = reverse
    def plot_grouped_data_from_excel(self):
        sorted_benchmarking, label_selected_benchmarking, analyte, convert_factor, ppd, bap, add_title = self.calculate_grouped_data(self.predata, "Daily", self.max_allowable_bias)
        self.plot_grouped_data(sorted_benchmarking, label_selected_benchmarking, analyte, self.max_allowable_bias, convert_factor, ppd, bap, add_title)

    def calculate_grouped_data(self, df, control_limit_type, max_allowable_bias,):
        convert_factor = float(df['Convert Factor'].values[0])
        ppd = float(df['Patient Per Day'].values[0])
        bap = float(df['Bias Adding Point'].values[0])
        analyte = df.loc[1, "Analyte"]

        
        results = []


        
        for index, row in df.iterrows():

            total_number_data = row["Test Data Size"]
            CVi = self.cvi_value
            CVg = self.cvg_value
            II = self.ii_value
            TEa = self.TEa_value
            RCVg = self.RCVg
            skewness = row["Skewness"]
            kurtosis = row["Kurtosis"]
            loss_row = row["Loss (%)"]
            transformation_parameter = row["Transformation"]
            truncation_parameter = row["Truncation"]
            statistical_parameter = row["Statistical Method"]
            block_size = row["Block Size"]                      
            sum_of_fpr = round(float(row['FPR (Bias = 0)']),2)
            sum_mnped = float(row['Sum of MNPed'])          
            RCVM = round(float(row['Robust CV Median (all NPed)']),2)
            patient_category = row["Patient Category"]        
            patient_cat_combination_parameter = f' {patient_category}'
            truncation_combination_parameter = f' {truncation_parameter}'
            transformation_combination_parameter = f' {transformation_parameter}'
            statistical_combination_parameter = f' {statistical_parameter}'
            block_size_combination_parameter = f' {block_size}'
            
            label = f"{patient_category}(~loss% = {loss_row:.0f}), {truncation_parameter}, {transformation_parameter}, {statistical_parameter}, BS({block_size}), FPR = {sum_of_fpr:.2f}"


            area_NPed_multiplied_bias = []
            area_NPed_bias = []
            bias_values_for_trapz = []
            missing_percentage_values = []


            for bias_category in BIAS_CATEGORIES:


                biased_column_name = f'Biased Results ({bias_category}%)'
                biased_median_column = f'Median({biased_column_name})'
                column_name = biased_median_column
                value = convert_factor * float(row[column_name])
                # Boş olanları NaN olarak atayın
                value = np.nan if pd.isna(value) else value
                area_NPed_multiplied_bias.append(value * abs(int(bias_category)) if not np.isnan(value) else np.nan)
                area_NPed_bias.append(value)
                bias_values_for_trapz.append(int(bias_category))  
                
                
                biased_column_name = f'Biased Results ({bias_category}%)'

            area_NPed_multiplied = np.trapz(
                [v for v in area_NPed_multiplied_bias if not np.isnan(v)],
                x=[b for i, b in enumerate(bias_values_for_trapz) if not np.isnan(area_NPed_multiplied_bias[i])])
            area_NPed = np.trapz(
                [v for v in area_NPed_bias if not np.isnan(v)],
                x=[b for i, b in enumerate(bias_values_for_trapz) if not np.isnan(area_NPed_bias[i])])

            allowable_area_NPed_multiplied = np.trapz(
                [p for i, p in enumerate(area_NPed_multiplied_bias) if not np.isnan(p) and abs(bias_values_for_trapz[i]) <= max_allowable_bias],
                x=[b for i, b in enumerate(bias_values_for_trapz) if not np.isnan(area_NPed_multiplied_bias[i]) and abs(b) <= max_allowable_bias])
            non_allowable_area_MNPed_multiplied = area_NPed_multiplied - allowable_area_NPed_multiplied


            
            
            if sum_of_fpr <= 5:
                
                results.append({
                    'index': index,
                    'CVi' : CVi,
                    'CVg' : CVg,
                    'II' : II,
                    'loss' : round(loss_row),
                    'TEa' : TEa,
                    'RCVg' : RCVg,
                    'label' : label,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'neat_total_data' : f'{total_number_data} ({round(loss_row)}%)',
                    'patient_cat_combination_parameter': patient_cat_combination_parameter,
                    'truncation_combination_parameter': truncation_combination_parameter,
                    'transformation_combination_parameter': transformation_combination_parameter,
                    'statistical_combination_parameter': statistical_combination_parameter,
                    'block_size_combination_parameter': block_size_combination_parameter,
                    'NPed_points': area_NPed_bias,
                    'missing_points': missing_percentage_values,          
                    'area_NPed': area_NPed,
                    'area_NPed_multiplied': area_NPed_multiplied,
                    'fpr':sum_of_fpr,
                    'allowable_area_NPed_multiplied': allowable_area_NPed_multiplied,
                    'non_allowable_area_MNPed_multiplied': non_allowable_area_MNPed_multiplied,
                    'sum_mnped': sum_mnped,
                    'RCVM': round(RCVM,2),
                })

            # Sorting and filtering results based on selected benchmarking
        benchmark = []
        if self.selected_benchmarking == "Lowest Total AUC (x Bias)":
            
            label_selected_benchmarking = 'area_NPed_multiplied'
            benchmark = sorted([x for x in results if x.get(label_selected_benchmarking, 0) != 0
                        and -0.5 < x['skewness'] < 0.5  
                        and -0.5 < x['kurtosis'] < 0.5   
                        and x['loss'] <= self.cutoffLoss
                        and not np.isnan(x[label_selected_benchmarking])
                        and not np.isinf(x[label_selected_benchmarking])
                        and x.get('allowable_area_NPed_multiplied', 0) > x.get('non_allowable_area_MNPed_multiplied_multiplied', 0)], 
                       key=lambda x: x['area_NPed_multiplied'])


        elif self.selected_benchmarking == "Lowest Sum of MNPed":
            label_selected_benchmarking = 'sum_mnped'
            benchmark = sorted([x for x in results if x.get(label_selected_benchmarking, 0) != 0
                        and -0.5 < x['skewness'] < 0.5  
                        and -0.5 < x['kurtosis'] < 0.5  
                        and x['loss'] <= self.cutoffLoss
                        and not np.isnan(x[label_selected_benchmarking])
                        and not np.isinf(x[label_selected_benchmarking])
                        and x.get('allowable_area_NPed_multiplied', 0) > x.get('non_allowable_area_MNPed_multiplied_multiplied', 0)], 
                       key=lambda x: x['sum_mnped'])


        add_title = f'Benchmarking : {self.selected_benchmarking}'
        
        sorted_benchmarking = benchmark.copy()
        
        return sorted_benchmarking, label_selected_benchmarking, analyte, convert_factor, ppd, bap, add_title


    def plot_grouped_data(self, all_sorted_benchmarking, label_selected_benchmarking, selected_analyte, max_allowable_bias, 
                           convert_factor, ppd, bap, add_title):
        try:
            plot_folder_name = f'Performances/Supplementary/{selected_analyte}'
            table_folder_name = f'Performances/Supplementary/{selected_analyte}'

            if not os.path.exists(plot_folder_name):
                os.makedirs(plot_folder_name)

            if not os.path.exists(table_folder_name):
                os.makedirs(table_folder_name)

            
            if self.list_count == "all":
               length_of_process = len(all_sorted_benchmarking)
            else:
               length_of_process =  self.list_count*self.max_sorted_number 
               
            for j in range(0, length_of_process,self.max_sorted_number):
                print(f"Processing batch starting at index {j}, range: {j}:{j + self.max_sorted_number}")
                sorted_benchmarking = all_sorted_benchmarking[j:j + self.max_sorted_number]

                
                fig, ax_main = plt.subplots(figsize=(16, 8))

       
                    
                columns = ["Analyte", "Patient Per Day", "Bias Point", "CVi", "CVg", "II", "TEa", "RCVg",  "Performance Order", "Patient Group Included", "Truncation (Exclusion)",  "Transformation", "Moving Statistics", "Block Size",
                           "Data Size after Truncation", "Robust CV MAD", "FPR",  "Sum of MNPed", "Sum of Total AUC (x Bias)",  ]
               
                if self.performance_metric == "MNPed":
                    comparision_metric = 'NPed_points'
                elif self.performance_metric == "Lowest Total AUC (x Bias)":
                    comparision_metric = 'area_NPed_multiplied' 
                else:
                    comparision_metric = 'NPed_points'
                results_df = pd.DataFrame(columns=columns)
        
                for i in range(self.max_sorted_number - 1, -1, -1): 
                    if i < len(sorted_benchmarking):
                        print(f"Index j={j}, i={i}") 
                        p_value = kstest(sorted_benchmarking[i][comparision_metric], 'norm').pvalue
                        result = "p < 0.05" if p_value < 0.05 else "p >= 0.05"
                        print(f'Index {j}({i}): p-value = {result}')
    
                        if sorted_benchmarking[i][comparision_metric]:
                            filtered_bias_categories = []
                            filtered_NPed_points = []

                            for b, v in zip(BIAS_CATEGORIES, sorted_benchmarking[i][comparision_metric]):
                     
                                if b == -1 and not np.isnan(v):
                               
                                    filtered_bias_categories.append(b)
                                    filtered_NPed_points.append(v)
                           
                                    virtual_bias_points = np.linspace(-0.9, -0.01, 10)  
                                    virtual_y_values = (10**4) * -1 / virtual_bias_points  
                                    filtered_bias_categories.extend(virtual_bias_points)
                                    filtered_NPed_points.extend(virtual_y_values)

                                elif b == 1 and not np.isnan(v):
                                    filtered_bias_categories.append(b)
                                    filtered_NPed_points.append(v)

                                    virtual_bias_points = np.linspace(0.01, 0.9, 10)  
                                    virtual_y_values = (10**4) / virtual_bias_points  
                                    filtered_bias_categories.extend(virtual_bias_points)
                                    filtered_NPed_points.extend(virtual_y_values)

                                elif not np.isnan(v) and not np.isinf(v):
                                    filtered_bias_categories.append(b)
                                    filtered_NPed_points.append(v)


                            if filtered_NPed_points: 
                                line2, = ax_main.plot(
                                filtered_bias_categories, filtered_NPed_points,
                                linestyle='-' if i != 0 else 'dashed',
                                linewidth=2,
                                color="blue" if i == 0 else None,
                                alpha=0.5,
                                label=(
                                    f"{sorted_benchmarking[i]['label']}"))
             
                                temp_df = pd.DataFrame({    
                                "Analyte": selected_analyte, 
                                "Patient Per Day" : ppd,
                                "Bias Point": bap,
                                "Performance Order": i,
                                "CVi": [sorted_benchmarking[i]['CVi']],
                                "CVg": [sorted_benchmarking[i]['CVg']],
                                "II": [sorted_benchmarking[i]['II']],
                                "TEa": [sorted_benchmarking[i]['TEa']],
                                "RCVg": [sorted_benchmarking[i]['RCVg']],
                                "Patient Group Included": [sorted_benchmarking[i]['patient_cat_combination_parameter']],
                                "Truncation (Exclusion)": [sorted_benchmarking[i]['truncation_combination_parameter']],
                                "Data Size after Truncation": [sorted_benchmarking[i]['neat_total_data']],
                                "Transformation": [sorted_benchmarking[i]['transformation_combination_parameter']],
                                "Moving Statistics": [sorted_benchmarking[i]['statistical_combination_parameter']],
                                "Block Size": [sorted_benchmarking[i]['block_size_combination_parameter']],
                                "Robust CV MAD": [sorted_benchmarking[i]['RCVM']], 
                                "Sum of MNPed": [int(sorted_benchmarking[i]['sum_mnped'])],
                                "Sum of Total AUC (x Bias)": [int(sorted_benchmarking[i]['area_NPed_multiplied'])],
                                "FPR": [sorted_benchmarking[i]['fpr']],                                  
                  
                                })
                                results_df = pd.concat([results_df, temp_df], ignore_index=True)  
                            
                                if i == 0: 
                                    for j in range(1, len(filtered_bias_categories)):
                                        x1, x2 = filtered_bias_categories[j-1], filtered_bias_categories[j]
                                        y1, y2 = filtered_NPed_points[j-1], filtered_NPed_points[j]

                                        if x1 < -max_allowable_bias < x2 or x1 > -max_allowable_bias > x2:
                                            boundary_x = -max_allowable_bias
                                            boundary_y = y1 + (y2 - y1) * (boundary_x - x1) / (x2 - x1)

                                            ax_main.plot(boundary_x, boundary_y, 'o', color='blue', markersize=5)
                                            ax_main.text(boundary_x, boundary_y, f'({boundary_y:.0f})', fontsize=12, ha='right')

                                        if x1 < max_allowable_bias < x2 or x1 > max_allowable_bias > x2:
                                            boundary_x = max_allowable_bias
                                            boundary_y = y1 + (y2 - y1) * (boundary_x - x1) / (x2 - x1)

                                            ax_main.plot(boundary_x, boundary_y, 'o', color='blue', markersize=5)
                                            ax_main.text(boundary_x, boundary_y, f'({boundary_y:.0f})', fontsize=12, ha='left')                            
                                                                      
                            else:
                                print(f"{i}. empty list")
                    
                table_file_name = f"{selected_analyte}_{label_selected_benchmarking}_{ppd}_{bap}_table.xlsx"
                save_path_table = os.path.join(table_folder_name, table_file_name)
                sorted_df = results_df.sort_values(by='Sum of MNPed', ascending=True)

                sorted_df.to_excel(save_path_table, index=False)

                ymin, ymax = ax_main.get_ylim()

                if not np.isnan(ymin) and not np.isnan(ymax) and not np.isinf(ymin) and not np.isinf(ymax):
                    ax_main.set_ylim(ymin, ymax)
                gradient = np.linspace(0, 1, 256)
                gradient = np.vstack((gradient, gradient))
                allowable_cmap = LinearSegmentedColormap.from_list("allowable_cmap", ["lightgreen", "darkgreen"])
                non_allowable_cmap = LinearSegmentedColormap.from_list("non_allowable_cmap", ["darkred", "lightcoral"])

                ax_main.imshow(gradient, aspect='auto', cmap=non_allowable_cmap,
                          extent=(min(BIAS_CATEGORIES), -max_allowable_bias, ymin, ymax), alpha=0.3)
                ax_main.imshow(gradient, aspect='auto', cmap=non_allowable_cmap,
                          extent=(max(BIAS_CATEGORIES), max_allowable_bias, ymin, ymax), alpha=0.3)
                ax_main.imshow(gradient, aspect='auto', cmap=allowable_cmap,
                          extent=(-max_allowable_bias, 0, ymin, ymax), alpha=0.3)
                ax_main.imshow(gradient, aspect='auto', cmap=allowable_cmap,
                          extent=(max_allowable_bias, 0, ymin, ymax), alpha=0.3)

                xticks = [int(b) for b in BIAS_CATEGORIES if int(b) % 10 == 0]

                xticks.append(-max_allowable_bias)
                xticks.append(max_allowable_bias)

                xticklabels = [f"{b}%" for b in BIAS_CATEGORIES if int(b) % 10 == 0]
                xticklabels.append(f"-{max_allowable_bias:.1f}%")
                xticklabels.append(f"{max_allowable_bias:.1f}%")

                ax_main.set_xticks(xticks)
                ax_main.set_xticklabels(xticklabels, rotation=90)
            
                ax_main.set_xlabel('Systematic Errors (Bias)', fontsize=12, fontweight='bold')
                ax_main.set_ylabel(f"{self.performance_metric}", fontsize=12, fontweight='bold')
                title = f'{selected_analyte} PBRTQC Performance Comparison ({add_title}) '
                ax_main.set_title(title, fontsize=14, fontweight='bold')

                y_max_value = sorted(
                                    [v for v in sorted_benchmarking[0][comparision_metric] if not np.isnan(v) and not np.isinf(v)],
                                        reverse=True
                                        )[1] 


                ax_main.set_ylim(0, y_max_value)
                ax_main.invert_xaxis()


                handles, labels = ax_main.get_legend_handles_labels()

                handles = handles[::-1]
                labels = labels[::-1]

                fig.legend(
                    handles,
                    labels,
                    loc='upper center',
                    prop={'family': 'sans-serif', 'size': 14} ,
                    bbox_to_anchor=(0.5, -0.05), 
                    ncol=1
                )

                for text_plot in fig.legends[0].get_texts():
                    if text_plot.get_text().startswith(sorted_benchmarking[0]['label']):
                        text_plot.set_weight('bold')  
                        text_plot.set_fontsize(15) 
                        text_plot.set_family("sans-serif")  

                plt.gcf().set_size_inches(12, 6)

                file_name = f'{selected_analyte}.tiff'

                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=600, bbox_inches='tight') 
                plt.show()
                plt.close()

                buffer.seek(0)
                image = Image.open(buffer)

                save_path = os.path.join(plot_folder_name, file_name)
                image.save(save_path, format="TIFF", compression="tiff_lzw")
            
            return fig
        except Exception as e:
            print(f'Hata: {e}')

def convert_list_to_array(series, data_title=None):
    
    combined_array = np.concatenate(series.values)
    if data_title : 
       print(f"\n{data_title} - Combined array shape: {combined_array.shape}, {type(combined_array)}")
       print(f"{data_title} - First 5 data:\n {combined_array[:5]}")
       print(f"{data_title} List Length = {len(combined_array)}")
    return combined_array

def convert_array_to_list(array, original_shapes, data_type, data_title=None):
    new_list = []
    index = 0
    
    if data_type == "python_list":
        for shape in original_shapes:
            new_list.append(array[index:index + shape].tolist())  
            index += shape
    else:
        for shape in original_shapes:
            new_list.append(array[index:index + shape]) 
            index += shape
    if data_title :        
       print(f"\n{data_title} (Entire Data: {type(pd.Series(new_list))},(List Data: {type(pd.Series(new_list)[0])}:")
       print(f"{data_title} - First 5 data:\n {pd.Series(new_list)[:5]}")
       for index, day_data in pd.Series(new_list)[:5].items():
           print(f"Day {index}: {data_title} List Length = {len(day_data)}")
    return pd.Series(new_list)  


def remove_nan_from_lists(series, data_title=None, from_index=None):
    def clean_list(x):
        if isinstance(x, list):

            if from_index is not None and from_index < len(x):
                original_index = from_index  
            else:
                original_index = None

            cleaned_list = [i for i in x if not pd.isna(i)]

            if original_index is not None and original_index < len(x):
                new_index = len([i for i in x[:original_index] if not pd.isna(i)])  
                return cleaned_list, new_index
            else:
                return cleaned_list, None
        else:  # Eğer numpy array ise
            if from_index is not None and from_index < len(x):
                original_index = from_index
            else:
                original_index = None

            cleaned_array = x[~pd.isna(x)]

            if original_index is not None and original_index < len(x):
                new_index = len(x[:original_index][~pd.isna(x[:original_index])]) 
                return cleaned_array, new_index
            else:
                return cleaned_array, None

    cleaned_series = series.apply(lambda x: clean_list(x)[0])
    new_index_series = series.apply(lambda x: clean_list(x)[1])

    if data_title:
        print(f"\n{data_title} - Combined list shape: {cleaned_series.shape}, length({len(cleaned_series)}), {type(cleaned_series)}")
        print(f"{data_title} - First 5 data:\n {cleaned_series[:5]}")
        for index, day_data in enumerate(cleaned_series[:5]):
            print(f"Day {index}: {data_title} List Length = {len(day_data)}")

    if from_index is not None:
        return cleaned_series, new_index_series
    else:
        return cleaned_series


def remove_nan_from_array(array, data_title=None):
    
    cleaned_array = array[~np.isnan(array)]  

    if data_title:
        print(f"\n{data_title} - Cleaned array shape: {cleaned_array.shape}, length: {len(cleaned_array)}")
        print(f"{data_title} - First 5 data after NaN removal: {cleaned_array[:5]}")

    return cleaned_array
        
def df_to_list(data, first_part_ratio, second_part_ratio, data_per_day, return_type='pandas', exclude_last_list=True):
  
    if first_part_ratio + second_part_ratio != 1:
        raise ValueError("The sum of first_part_ratio and second_part_ratio must be 1.")

    total_len = len(data)

    first_part_len = int(total_len * first_part_ratio)

    first_part = data.iloc[:first_part_len]
    second_part = data.iloc[first_part_len:]

    def split_into_chunks(df, chunk_size, exclude_last):
        chunked_data = [np.array(df[i:i + chunk_size]) for i in range(0, len(df), chunk_size)]
        if exclude_last and len(chunked_data[-1]) < chunk_size:
            chunked_data = chunked_data[:-1] 
        return chunked_data

    first_part_chunks = split_into_chunks(first_part, data_per_day, exclude_last_list)
    second_part_chunks = split_into_chunks(second_part, data_per_day, exclude_last_list)

    if return_type == 'pandas':
        return pd.Series(first_part_chunks), pd.Series(second_part_chunks)
    elif return_type == 'numpy':
        return np.array(first_part_chunks), np.array(second_part_chunks)
    else:
        raise ValueError("Invalid return_type. Use 'pandas' or 'numpy'.")




def add_bias(series, bias_adding_point, factor, random_mode=False):
    def apply_bias_to_list(data_list, start_point, end_point):
        for i in range(start_point, end_point):
            data_list[i] *= (1 + (factor / 100))
        return data_list


    return series.apply(lambda x: apply_bias_to_list(np.array(x), bias_adding_point, len(x)))

     
    
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
            
