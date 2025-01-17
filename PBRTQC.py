import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statistics
from scipy import stats
from stqdm import stqdm

############################
# Streamlit Page Config
############################
st.set_page_config(
    layout="wide",
    page_title="QC Constellation",
    page_icon="📈"
)

st.title("MA/EWMA Bias Detection - Optimize λ, Truncation, Block Size, Control Limit")
st.write("---")

############################
# 1) Upload data
############################
uploaded_file = st.file_uploader("Upload .xlsx or .csv file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Đọc file
    try:
        if uploaded_file.name.endswith(".xlsx"):
            data_raw = pd.read_excel(uploaded_file)
        else:
            data_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Chọn cột data & cột day/batch
    col_data = st.selectbox("Select numeric column for analyte results", data_raw.columns)
    col_day = st.selectbox("Select day/batch column", data_raw.columns)

    data_series = data_raw[col_data].dropna().reset_index(drop=True)
    day_series = data_raw[col_day].dropna().reset_index(drop=True)

    # Sanity check: chỉ giữ indices chung
    common_len = min(len(data_series), len(day_series))
    data_series = data_series.iloc[:common_len]
    day_series = day_series.iloc[:common_len]
    data_series = data_series.reset_index(drop=True)
    day_series = day_series.reset_index(drop=True)

    st.write(f"Data length: {len(data_series)} records")
    st.write("---")

    # 2) Input widgets
    error_added_point = 10
    st.info(f"We will add bias after index={error_added_point} for each day/batch")

    col1, col2 = st.columns(2)
    TEa = col1.number_input("Allowable error (%)", min_value=0.1, value=5.0, step=0.1, format="%.1f")
    allowable_FPR = col2.number_input("Allowable false positive rate (%)", min_value=0.1, value=10.0, step=0.1,
                                      format="%.1f")

    FPR_filter = allowable_FPR / 100
    max_block_size = 160

    # convenience
    renewable_data = data_series
    renewable_day_data = day_series

    st.write("---")


    ############################
    # metric_maker
    ############################
    def metrics_maker(final_df):
        final_df = final_df.copy()
        final_df['youden'] = final_df['value_error'] - final_df['value_0']
        df_performance_of_best_parameters = {
            'Metric': [
                'Sensitivity (True Positive Rate)',
                'Specificity',
                'False Positive Rate',
                'Youden Index',
                'ANPed',
                'MNPed'
            ],
            'Value': [
                final_df['value_error'].iloc[0],
                1 - final_df['value_0'].iloc[0],
                final_df['value_0'].iloc[0],
                final_df['youden'].iloc[0],
                final_df['ANPed_error'].iloc[0],
                final_df['MNPed_error'].iloc[0]
            ]
        }
        return pd.DataFrame(df_performance_of_best_parameters)


    ############################
    # Plot function (dummy)
    ############################
    def line_plot_of_ANPed(best_param_df, data_original, day_original):
        st.write("**Line plot**: Illustrate ANPed/MNPed vs. bias (example).")
        # Thay đổi/hoàn thiện tuỳ ý.


    ############################
    # Optimize EWMA
    ############################
    if st.button("Optimize EWMA Scheme"):
        st.write("Running optimization... Please wait.")

        TEa_float = TEa
        error_rates = np.arange(0, 1.2 * TEa_float, 1.0 * TEa_float)
        # + truncated + transform + control limit + block_size + day_data + λ
        total_iterations_ewma = (
                len(error_rates)
                * 4  # pairs percentile
                * 2  # transformations
                * len(np.arange(0.5, 4.2, 0.5))
                * len(np.arange(10, max_block_size, 10))
                * day_series.nunique()
                * 10  # lam = 0.1->1.0
        )

        ############################
        # Tạo ST progress bar: 0-100%
        ############################
        progress_bar = st.progress(0)  # Thanh tiến trình
        current_iter = 0  # Đếm iteration

        # Tạo stqdm p
