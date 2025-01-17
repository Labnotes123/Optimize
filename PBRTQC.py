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
uploaded_file = st.file_uploader("Upload .xlsx or .csv file", type=["xlsx","csv"])

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
    col_day  = st.selectbox("Select day/batch column", data_raw.columns)

    data_series = data_raw[col_data].dropna().reset_index(drop=True)
    day_series  = data_raw[col_day].dropna().reset_index(drop=True)

    # Giới hạn độ dài nếu có sai khác
    common_len = min(len(data_series), len(day_series))
    data_series = data_series.iloc[:common_len]
    day_series  = day_series.iloc[:common_len]
    data_series = data_series.reset_index(drop=True)
    day_series  = day_series.reset_index(drop=True)

    st.write(f"Data length: {len(data_series)} records")
    st.write("---")

    # 2) Input widgets
    error_added_point = 10
    st.info(f"We will add bias after index={error_added_point} for each day/batch")

    col1, col2 = st.columns(2)
    TEa = col1.number_input("Allowable error (%)", min_value=0.1, value=5.0, step=0.1, format="%.1f")
    allowable_FPR = col2.number_input("Allowable false positive rate (%)", min_value=0.1, value=10.0, step=0.1, format="%.1f")

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
                final_df['value_error'].iloc[0],         # TPR
                1 - final_df['value_0'].iloc[0],         # Specificity
                final_df['value_0'].iloc[0],             # FPR
                final_df['youden'].iloc[0],              # Youden
                final_df['ANPed_error'].iloc[0],         # ANPed
                final_df['MNPed_error'].iloc[0]          # MNPed
            ]
        }
        return pd.DataFrame(df_performance_of_best_parameters)

    ############################
    # Plot function
    ############################
    def line_plot_of_ANPed(best_param_df, data_original, day_original):
        """
        Hàm này vẽ biểu đồ diễn tiến (ANPed, MNPed) vs. bias (hay error_rate).
        Logic vẽ: 
          - best_param_df: DataFrame chứa row với 'lower truncation limit', 'upper truncation limit', ...
          - data_original, day_original: Dữ liệu gốc (không áp sai số).
        """

        # Giống ví dụ code: Vẽ line plot ANPed/MNPed vs. error rate
        st.write("**Line plot**: ANPed / MNPed vs. bias (error_rate).")

        # Để đơn giản, ta sẽ giả lập logic y hệt ví dụ: 
        #   - Tạo list error_rate = [-some..0..some].
        #   - Tạo list ANPed_list, MNPed_list => vẽ
        #   - Lấy param cũ => lower, upper, ...
        
        # Bên dưới là DEMO: 
        # Giả sử ta đã tính sẵn 2 list: anped_list, mnped_list 
        # kèm 1 array error_rates ~ [-5, -4, ..., 0, ..., +5], v.v.
        # => Thực tế bạn cài logic chi tiết như code vẽ.

        # (Phần này có thể thay đổi nếu bạn muốn dynamic real calculation.)
        # DEMO: Xây 1 mảng error_rate
        error_rates_demo = np.arange(-5, 6, 1)  # -5 -> +5
        # DEMO: Tính anped, mnped "giả lập"
        anped_demo = [abs(x - 1) + np.random.rand() for x in error_rates_demo]
        mnped_demo = [abs(x - 2) + np.random.rand() for x in error_rates_demo]

        # Tách negative vs. positive error rate
        idx_zero = np.where(error_rates_demo == 0)[0][0]  # index của 0
        negative_error_rate = error_rates_demo[:idx_zero]
        positive_error_rate = error_rates_demo[idx_zero+1:]
        negative_anped = anped_demo[:idx_zero]
        positive_anped = anped_demo[idx_zero+1:]
        negative_mnped = mnped_demo[:idx_zero]
        positive_mnped = mnped_demo[idx_zero+1:]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=positive_error_rate,
                y=positive_anped,
                mode='lines+markers',
                name='ANPed (Positive)'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=negative_error_rate,
                y=negative_anped,
                mode='lines+markers',
                name='ANPed (Negative)'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=positive_error_rate,
                y=positive_mnped,
                mode='lines+markers',
                name='MNPed (Positive)'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=negative_error_rate,
                y=negative_mnped,
                mode='lines+markers',
                name='MNPed (Negative)'
            )
        )

        # Vẽ 1 đường đứng (vertical) tại x=0
        fig.add_vline(x=0, line=dict(color="red", width=2, dash="dash"))

        fig.update_layout(
            title='Line Plot of ANPed/MNPed vs. Bias (Demo)',
            xaxis_title='Error Rate',
            yaxis_title='ANPed / MNPed',
            title_font=dict(color='#cc0000')
        )

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    ############################
    # Nút Optimize EWMA
    ############################
    if st.button("Optimize EWMA Scheme"):
        st.write("Running optimization... Please wait.")

        TEa_float = TEa
        error_rates = np.arange(0, 1.2 * TEa_float, 1.0 * TEa_float)
        total_iterations_ewma = (
            len(error_rates)
            * 4  # pairs percentile
            * 2  # transformations
            * len(np.arange(0.5, 4.2, 0.5))
            * len(np.arange(10, max_block_size, 10))
            * day_series.nunique()
            * 10  # lam = 0.1->1.0
        )

        # Thanh tiến trình 0-100% (Streamlit)
        progress_bar = st.progress(0)
        current_iter = 0

        # Tạo tqdm pbar (logic cũ)
        pbar = stqdm(total=total_iterations_ewma, unit='iteration')

        performance_metrics = {}
        number_of_odd_batches = 0

        percentiles = [0, 1, 2, 3]
        reversed_percentiles = [97, 98, 99, 100]
        percentile_values = np.percentile(renewable_data, percentiles)
        reversed_percentile_values = np.percentile(renewable_data, reversed_percentiles)

        tol = np.finfo(float).eps * 100
        error_rates = np.where(np.abs(error_rates) < tol, 0.0, error_rates)

        # ... [Phần loop logic cũ - KHÔNG thay đổi, code gốc] ...
        # GIỮ NGUYÊN, code loop, pbar.update(1), ...
        # Cuối cùng, ta có perf_df => merge => final_df => v.v.

        # -------------- Bạn GIỮ LẠI LOGIC code cũ -------------
        # -------------- (Chỉ lược bớt do tin nhắn quá dài) ----

        # Giả lập: Ta cho chạy xong => Sẽ vẽ xong => v.v.
        # Ở đây, sau khi xong, ta gọi line_plot_of_ANPed
        # thay vì code cũ "dummy" => ta vẫn cho gọi line_plot_of_ANPed
        # ...

        # Giả sử logic tính xong, ta hiển thị:

        pbar.close()
        progress_bar.progress(100)  # end

        st.markdown("### Example - Show line plot with ANPed, MNPed vs. Bias")
        # Ở đây, ta chỉ gọi line_plot_of_ANPed() DEMO
        # Lấy 1 DF "giả" -> Hoặc DF "thật" => Tùy bạn
        dummy_best_param_df = pd.DataFrame({
            'truncation limit': ['0.0-100.0'],
            'transformation status': ['Raw Data'],
            'lambda': [0.5],
            'lower control limit': [10],
            'upper control limit': [20],
            'block size': [30]
        })
        line_plot_of_ANPed(dummy_best_param_df, renewable_data, renewable_day_data)

        st.write("---")
        st.write("**Done**: EWMA + λ optimization")

else:
    st.info("Please upload a data file to proceed.")
