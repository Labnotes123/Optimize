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

    # Sanity check: chỉ giữ indices chung
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

    FPR_filter = allowable_FPR/100
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
        error_rates = np.arange(0, 1.2*TEa_float, 1.0*TEa_float)
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
        current_iter = 0               # Đếm iteration

        # Tạo stqdm pbar (logic cũ)
        pbar = stqdm(total=total_iterations_ewma, unit='iteration')

        performance_metrics = {}
        number_of_odd_batches = 0

        percentiles = [0,1,2,3]
        reversed_percentiles = [97,98,99,100]
        percentile_values = np.percentile(renewable_data, percentiles)
        reversed_percentile_values = np.percentile(renewable_data, reversed_percentiles)

        tol = np.finfo(float).eps * 100
        error_rates = np.where(np.abs(error_rates) < tol, 0.0, error_rates)

        # MAIN LOOP
        for error_rate in error_rates:
            for lower_val, upper_val in zip(percentile_values, reversed_percentile_values):
                def truncate_func(arr, lims):
                    return arr[(arr >= lims[0]) & (arr <= lims[1])]
                
                data_trunc = truncate_func(renewable_data, (lower_val, upper_val))

                for transformation in ("Raw Data", "Box-Cox Transformed Data"):
                    if transformation == "Raw Data":
                        data2 = data_trunc
                    else:
                        if len(data_trunc[data_trunc>0])<2:
                            continue
                        fitted_data, fitted_lambda = stats.boxcox(data_trunc[data_trunc>0])
                        data2 = pd.Series(fitted_data, index=data_trunc[data_trunc>0].index).dropna()

                    day_masked = renewable_day_data[data2.index]
                    data2 = data2.reset_index(drop=True)
                    day_masked = day_masked.reset_index(drop=True)

                    if len(data2)<2:
                        continue
                    mean_ = data2.mean()
                    std_  = data2.std()

                    df = pd.DataFrame({"Day": day_masked, "Data": data2})

                    for limit_val in np.arange(0.5*std_, 4.2*std_, 0.5*std_):
                        UCL_input = mean_ + limit_val
                        LCL_input = mean_ - limit_val

                        for block_size_ in np.arange(10, max_block_size, 10):
                            
                            for lam in np.arange(0.1, 1.1, 0.1):

                                alerts = 0
                                eD_index_list = []
                                unique_days = df["Day"].unique()

                                for dd in unique_days:
                                    day_df = df[df["Day"]==dd].reset_index(drop=True)

                                    if len(day_df)> error_added_point:
                                        day_df.loc[day_df.index[error_added_point:], "Data"] *= (1 + error_rate/100.0)
                                    else:
                                        number_of_odd_batches += 1

                                    arr_ = day_df["Data"].values
                                    ewma_vals = [arr_[0]]
                                    for i_ in range(1, len(arr_)):
                                        new_ewma = lam*arr_[i_] + (1-lam)*ewma_vals[-1]
                                        ewma_vals.append(new_ewma)
                                    day_df["ewma"] = ewma_vals

                                    day_df["EWMA higher than UCL"] = day_df["ewma"] >= UCL_input
                                    day_df["EWMA lower than LCL"] = day_df["ewma"] <= LCL_input

                                    if day_df["EWMA higher than UCL"].any() or day_df["EWMA lower than LCL"].any():
                                        alerts += 1

                                    first_u = day_df[(day_df.index>=error_added_point) & (day_df["EWMA higher than UCL"])].index.min()
                                    first_l = day_df[(day_df.index>=error_added_point) & (day_df["EWMA lower than LCL"])].index.min()

                                    if first_u is not None and first_u+1>=error_added_point:
                                        eD_index_list.append(first_u+1-error_added_point)
                                    if first_l is not None and first_l+1>=error_added_point:
                                        eD_index_list.append(first_l+1-error_added_point)

                                    # Cập nhật stqdm
                                    pbar.update(1)

                                    # Cập nhật progress_bar % (0-100)
                                    current_iter += 1
                                    percent_done = int( (current_iter / total_iterations_ewma) * 100 )
                                    if percent_done>100:
                                        percent_done=100
                                    progress_bar.progress(percent_done)

                                positive_rate = alerts / len(unique_days)
                                ANPed_ = statistics.mean(eD_index_list) if eD_index_list else float('nan')
                                MNPed_ = statistics.median(eD_index_list) if eD_index_list else float('nan')

                                key_str = f"{lower_val}-{upper_val}, {transformation}, {lam}, {LCL_input}, {UCL_input}, {block_size_}, {ANPed_}, {MNPed_}, {error_rate}"
                                performance_metrics[key_str] = positive_rate

        pbar.close()
        progress_bar.progress(100)  # đảm bảo kết thúc = 100%

        ############################
        # Xử lý kết quả
        ############################
        perf_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=["value"])
        perf_df.reset_index(inplace=True)
        perf_df.rename(columns={"index":"parameter"}, inplace=True)

        splitted = perf_df["parameter"].str.split(", ", expand=True)
        splitted.columns = [
            "truncation limit",
            "transformation status",
            "lambda",
            "lower control limit",
            "upper control limit",
            "block size",
            "ANPed",
            "MNPed",
            "TEa"
        ]
        perf_df = pd.concat([perf_df, splitted], axis=1).drop("parameter", axis=1)

        perf_df["value"] = perf_df["value"].astype(float)
        perf_df["TEa"]   = perf_df["TEa"].astype(float)
        perf_df["lambda"] = perf_df["lambda"].astype(float)
        perf_df["lower control limit"] = perf_df["lower control limit"].astype(float)
        perf_df["upper control limit"] = perf_df["upper control limit"].astype(float)
        perf_df["block size"] = perf_df["block size"].astype(int)
        perf_df["ANPed"] = perf_df["ANPed"].astype(float)
        perf_df["MNPed"] = perf_df["MNPed"].astype(float)

        df_TEa_0 = perf_df[perf_df["TEa"]==0]
        df_TEa_3 = perf_df[perf_df["TEa"]==TEa_float]

        merged_df = pd.merge(
            df_TEa_0, df_TEa_3,
            on=["truncation limit","transformation status","lambda","lower control limit","upper control limit","block size"],
            suffixes=("_0","_error")
        )

        merged_df.dropna(subset=["ANPed_error"], inplace=True)
        merged_df["ANPed_error"] = merged_df["ANPed_error"].round().astype(int)

        filtered_df = merged_df[(merged_df["value_0"]<FPR_filter) & (merged_df["TEa_0"]==0)]

        if len(filtered_df)>0:
            final_df = filtered_df.copy()
            final_df["youden"] = final_df["value_error"] - final_df["value_0"]
        else:
            st.write("False positive rate unattainable. Use entire merged_df.")
            final_df = merged_df.copy()
            final_df["youden"] = final_df["value_error"] - final_df["value_0"]

        best_youden_val = final_df["youden"].max()
        best_perf_out = final_df[ final_df["youden"]==best_youden_val ]
        
        min_anped = best_perf_out["ANPed_error"].min()
        Among_best_youden_best_ANPed_df = best_perf_out[ best_perf_out["ANPed_error"]==min_anped ]

        st.markdown("#### EWMA scheme parameters (lambda included) based on highest Youden + best ANPed")
        param_cols = [
            "truncation limit","transformation status","lambda",
            "lower control limit","upper control limit","block size"
        ]
        best_params_df = Among_best_youden_best_ANPed_df[param_cols].copy()
        st.dataframe(best_params_df, hide_index=True)

        def _metrics_maker(_df):
            _df2 = _df.copy()
            _df2["youden"] = _df2["value_error"] - _df2["value_0"]
            best_perf_dict = {
                "Metric": [
                    "Sensitivity (TPR)","Specificity","FPR","Youden Index","ANPed","MNPed"
                ],
                "Value": [
                    _df2["value_error"].iloc[0],
                    1-_df2["value_0"].iloc[0],
                    _df2["value_0"].iloc[0],
                    _df2["youden"].iloc[0],
                    _df2["ANPed_error"].iloc[0],
                    _df2["MNPed_error"].iloc[0]
                ]
            }
            return pd.DataFrame(best_perf_dict)
        
        perf_chosen_df = _metrics_maker(Among_best_youden_best_ANPed_df)
        st.dataframe(perf_chosen_df, hide_index=True)

        line_plot_of_ANPed(best_params_df, renewable_data, renewable_day_data)

        st.write("---")
        st.write("**Done**: EWMA + λ optimization")
else:
    st.info("Please upload a data file to proceed.")
