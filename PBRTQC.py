import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statistics
from stqdm import stqdm

# Cấu hình trang
st.set_page_config(layout="wide", page_title="PBQC Optimizer (Raw Data)", page_icon="📈")

st.markdown("## **:blue[PBQC Parameters Optimizer (Fixed Truncation & Raw Data)]**")
st.markdown("Phần mềm tối ưu hóa tham số QC dựa trên bệnh nhân, sử dụng dữ liệu thô và giới hạn cắt cố định.")
st.write("---")

# --- SIDEBAR: UPLOAD & DATA SELECTION ---
with st.sidebar:
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx) hoặc CSV (.csv)", type=['csv', 'xlsx'])

    @st.cache_data
    def load_data(file):
        try:
            df = pd.read_excel(file)
        except:
            df = pd.read_csv(file, sep=None, engine='python')
        return df

    analyte_data = None
    day_data = None

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        st.header("2. Column Selection")
        col_result = st.selectbox("Chọn cột Kết quả xét nghiệm", tuple(df.columns))
        col_date = st.selectbox("Chọn cột Ngày/Batch", tuple(df.columns))
        
        # Xử lý dữ liệu
        analyte_data = df[col_result].dropna().reset_index(drop=True)
        # Lấy ngày tương ứng với dữ liệu kết quả (sau khi dropna)
        day_data = df.loc[analyte_data.index, col_date].reset_index(drop=True)
        
        st.success(f"Đã tải {len(analyte_data)} dòng dữ liệu.")
    else:
        st.info("Vui lòng upload dữ liệu để bắt đầu.")

# --- MAIN CONTENT ---

if analyte_data is not None:
    # --- INPUT PARAMETERS ---
    st.subheader("🛠️ Thiết lập tham số")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        TEa = st.number_input('**:red[TEa (%)]** - Tổng sai số cho phép', value=5.0, step=0.1, format="%.2f")
    with c2:
        allowable_FPR = st.number_input('**:red[Max FPR (%)]** - Tỷ lệ dương tính giả tối đa', value=10.0, step=0.1, format="%.1f")
    
    # Tính toán min/max gợi ý cho truncation
    data_min = float(analyte_data.min())
    data_max = float(analyte_data.max())

    with c3:
        # User tự nhập Truncation Limit (Thay vì thuật toán tự tìm)
        lower_trunc = st.number_input('**:blue[Lower Truncation Limit]**', value=data_min, format="%.2f")
    with c4:
        upper_trunc = st.number_input('**:blue[Upper Truncation Limit]**', value=data_max, format="%.2f")

    st.write("---")
    
    # Nút chạy tối ưu
    if st.button("🚀 **Bắt đầu Tối ưu hóa (EWMA)**", type="primary"):
        
        # 1. TRUNCATE DATA THEO INPUT CỦA NGƯỜI DÙNG
        mask = (analyte_data >= lower_trunc) & (analyte_data <= upper_trunc)
        clean_data = analyte_data[mask].reset_index(drop=True)
        clean_day = day_data[mask].reset_index(drop=True)
        
        if len(clean_data) < 50:
            st.error("Dữ liệu sau khi cắt lọc quá ít để phân tích. Vui lòng nới rộng khoảng Truncation Limit.")
            st.stop()

        # Tính Mean/SD trên Raw Data (đã lọc)
        mean_val = np.mean(clean_data)
        std_val = np.std(clean_data)
        
        st.info(f"**Thống kê dữ liệu sau lọc:** Mean = {mean_val:.2f}, SD = {std_val:.2f}, N = {len(clean_data)}")

        # 2. CHUẨN BỊ VÒNG LẶP TỐI ƯU
        # Các tham số cố định và biến đổi
        error_added_point = 10  # Warm-up period
        max_block_size_limit = 160
        
        # Vòng lặp: Error Rate (0% để tính FPR, TEa% để tính Độ nhạy)
        sim_errors = [0.0, TEa] 
        
        # Vòng lặp: Control Limits (Sigma multipliers từ 0.5 đến 4.0)
        limits_range = np.arange(0.5 * std_val, 4.2 * std_val, 0.5 * std_val)
        
        # Vòng lặp: Block Size (Span của EWMA từ 10 đến 160)
        block_sizes = np.arange(10, max_block_size_limit, 10)

        # Tổng số lần lặp để hiển thị thanh tiến trình
        total_iterations = len(sim_errors) * len(limits_range) * len(block_sizes) * clean_day.nunique()
        
        performance_metrics = {}
        
        # Khởi tạo dataframe cơ sở để tính toán nhanh hơn
        base_df = pd.DataFrame({'Day': clean_day, 'Data': clean_data})
        
        # --- BẮT ĐẦU CHẠY SIMULATION ---
        with stqdm(total=total_iterations, unit='iter', desc="Đang tối ưu hóa...") as pbar:
            
            for error_rate in sim_errors:
                # Tạo dữ liệu lỗi giả lập
                # Nếu error_rate = 0 -> Dữ liệu gốc -> Tính FPR
                # Nếu error_rate = TEa -> Dữ liệu lỗi -> Tính Sensitivity (Ped)
                
                # Copy dữ liệu để không ảnh hưởng vòng lặp sau
                sim_df = base_df.copy()
                
                # Control Limit Loop
                for limit_width in limits_range:
                    UCL = mean_val + limit_width
                    LCL = mean_val - limit_width
                    
                    # Block Size Loop
                    for block_size in block_sizes:
                        
                        alerts = 0
                        detection_indexes = [] # Lưu chỉ số phát hiện lỗi
                        
                        # Loop qua từng ngày/batch
                        unique_days = sim_df['Day'].unique()
                        
                        for day in unique_days:
                            day_df = sim_df[sim_df['Day'] == day].copy()
                            day_df = day_df.reset_index(drop=True)
                            
                            # Thêm lỗi nhân tạo (Simulation Error)
                            if len(day_df) > error_added_point:
                                # Chỉ thêm lỗi vào phần sau giai đoạn warm-up
                                day_df.loc[error_added_point:, 'Data'] *= (1 + error_rate / 100)
                            
                            # Tính EWMA
                            ewma = day_df['Data'].ewm(span=block_size, adjust=False).mean()
                            
                            # Check Alerts
                            is_high = ewma >= UCL
                            is_low = ewma <= LCL
                            
                            if is_high.any() or is_low.any():
                                alerts += 1
                                
                                # Tìm điểm phát hiện lỗi đầu tiên
                                first_high = day_df[(day_df.index >= error_added_point) & is_high].index.min()
                                first_low = day_df[(day_df.index >= error_added_point) & is_low].index.min()
                                
                                detected_idx = None
                                if pd.notna(first_high) and pd.notna(first_low):
                                    detected_idx = min(first_high, first_low)
                                elif pd.notna(first_high):
                                    detected_idx = first_high
                                elif pd.notna(first_low):
                                    detected_idx = first_low
                                    
                                if detected_idx is not None:
                                    detection_indexes.append(detected_idx + 1 - error_added_point)

                            pbar.update(1)
                        
                        # Tổng hợp kết quả cho bộ tham số này
                        positive_rate = alerts / len(unique_days)
                        anped = statistics.mean(detection_indexes) if detection_indexes else np.nan
                        mnped = statistics.median(detection_indexes) if detection_indexes else np.nan
                        
                        # Key lưu trữ: error_rate | limit_width | block_size
                        performance_metrics[f"{error_rate}|{limit_width}|{block_size}"] = {
                            'error_rate': error_rate,
                            'control_limit_width': limit_width,
                            'block_size': block_size,
                            'positive_rate': positive_rate,
                            'ANPed': anped,
                            'MNPed': mnped
                        }

        # --- XỬ LÝ KẾT QUẢ ---
        results_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
        
        # Tách kết quả thành 2 bảng: FPR (error=0) và Error Detection (error=TEa)
        df_fpr = results_df[results_df['error_rate'] == 0].copy()
        df_ed = results_df[results_df['error_rate'] == TEa].copy()
        
        # Đổi tên cột để merge
        df_fpr = df_fpr[['control_limit_width', 'block_size', 'positive_rate']].rename(columns={'positive_rate': 'FPR'})
        df_ed = df_ed[['control_limit_width', 'block_size', 'positive_rate', 'ANPed', 'MNPed']].rename(columns={'positive_rate': 'TPR'})
        
        # Merge dựa trên tham số (Limit & Block Size)
        merged = pd.merge(df_fpr, df_ed, on=['control_limit_width', 'block_size'])
        
        # Tính chỉ số Youden (Sensitivity + Specificity - 1) = TPR - FPR
        merged['Youden'] = merged['TPR'] - merged['FPR']
        
        # Lọc theo điều kiện người dùng: FPR <= Allowable FPR
        allowable_fpr_frac = allowable_FPR / 100.0
        valid_candidates = merged[merged['FPR'] <= allowable_fpr_frac]
        
        best_params = None
        
        if valid_candidates.empty:
            st.warning("Không tìm thấy tham số nào thỏa mãn mức FPR yêu cầu. Đang hiển thị tham số có Youden cao nhất bất kể FPR.")
            best_params = merged.loc[merged['Youden'].idxmax()]
        else:
            # Trong các ứng viên thỏa FPR, chọn cái có ANPed thấp nhất (phát hiện lỗi nhanh nhất)
            # Hoặc chọn Youden cao nhất. Ở đây ưu tiên Youden cao nhất trong nhóm FPR hợp lệ.
            best_params = valid_candidates.loc[valid_candidates['Youden'].idxmax()]

        # --- HIỂN THỊ KẾT QUẢ TỐI ƯU ---
        st.subheader("🏆 Tham số tối ưu nhất")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("##### Cấu hình đề xuất:")
            st.info(f"""
            - **Block Size (Span):** {int(best_params['block_size'])}
            - **Control Limit Width:** ±{best_params['control_limit_width']:.4f} (tương đương {best_params['control_limit_width']/std_val:.2f} SD)
            - **Truncation Limits:** {lower_trunc} - {upper_trunc}
            """)
            
        with res_col2:
            st.markdown("##### Hiệu năng dự kiến:")
            st.success(f"""
            - **False Positive Rate (FPR):** {best_params['FPR']*100:.2f}% (Yêu cầu: < {allowable_FPR}%)
            - **True Positive Rate (Detection @ {TEa}% error):** {best_params['TPR']*100:.2f}%
            - **ANPed (Số mẫu trung bình để phát hiện lỗi):** {best_params['ANPed']:.1f}
            - **Youden Index:** {best_params['Youden']:.3f}
            """)
            
        # --- VẼ BIỂU ĐỒ HIỆU NĂNG (ANPed vs Error Rate) ---
        st.subheader("📊 Biểu đồ hiệu năng (ANPed Curve)")
        
        # Chạy simulation lại cho bộ tham số tốt nhất trên một dải error rate rộng hơn để vẽ biểu đồ
        plot_errors = np.concatenate([
            np.arange(-1.0 * TEa, 0, 0.2 * TEa), # Lỗi âm
            np.arange(0.2 * TEa, 1.2 * TEa, 0.2 * TEa) # Lỗi dương
        ])
        
        anped_list = []
        mnped_list = []
        err_list = []
        
        best_block = int(best_params['block_size'])
        best_UCL = mean_val + best_params['control_limit_width']
        best_LCL = mean_val - best_params['control_limit_width']
        
        with st.spinner("Đang vẽ biểu đồ..."):
            for err in plot_errors:
                det_indices = []
                for day in clean_day.unique():
                    d_df = base_df[base_df['Day'] == day].reset_index(drop=True).copy()
                    if len(d_df) > error_added_point:
                        d_df.loc[error_added_point:, 'Data'] *= (1 + err / 100)
                    
                    ewma_vals = d_df['Data'].ewm(span=best_block, adjust=False).mean()
                    
                    # Check breach
                    breaches = (ewma_vals >= best_UCL) | (ewma_vals <= best_LCL)
                    
                    first_idx = d_df[(d_df.index >= error_added_point) & breaches].index.min()
                    
                    if pd.notna(first_idx):
                        det_indices.append(first_idx + 1 - error_added_point)
                
                if det_indices:
                    anped_list.append(statistics.mean(det_indices))
                    mnped_list.append(statistics.median(det_indices))
                else:
                    anped_list.append(None)
                    mnped_list.append(None)
                err_list.append(err)

        # Plotly Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=err_list, y=anped_list, mode='lines+markers', name='ANPed (Average)'))
        fig.add_trace(go.Scatter(x=err_list, y=mnped_list, mode='lines+markers', name='MNPed (Median)', line=dict(dash='dot')))
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Error")
        
        fig.update_layout(
            title="Tốc độ phát hiện lỗi (ANPed) theo mức độ lỗi",
            xaxis_title="Error Rate (%)",
            yaxis_title="Số mẫu bệnh nhân (N)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
