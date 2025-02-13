import streamlit as st
import requests
import math
import pandas as pd

@st.cache_data
def fetch_detail(spec_id):
    """
    Gọi endpoint chi tiết của 1 specification, trả về (within_subject, between_subject).
    Hàm này được cache để tránh gọi lại API nhiều lần cho cùng một spec_id.
    """
    base_url = "https://biologicalvariation.eu/api/bv_specifications"
    detail_url = f"{base_url}/{spec_id}"
    resp = requests.get(detail_url)
    if resp.status_code == 200:
        detail_data = resp.json()
        estimates = detail_data.get("bv_estimates", [])
        if len(estimates) > 0:
            within_subj = estimates[0].get("within_subject_variation", None)
            between_subj = estimates[0].get("between_subject_variation", None)
            return within_subj, between_subj
    return None, None

@st.cache_data
def fetch_page_data(offset, limit):
    """
    Gọi API lấy dữ liệu trang theo offset và limit.
    Hàm này được cache để tránh gọi lại API cho cùng một offset.
    """
    base_url = "https://biologicalvariation.eu/api/bv_specifications"
    resp_page = requests.get(base_url, params={"limit": limit, "offset": offset})
    if resp_page.status_code == 200:
         return resp_page.json()
    else:
         return None

def scrape_all_data(limit=20):
    """
    Lấy toàn bộ dữ liệu:
      - Measurand
      - Reference
      - within_subject_variation và between_subject_variation (từ trang chi tiết)
      
    Đồng thời hiển thị progress bar theo từng trang được load.
    """
    base_url = "https://biologicalvariation.eu/api/bv_specifications"
    
    # Gọi trang đầu tiên để biết tổng số item
    resp = requests.get(base_url, params={"limit": limit, "offset": 0})
    if resp.status_code != 200:
        st.error("Không thể truy cập API trang BV Specifications!")
        return []
    
    json_data = resp.json()
    if "meta" not in json_data or "total" not in json_data["meta"]:
        st.error(f"Không tìm thấy thông tin meta.total trong dữ liệu API. Dữ liệu trả về: {json_data}")
        return []
    
    total = json_data["meta"]["total"]
    st.write(f"Tổng số item: {total}")
    
    total_pages = math.ceil(total / limit)
    all_results = []
    
    # Khởi tạo progress bar
    progress_bar = st.progress(0)
    
    # Duyệt qua từng trang (mỗi trang có 'limit' item)
    for page_idx in range(total_pages):
        offset = page_idx * limit
        st.write(f"Đang tải trang {page_idx+1}/{total_pages} (offset={offset})...")
        
        data_page = fetch_page_data(offset, limit)
        if not data_page:
            st.warning(f"Không load được offset={offset}")
            continue
        
        items = data_page.get("data", [])
        for item in items:
            spec_id = item["id"]
            measurand = item.get("measurand", "")
            reference = item.get("reference", "")
            within_subj, between_subj = fetch_detail(spec_id)
            
            all_results.append({
                "Measurand": measurand,
                "Reference": reference,
                "WithinSubject": within_subj,
                "BetweenSubject": between_subj
            })
        
        # Cập nhật progress bar theo số trang đã load
        progress_bar.progress((page_idx + 1) / total_pages)
    
    return all_results

def main():
    # Đổi tiêu đề và hiển thị đường link tham khảo
    st.title("Website trích xuất dữ liệu biến thiên sinh học")
    st.write("Tham khảo: [Biological Variation](https://biologicalvariation.eu/bv_specifications)")
    
    if st.button("Bắt đầu lấy dữ liệu"):
        results = scrape_all_data(limit=20)
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            st.success("Hoàn thành!")
        else:
            st.error("Không có dữ liệu nào được lấy về.")

if __name__ == "__main__":
    main()
