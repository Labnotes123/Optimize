import streamlit as st
import requests
import math
import pandas as pd
import concurrent.futures

@st.cache_data(show_spinner=False)
def fetch_detail(spec_id):
    """
    Gọi endpoint chi tiết của 1 specification, trả về (within_subject, between_subject).
    Hàm này được cache để tránh gọi lại API nhiều lần cho cùng một spec_id.
    """
    base_url = "https://biologicalvariation.eu/api/bv_specifications"
    detail_url = f"{base_url}/{spec_id}"
    try:
        resp = requests.get(detail_url, timeout=10)
        resp.raise_for_status()
        detail_data = resp.json()
        estimates = detail_data.get("bv_estimates", [])
        if estimates:
            within_subj = estimates[0].get("within_subject_variation", None)
            between_subj = estimates[0].get("between_subject_variation", None)
            return within_subj, between_subj
    except Exception as e:
        st.warning(f"Lỗi khi lấy chi tiết spec_id {spec_id}: {e}")
    return None, None

@st.cache_data(show_spinner=False)
def fetch_page_data(offset, limit):
    """
    Gọi API lấy dữ liệu trang theo offset và limit.
    Hàm này được cache để tránh gọi lại API cho cùng một offset.
    """
    base_url = "https://biologicalvariation.eu/api/bv_specifications"
    try:
        resp_page = requests.get(base_url, params={"limit": limit, "offset": offset}, timeout=10)
        resp_page.raise_for_status()
        return resp_page.json()
    except Exception as e:
        st.warning(f"Lỗi khi lấy dữ liệu trang offset {offset}: {e}")
        return None

def process_item(item):
    """
    Xử lý từng item: lấy thông tin cơ bản và gọi fetch_detail để lấy dữ liệu chi tiết.
    """
    spec_id = item.get("id")
    measurand = item.get("measurand", "")
    reference = item.get("reference", "")
    within_subj, between_subj = fetch_detail(spec_id)
    return {
        "Measurand": measurand,
        "Reference": reference,
        "WithinSubject": within_subj,
        "BetweenSubject": between_subj
    }

def scrape_all_data(limit=20, max_pages=None):
    """
    Lấy toàn bộ dữ liệu:
      - Measurand
      - Reference
      - within_subject_variation và between_subject_variation (từ trang chi tiết)
    
    Sử dụng progress bar và giới hạn số trang theo lựa chọn của người dùng.
    """
    base_url = "https://biologicalvariation.eu/api/bv_specifications"
    
    # Lấy trang đầu tiên để xác định tổng số item
    try:
        resp = requests.get(base_url, params={"limit": limit, "offset": 0}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Không thể truy cập API trang BV Specifications: {e}")
        return []
    
    json_data = resp.json()
    if "meta" not in json_data or "total" not in json_data["meta"]:
        st.error(f"Không tìm thấy thông tin meta.total trong dữ liệu API. Dữ liệu trả về: {json_data}")
        return []
    
    total = json_data["meta"]["total"]
    st.write(f"Tổng số item: {total}")
    
    total_pages = math.ceil(total / limit)
    if max_pages is None:
        pages_to_scrape = total_pages
    else:
        pages_to_scrape = min(max_pages, total_pages)
    st.write(f"Sẽ lấy dữ liệu từ {pages_to_scrape} trang.")
    
    all_results = []
    progress_bar = st.progress(0)
    
    with st.spinner("Đang tải dữ liệu..."):
        for page_idx in range(pages_to_scrape):
            offset = page_idx * limit
            st.write(f"Đang tải trang {page_idx+1}/{pages_to_scrape} (offset={offset})...")
            data_page = fetch_page_data(offset, limit)
            if not data_page:
                st.warning(f"Không load được offset={offset}")
                continue
            
            items = data_page.get("data", [])
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(process_item, items))
                all_results.extend(results)
            
            progress_bar.progress((page_idx + 1) / pages_to_scrape)
    
    return all_results

def main():
    st.title("Website trích xuất dữ liệu biến thiên sinh học")
    st.write("Tham khảo: [Biological Variation](https://biologicalvariation.eu/bv_specifications)")
    
    num_pages = st.number_input("Chọn số trang muốn lấy dữ liệu:", min_value=1, value=1, step=1)
    
    if st.button("Bắt đầu lấy dữ liệu"):
        results = scrape_all_data(limit=20, max_pages=num_pages)
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            st.success("Hoàn thành!")
        else:
            st.error("Không có dữ liệu nào được lấy về.")

if __name__ == "__main__":
    main()
