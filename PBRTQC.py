import streamlit as st
import requests
import math
import pandas as pd

def fetch_detail(spec_id):
    """
    Gọi endpoint chi tiết của 1 specification, trả về (within_subject, between_subject).
    """
    base_url = "https://biologicalvariation.eu/api/bv_specifications"
    detail_url = f"{base_url}/{spec_id}"
    resp = requests.get(detail_url)
    if resp.status_code == 200:
        detail_data = resp.json()
        # 'bv_estimates' là mảng, thường phần tử đầu có within_subject_variation, between_subject_variation
        estimates = detail_data.get("bv_estimates", [])
        if len(estimates) > 0:
            within_subj = estimates[0].get("within_subject_variation", None)
            between_subj = estimates[0].get("between_subject_variation", None)
            return within_subj, between_subj
    return None, None

def scrape_all_data(limit=20):
    """
    Lấy toàn bộ dữ liệu:
      - Measurand
      - Reference
      - within_subject_variation
      - between_subject_variation (từ trang chi tiết)
    """
    base_url = "https://biologicalvariation.eu/api/bv_specifications"
    
    # Gọi trang đầu để biết total
    resp = requests.get(base_url, params={"limit": limit, "offset": 0})
    if resp.status_code != 200:
        st.error("Không thể truy cập API trang BV Specifications!")
        return []
    
    json_data = resp.json()
    total = json_data["meta"]["total"]  # tổng số item
    st.write(f"Tổng số item: {total}")
    
    # Tính số vòng lặp
    total_pages = math.ceil(total / limit)
    all_results = []
    
    for page_idx in range(total_pages):
        offset = page_idx * limit
        st.write(f"Đang tải trang {page_idx+1}/{total_pages} (offset={offset})...")
        
        resp_page = requests.get(base_url, params={"limit": limit, "offset": offset})
        if resp_page.status_code != 200:
            st.warning(f"Không load được offset={offset}")
            continue
        
        data_page = resp_page.json()
        items = data_page.get("data", [])
        
        # Duyệt qua các item
        for item in items:
            spec_id = item["id"]
            measurand = item.get("measurand", "")
            reference = item.get("reference", "")
            
            # Lấy detail
            within_subj, between_subj = fetch_detail(spec_id)
            
            all_results.append({
                "Measurand": measurand,
                "Reference": reference,
                "WithinSubject": within_subj,
                "BetweenSubject": between_subj
            })
    
    return all_results

def main():
    st.title("Demo Scrape BiologicalVariation.eu")
    if st.button("Bắt đầu lấy dữ liệu"):
        results = scrape_all_data(limit=20)
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.success("Hoàn thành!")

if __name__ == "__main__":
    main()
