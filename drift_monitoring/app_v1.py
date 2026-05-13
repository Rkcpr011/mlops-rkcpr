import streamlit as st
import pandas as pd
import os
import glob
# ✅ REPLACE WITH (0.6.7 imports)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from datetime import datetime, timedelta

# Local paths
DRIFT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data")
BASELINE_PATH = os.path.join(
    DRIFT_DATA_PATH, "baseline.csv")
DATADRIFT_PATH = os.path.join(
    DRIFT_DATA_PATH, "datadrift")

# ✅ Local folders list karo
def list_folders(base_path):
    if not os.path.exists(base_path):
        return []
    return [f for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f))]

# ✅ Local CSV files list karo
def list_csv_files(folder_path):
    return glob.glob(os.path.join(folder_path, "*.csv"))

# ✅ Local se CSV load karo
def load_csv(filepath):
    return pd.read_csv(filepath)

# ✅ Most recent folder dhundo — same logic, local
def find_most_recent_folder(max_days=7):
    for i in range(max_days):
        check_date = (datetime.now() - timedelta(days=i))\
                     .strftime('%Y-%m-%d')
        folder_path = os.path.join(DATADRIFT_PATH, check_date)
        if os.path.exists(folder_path):
            return folder_path
    return None

# ✅ Evidently — same, AWS se koi lena dena nahi tha
def calculate_data_drift_evidently(baseline_df, latest_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=baseline_df,
        current_data=latest_df)
    return report

def calculate_data_quality_evidently(baseline_df, latest_df):
    report = Report(metrics=[DataQualityPreset()])
    report.run(
        reference_data=baseline_df,
        current_data=latest_df)
    return report

# ✅ Streamlit UI — same logic, local data
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:", 
        ["Data Drift", "Data Quality"])

    if page == "Data Drift":
        st.header('Data Drift Analysis')
        most_recent_folder = find_most_recent_folder()

        if most_recent_folder:
            baseline_df = load_csv(BASELINE_PATH)
            baseline_df = baseline_df.drop(
                columns=['Loan_ID', 'Loan_Status'], 
                errors='ignore')

            latest_csv_files = list_csv_files(most_recent_folder)
            selected_file = st.selectbox(
                'Select the target dataset', 
                latest_csv_files)

            if selected_file:
                latest_df = load_csv(selected_file)
                drift_report = calculate_data_drift_evidently(
                    baseline_df, latest_df)

                safe_filename = os.path.basename(selected_file)\
                                .replace(":", "_")
                report_filename = f'drift_report_{safe_filename}.html'
                # drift_report.save_html(report_filename)
                # ✅ AFTER (replace both occurrences)
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(drift_report.get_html())

                with open(report_filename, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    st.components.v1.html(
                        html_content, 
                        height=1000, width=1000, scrolling=True)
        else:
            st.write('No folder found')

    elif page == "Data Quality":
        st.header('Data Quality Analysis')
        most_recent_folder = find_most_recent_folder()

        if most_recent_folder:
            baseline_df = load_csv(BASELINE_PATH)
            baseline_df = baseline_df.drop(
                columns=['Loan_ID', 'Loan_Status'], 
                errors='ignore')

            latest_csv_files = list_csv_files(most_recent_folder)
            selected_file = st.selectbox(
                'Select the target dataset', 
                latest_csv_files)

            if selected_file:
                latest_df = load_csv(selected_file)
                latest_df = latest_df.drop(
                    ['Prediction'], axis=1, errors='ignore')

                drift_report = calculate_data_quality_evidently(
                    baseline_df, latest_df)

                safe_filename = os.path.basename(selected_file)\
                                .replace(":", "_")
                report_filename = f'drift_report_{safe_filename}.html'

                # drift_report.save_html(report_filename)
                # ✅ AFTER (replace both occurrences)
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(drift_report.get_html())

                with open(report_filename, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    st.components.v1.html(
                        html_content, 
                        height=1000, width=1300, scrolling=True)
        else:
            st.write('No folder found')

if __name__ == "__main__":
    main()