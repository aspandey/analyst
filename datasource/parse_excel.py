import pandas as pd
from typing import List, Dict
import os
import glob

HOME_DIR = os.path.expanduser("~")
STOCK_INFO_PATH = os.path.join(HOME_DIR, "factory/public/stocks_xlsx/")

def parse_xlsx_to_dicts(file_path: str) -> List[Dict]:
    """
    Parse an Excel (.xlsx) file into a list of dictionaries.
    
    Each row becomes a dict where keys = column names.
    """
    df = pd.read_excel(file_path)
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    records = []
    for _, row in df.iterrows():
        record = {k: v for k, v in row.items() if pd.notnull(v)}
        if record:
            records.append(record)
    return records

def clean_row_text(List_of_dicts: List[Dict]) -> List[Dict]:
    stocks_info : List[dict] = []
    for record in List_of_dicts:
        clean_row = {str(k).replace('\n', '').lower(): str(v).replace('\n', '').lower() if isinstance(v, str) else v for k, v in record.items()}
        stocks_info.append(clean_row)
    return stocks_info

def get_stock_info_from_xlsx(xls_folder_path: str) -> List[Dict[str, str]]:
    """
    Get stock information from an Excel files.
    """
    stocks_info = []
    excel_files = glob.glob(os.path.join(xls_folder_path, "*.xls*"))
    for file_path in excel_files:
        print(f"Processing file: {file_path}")
        rows = parse_xlsx_to_dicts(file_path)
        stocks_info.extend(clean_row_text(rows))
    return stocks_info

if __name__ == "__main__":

    stocks_info = get_stock_info_from_xlsx(STOCK_INFO_PATH)

    for row in stocks_info:        
        print(row)
