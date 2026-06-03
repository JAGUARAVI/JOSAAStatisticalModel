import pandas as pd
import json
import os
import glob
import math
from io import StringIO

# Minimum rows to consider a file worth processing
MIN_VALID_ROWS = 50

def categorize_institute(name):
    name = name.upper()
    if "INDIAN INSTITUTE OF TECHNOLOGY" in name or "IIT " in name or "IIT," in name:
        return "IIT"
    if "NATIONAL INSTITUTE OF TECHNOLOGY" in name or "NIT " in name or "NIT," in name:
        return "NIT"
    if "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY" in name or "IIIT" in name:
        return "IIIT"
    return "GFTI"

def load_all_data(max_year=None):
    """Load all raw HTML data into a single DataFrame with human-readable columns.
    
    Returns DataFrame with columns: institute, program, quota, category, gender,
    opening_rank, closing_rank, year, round, type
    
    Args:
        max_year: If set, only load data up to (and including) this year.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, "raw_data")
    
    html_files = sorted(glob.glob(os.path.join(raw_data_dir, "*.html")))
    if not html_files:
        print("❌ No HTML files found in raw_data/")
        return pd.DataFrame()

    all_data = []
    
    for html_file in html_files:
        filename = os.path.basename(html_file)
        parts = filename.replace(".html", "").split("_round_")
        if len(parts) != 2: continue
        try:
            year, round_num = int(parts[0]), int(parts[1])
        except ValueError: continue
        
        if max_year is not None and year > max_year:
            continue
        
        with open(html_file, "r") as f:
            html = f.read()
        
        if html.count("<tr") - 1 < MIN_VALID_ROWS: continue
            
        try:
            df_list = pd.read_html(StringIO(html))
            if not df_list: continue
            df = df_list[0]
            
            num_cols = len(df.columns)
            if num_cols == 7:
                df.columns = ["institute", "program", "quota", "category", "gender", "opening_rank", "closing_rank"]
            elif num_cols == 6:
                df.columns = ["institute", "program", "quota", "category", "opening_rank", "closing_rank"]
                df["gender"] = "Gender-Neutral"
            else: continue

            df = df[~df["closing_rank"].astype(str).str.contains("P", na=False)]
            df = df[~df["opening_rank"].astype(str).str.contains("P", na=False)]
            df["opening_rank"] = pd.to_numeric(df["opening_rank"], errors="coerce")
            df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce")
            df = df.dropna(subset=["opening_rank", "closing_rank"])
            df["opening_rank"] = df["opening_rank"].astype(int)
            df["closing_rank"] = df["closing_rank"].astype(int)
            df = df[(df["opening_rank"] > 0) & (df["closing_rank"] > 0)]
            df["gender"] = df["gender"].fillna("Gender-Neutral")

            df["year"] = year
            df["round"] = round_num
            df["type"] = df["institute"].apply(categorize_institute)
            
            all_data.append(df)
        except: continue

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def clean_data():
    """Generate optimized ranks.json + metadata.json for the frontend."""
    combined_df = load_all_data()
    if combined_df.empty:
        print("❌ No data loaded")
        return
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "app", "public", "data")
    output_json = os.path.join(output_dir, "ranks.json")
    output_meta = os.path.join(output_dir, "metadata.json")
    
    # Rename to short keys for frontend
    df = combined_df.rename(columns={
        "institute": "i", "program": "p", "quota": "q", 
        "category": "c", "gender": "g", "closing_rank": "cr",
        "year": "y", "round": "r", "type": "t"
    })
    
    df = df[["i", "p", "q", "c", "g", "cr", "y", "r", "t"]]
    
    # ID Mapping Optimization
    categorical_cols = ["i", "p", "q", "c", "g", "t"]
    metadata = {}
    
    for col in categorical_cols:
        unique_vals = sorted(df[col].unique().tolist())
        mapping = {val: i for i, val in enumerate(unique_vals)}
        metadata[col] = unique_vals
        df[col] = df[col].map(mapping)

    records = df.to_dict(orient="records")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_json, "w") as f:
        json.dump(records, f)
        
    with open(output_meta, "w") as f:
        json.dump(metadata, f)
    
    print(f"✅ Saved optimized ranks.json with {len(records)} records")
    print(f"✅ Saved metadata.json for categorical decoding")

if __name__ == "__main__":
    clean_data()
