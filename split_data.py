import pandas as pd
from sklearn.model_selection import train_test_split
import sys, os

def split_csv(file, train_ratio=0.7):
    """Splitting dataset"""
    df = pd.read_csv(file)
    
    # Splitting using sklearn
    train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=42, shuffle=True)
    
    # Output file name
    base, ext = os.path.splitext(file)
    train_file = f"{base}_train{ext}"
    test_file = f"{base}_test{ext}"
    
    # Save
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Split complete: {train_file} ({len(train_df)} rows), {test_file} ({len(test_df)} rows)")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_csv.py <input_csv> [train_ratio]")
        sys.exit(1)
    input_csv = sys.argv[1]
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    split_csv(input_csv, train_ratio=ratio)