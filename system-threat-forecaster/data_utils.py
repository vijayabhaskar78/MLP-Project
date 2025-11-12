"""
Data Validation and Utility Functions for System Threat Forecaster
"""

import pandas as pd
import numpy as np


def validate_dataframe(df, name="DataFrame"):
    """
    Validate and print comprehensive statistics about a DataFrame.
    
    Args:
        df (pd.DataFrame): pandas DataFrame to validate
        name (str): Name of the DataFrame for display purposes
    
    Returns:
        dict: Dictionary containing validation results with keys:
            - shape: tuple of (rows, columns)
            - missing_columns: number of columns with missing values
            - duplicates: number of duplicate rows
            - infinite_values: total count of infinite values
    
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv('train.csv')
        >>> results = validate_dataframe(df, "Training Data")
    """
    print(f"\n{'='*50}")
    print(f"Validation Report for: {name}")
    print(f"{'='*50}")
    
    # Basic info
    print(f"\n📊 Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    print(f"\n📋 Data Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  • {dtype}: {count} columns")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    has_missing = missing[missing > 0]
    
    if len(has_missing) > 0:
        print(f"\n⚠️  Missing Values: {len(has_missing)} columns affected")
        print(f"  Top 5 columns with missing data:")
        for col in has_missing.nlargest(5).index:
            print(f"    • {col}: {missing[col]:,} ({missing_pct[col]}%)")
    else:
        print("\n✅ No missing values detected")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"\n⚠️  Duplicate Rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
    else:
        print("\n✅ No duplicate rows detected")
    
    # Infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print(f"\n⚠️  Infinite Values: {len(inf_counts)} columns affected")
        for col, count in list(inf_counts.items())[:5]:
            print(f"    • {col}: {count:,}")
    else:
        print("\n✅ No infinite values detected")
    
    print(f"\n{'='*50}\n")
    
    return {
        'shape': df.shape,
        'missing_columns': len(has_missing),
        'duplicates': dup_count,
        'infinite_values': sum(inf_counts.values()) if inf_counts else 0
    }


def clean_column_names(df):
    """
    Clean column names by removing special characters and standardizing format.
    
    Args:
        df (pd.DataFrame): DataFrame with columns to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column names
    """
    df.columns = (df.columns
                  .str.strip()
                  .str.replace(' ', '_')
                  .str.replace('[^a-zA-Z0-9_]', '', regex=True)
                  .str.lower())
    return df


def get_missing_value_summary(df):
    """
    Get a detailed summary of missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    
    Returns:
        pd.DataFrame: Summary DataFrame with columns: 
            - Missing_Count, Missing_Percentage, Data_Type
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    dtypes = df.dtypes
    
    summary = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_pct,
        'Data_Type': dtypes
    })
    
    # Sort by missing percentage descending
    summary = summary[summary['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    return summary


def detect_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        columns (list): List of column names to check. If None, checks all numeric columns
        threshold (float): IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
        dict: Dictionary with column names as keys and outlier counts as values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = {}
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': round(outlier_count / len(df) * 100, 2),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
    
    return outliers
