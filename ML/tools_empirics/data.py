# ======================================
# DATA
# Author:      Tommaso Zipoli
# Last Mod:    06/11/2025
# ======================================

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from copy import deepcopy

# =============================
# TRANSFORMATIONS
# =============================

transformations = {
    1: lambda x: x,                           # 1: no transformation
    2: lambda x: x.diff(),                    # 2: first difference Δx_t
    3: lambda x: x.diff().diff(),             # 3: second difference Δ²x_t
    4: lambda x: np.log(x),                   # 4: log(x_t)
    5: lambda x: np.log(x).diff(),            # 5: first difference of log Δlog(x_t)
    6: lambda x: np.log(x).diff().diff(),     # 6: second difference of log Δ²log(x_t)
    7: lambda x: (x / x.shift(1) - 1).diff()  # 7: Δ(x_t / x_{t-1} - 1)
}


# =============================
# DATA CLASS
# Handles data loading and processing
# =============================

class Data:
    def __init__(self, path):
        self.raw = pd.read_excel(path)
        self.processed = deepcopy(self.raw)
        self.T = 0
        self.N = 0
        self.transformations = transformations

    def process(self, col_to_drop, scaler=None, transform=True, verbose=True):
        """
        Designed for processing FRED Monthly Data.
    
        Steps:
        1. Converts first column to datetime and sets as index
        2. Transforms variables according to selected criteria
        3. Drops columns with given names
        4. Drops rows with NAs
        5. Scales all features according to given scaler
    
        Params:
        ----------
        col_to_drop (list): names of columns to be dropped
        scaler: options None, StandardScaler(), MinMaxScaler()
        transform (bool): whether to apply transformations
        verbose (bool): whether to print diagnostics
        """

        # Step 0: copy dataframe
        df = self.processed.copy()
        print(f" ⏳ Processing data...")
        
        # Step 1: convert first column to datetime and set as index
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')
        df = df.set_index(date_col)
        if verbose:
            print(f"    Column '{date_col}' converted to datetime and set as index.")

        # Step 2: drop selected columns
        if col_to_drop:
            df = df.drop(columns=col_to_drop, errors='ignore')
            if verbose:
                print(f"    Columns dropped: {col_to_drop}")
        
        # Step 3: extract codes row and remove it
        codes = df.iloc[0]
        df = df.iloc[1:].sort_index()

        # Step 4: apply transformations
        if transform:
            for col in df.columns:
                code = codes[col]
                func = self.transformations.get(code)
                if func:
                    df[col] = func(df[col])
                elif verbose:
                    print(f"⚠️ No transformation defined for code {code} in column '{col}'")
            if verbose:
                print("    Features transformed as indicated.")
        
        # Step 5: drop NAs
        df = df.dropna()
        if verbose:
            if not df.empty:
                print(f"     📅 First period: {df.index[0]}")
                print(f"     📅 Last period:  {df.index[-1]}")
            else:
                print("⚠️ DataFrame is empty after dropping NaNs.")

        # Step 6: scale features
        print("    DataFrame shape:", df.shape)
        
        # Check for infinite values
        inf_mask = ~np.isfinite(df.values)
        if inf_mask.any():
            print("    Found infinite values in the following cells:")
            rows, cols = np.where(inf_mask)
            for r, c in zip(rows, cols):
                print(f"  Row {df.index[r]}, Column '{df.columns[c]}': {df.iloc[r, c]}")
        else:
            print("    No infinite values found.")

        if scaler is not None:
            scaled_values = scaler.fit_transform(df)
            df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
            if verbose:
                print("    Features scaled.")

        print(" ✅ Data processed correctly")
        self.processed = df
        self.shape = df.shape
        self.T, self.N = df.shape[0], df.shape[1]
                
        
    def process_EA(self, scaler):
        """
        Designed for processing EA dataset
        by Barigozzi, M. and C. Lissona

        Removes first column; drop rows with NAs;
        scale all features; assigns shape.
        """
        df = self.raw
        df = df.iloc[:, 1:]
        df = df.dropna()
        if scaler is not None:
            scaled_values = scaler.fit_transform(df)
            df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

        self.shape = df.shape
        self.processed = df
        self.T = self.shape[0]
        self.N = self.shape[1]

    def export(self, export_path):
        if isinstance(self.processed, pd.DataFrame):
            self.processed.to_excel(export_path / "processed.xlsx", index=True) 
        else:
            raise TypeError("Data is not a DataFrame and cannot be exported. Ensure you call export() before converting to TensorDataset.")

    def to_tensor(self):
        """
        Convert the processed DataFrame to a PyTorch tensor.
        Keeps the DataFrame intact in self.processed.
        """
        if self.processed.empty:
            raise ValueError("Cannot convert empty DataFrame to tensor.")
    
        # Ensure numeric dtype
        self.tensor = torch.tensor(self.processed.astype(np.float32).values, dtype=torch.float32)
        print(f"Tensor created with shape {self.tensor.shape}")