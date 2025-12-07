import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

# ============================================================
# Stationarity test
# ============================================================

def adf_pvalue(x, reg='ct'):
    x = pd.Series(x).dropna().values
    if x.std(ddof=1) == 0 or len(x) < 10:
        return 1.0
    return adfuller(x, autolag='AIC', regression=reg)[1]


# ============================================================
# Principle function
# ============================================================

def load_and_process_data(file_path, output_path, alpha=0.05):
    """
    1. upload the dataset.
    2. Identify non stationary series (ADF p > alpha).
    3. Take first difference of those variables.
    4. Standardize globally.
    5. de-mean along columns(variables).
    6. export in Excel.
    """
    # 1) upload
    df = pd.read_excel(file_path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    print(f"\n📂 Dataset loaded: {file_path}")
    print(f"initial shape: {df.shape}")

    # 2) Test ADF
    pvals = {c: adf_pvalue(df[c], reg='ct') for c in df.columns}
    nonstat = [c for c, p in pvals.items() if p > alpha]
    print(f"\n📊 Serie non stazionarie (p > {alpha}): {nonstat if nonstat else 'nessuna'}")

    # 3) First differences
    df_proc = df.copy()
    for c in nonstat:
        df_proc[c] = df_proc[c].diff()

    if len(nonstat) > 0:
        df_proc = df_proc.iloc[1:, :]

    # 4) global standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_proc.values.astype(float))
    df_scaled = pd.DataFrame(X_scaled, index=df_proc.index, columns=df_proc.columns)

    # 5) Demeaning
    df_final = df_scaled - df_scaled.mean(axis=0)

    # 6) export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_excel(output_path)
    print(f"\n✅ Data are treated and exported in: {output_path}")
    print(f"Final shape: {df_final.shape}")
    print(f"{len(nonstat)} differentiated variables on {df_final.shape[1]} total.\n")

    return df_final, pd.Series(pvals, name="ADF_pval").sort_values()


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent
    input_file = base_path / "data" / "raw.xlsx"
    output_file = base_path / "data" / "processed2.xlsx"

    df_out, pvals = load_and_process_data(input_file, output_file)
