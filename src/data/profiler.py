import os
import pandas as pd
import numpy as np

class DataProfilerAgent:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.report = {}

    def load_data(self):
        """Dynamically loads dataset."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        print(f"Successfully loaded dataset: {self.file_path} with shape {self.df.shape}")

    def profile_data(self) -> dict:
        """
        Performs comprehensive data profiling: types, missing values, 
        statistical summaries, outliers, and anomalies.
        """
        if self.df is None:
            self.load_data()

        df = self.df
        total_rows = len(df)

        # 1. Basic Schema & Data Types
        data_types = df.dtypes.astype(str).to_dict()

        # 2. Missing Values Analysis
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / total_rows) * 100
        missing_summary = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentage
        }).to_dict(orient='index')

        # 3. Statistical Picture (Numerical Features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_summary = df[numeric_cols].describe().to_dict()

        # 4. Outlier Detection (using IQR method for numerical columns)
        outlier_summary = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_summary[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / total_rows) * 100
            }

        # 5. Compile Full Agentic Report
        self.report = {
            'total_rows': total_rows,
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'data_types': data_types,
            'missing_summary': missing_summary,
            'numerical_statistics': stats_summary,
            'outlier_summary': outlier_summary
        }

        return self.report

    def print_health_check(self):
        """Prints a clean human-readable executive summary of the data."""
        if not self.report:
            self.profile_data()

        print("\n" + "="*40)
        print("📊 AUTONOMOUS DATA PROFILING REPORT")
        print("="*40)
        print(f"Total Rows: {self.report['total_rows']}")
        print(f"Total Columns: {self.report['total_columns']}")
        print("\n--- Missing Values Check ---")
        for col, data in self.report['missing_summary'].items():
            if data['missing_count'] > 0:
                print(f"⚠️ {col}: {data['missing_count']} missing ({data['missing_percentage']:.2f}%)")
        print("✅ Missing values check completed.")
        print("="*40)

if __name__ == "__main__":
    # Test the profiler agent
    profiler = DataProfilerAgent("data/train.csv")
    report = profiler.profile_data()
    profiler.print_health_check()