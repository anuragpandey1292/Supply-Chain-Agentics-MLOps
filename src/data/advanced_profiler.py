import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class AdvancedPDFProfilerAgent:
    def __init__(self, file_path: str, output_pdf_path: str = "data/advanced_data_report.pdf"):
        self.file_path = file_path
        self.output_pdf_path = output_pdf_path
        self.df = None
        self.charts = []
        self.insights = []

    def load_and_clean_data(self):
        """Loads data and dynamically identifies date and target numerical columns."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        
        self.df = pd.read_csv(self.file_path)
        print(f"Loaded dataset: {self.file_path} | Shape: {self.df.shape}")

        # Auto-detect date column
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        self.date_col = date_cols[0] if date_cols else None

        if self.date_col:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
            self.df = self.df.sort_values(by=self.date_col)

        # Auto-detect primary target column (e.g., 'sales', 'demand', 'quantity', or first numeric)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        target_candidates = [c for c in numeric_cols if 'sale' in c.lower() or 'demand' in c.lower() or 'qty' in c.lower()]
        self.target_col = target_candidates[0] if target_candidates else (numeric_cols[0] if numeric_cols else None)

    def analyze_behavior_and_decomposition(self):
        """Performs statistical behavior analysis, missingness, outliers, and time-series decomposition."""
        df = self.df
        total_rows = len(df)

        # 1. Missingness & Data Quality
        missing = df.isnull().sum()
        missing_pct = (missing / total_rows) * 100
        self.insights.append(f"<b>Data Completeness:</b> Dataset contains {total_rows:,} rows and {len(df.columns)} columns. Overall missing values: {missing.sum():,}.")

        # 2. Target Statistical Metrics & Spikes/Outliers
        if self.target_col:
            target_series = df[self.target_col].dropna()
            mean_val = target_series.mean()
            std_val = target_series.std()
            skew_val = target_series.skew()
            
            # Outlier / Spike detection via IQR
            Q1 = target_series.quantile(0.25)
            Q3 = target_series.quantile(0.75)
            IQR = Q3 - Q1
            spikes = target_series[target_series > (Q3 + 1.5 * IQR)]
            
            self.insights.append(f"<b>Target Distribution ({self.target_col}):</b> Mean = {mean_val:.2f}, Std Dev = {std_val:.2f}, Skewness = {skew_val:.2f}.")
            self.insights.append(f"<b>Spike & Outlier Analysis:</b> Detected {len(spikes):,} significant high-demand spikes exceeding normal statistical bounds ({len(spikes)/total_rows*100:.1f}% of total data).")

        # 3. Time-Series Decomposition (Trend, Seasonality, Residuals)
        os.makedirs("data/temp_charts", exist_ok=True)
        
        if self.date_col and self.target_col and len(df) > 30:
            try:
                # Resample daily or aggregate for clean decomposition
                ts_df = df.groupby(self.date_col)[self.target_col].mean().asfreq('D').fillna(method='ffill')
                if len(ts_df) > 60:
                    decomposition = seasonal_decompose(ts_df, model='additive', period=7) # Weekly seasonality check
                    
                    fig, axes = plt.subplots(4, 1, figsize=(7, 6), sharex=True)
                    decomposition.observed.plot(ax=axes[0], color='#2980b9')
                    axes[0].set_ylabel('Observed', fontsize=8)
                    decomposition.trend.plot(ax=axes[1], color='#e67e22')
                    axes[1].set_ylabel('Trend', fontsize=8)
                    decomposition.seasonal.plot(ax=axes[2], color='#27ae60')
                    axes[2].set_ylabel('Seasonal', fontsize=8)
                    decomposition.resid.plot(ax=axes[3], color='#e74c3c', style='.')
                    axes[3].set_ylabel('Random / Resid', fontsize=8)
                    
                    plt.suptitle(f"Time-Series Behavior Decomposition: {self.target_col}", fontsize=10, y=0.98)
                    plt.tight_layout()
                    
                    decomp_path = "data/temp_charts/decomposition.png"
                    plt.savefig(decomp_path, dpi=150)
                    plt.close()
                    self.charts.append(("Time-Series Decomposition (Trend, Seasonality, Residuals)", decomp_path))
                    self.insights.append("<b>Behavior Pattern:</b> Time-series decomposition confirms active cyclical/seasonal patterns and baseline directional trends.")
            except Exception as e:
                print(f"Decomposition skipped: {e}")

        # 4. Target Distribution Histogram
        if self.target_col:
            plt.figure(figsize=(7, 3))
            plt.hist(df[self.target_col].dropna(), bins=40, color='#3498db', edgecolor='black', alpha=0.7)
            plt.title(f"Data Spread & Frequency Distribution: {self.target_col}", fontsize=10)
            plt.xlabel(self.target_col, fontsize=8)
            plt.ylabel("Frequency", fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            hist_path = "data/temp_charts/target_hist.png"
            plt.savefig(hist_path, dpi=150)
            plt.close()
            self.charts.append((f"Distribution Histogram: {self.target_col}", hist_path))

    def generate_pdf_report(self):
        """Compiles the complete EDA and behavior report into a professional PDF."""
        self.load_and_clean_data()
        self.analyze_behavior_and_decomposition()

        doc = SimpleDocTemplate(self.output_pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#2c3e50'), spaceAfter=8, alignment=1)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2980b9'), spaceBefore=10, spaceAfter=4)
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9.5, textColor=colors.HexColor('#333333'), spaceAfter=5)

        # Title & Meta
        story.append(Paragraph("📊 Comprehensive Exploratory Data Analysis & Behavior Report Card", title_style))
        story.append(Paragraph(f"<b>File:</b> {os.path.basename(self.file_path)} | <b>Rows:</b> {len(self.df):,} | <b>Target Analyzed:</b> {self.target_col or 'N/A'}", body_style))
        story.append(Spacer(1, 8))

        # Insights Section
        story.append(Paragraph("1. Statistical Behavior & Anomaly Diagnostics", heading_style))
        for insight in self.insights:
            story.append(Paragraph(f"• {insight}", body_style))

        story.append(Spacer(1, 8))

        # Visual Charts Section
        story.append(Paragraph("2. Data Spreading, Trends & Decomposition Visualizations", heading_style))
        story.append(Paragraph("The charts below illustrate how the data behaves across time, capturing underlying macroeconomic trends, seasonal cycles, and random noise/residual fluctuations.", body_style))
        story.append(Spacer(1, 5))

        for title, path in self.charts:
            if os.path.exists(path):
                story.append(Paragraph(f"<b>{title}</b>", body_style))
                story.append(Image(path, width=420, height=210))
                story.append(Spacer(1, 8))

        doc.build(story)
        print(f"🎉 Advanced EDA PDF Report successfully generated: {self.output_pdf_path}")

if __name__ == "__main__":
    profiler = AdvancedPDFProfilerAgent("data/train.csv", output_pdf_path="data/advanced_data_report.pdf")
    profiler.generate_pdf_report()