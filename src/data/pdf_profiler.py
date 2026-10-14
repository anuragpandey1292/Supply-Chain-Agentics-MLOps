import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class PDFDataProfilerAgent:
    def __init__(self, file_path: str, output_pdf_path: str = "data/data_health_report.pdf"):
        self.file_path = file_path
        self.output_pdf_path = output_pdf_path
        self.df = None
        self.charts = []

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        print(f"Loaded dataset: {self.file_path} | Shape: {self.df.shape}")

    def generate_charts(self):
        """Generates distribution histograms and missingness plots for the report."""
        os.makedirs("data/temp_charts", exist_ok=True)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:3] # Pick top 3 numeric cols for charts

        for col in numeric_cols:
            plt.figure(figsize=(6, 3))
            plt.hist(self.df[col].dropna(), bins=30, color='#3498db', edgecolor='black', alpha=0.7)
            plt.title(f"Distribution & Spread: {col}", fontsize=10)
            plt.xlabel(col, fontsize=8)
            plt.ylabel("Frequency", fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            chart_path = f"data/temp_charts/{col}_dist.png"
            plt.savefig(chart_path, dpi=150)
            plt.close()
            self.charts.append((col, chart_path))

    def generate_pdf_report(self):
        """Compiles statistical insights, anomaly detections, and charts into a PDF report card."""
        if self.df is None:
            self.load_data()
        
        self.generate_charts()

        df = self.df
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Initialize PDF Document
        doc = SimpleDocTemplate(self.output_pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        styles = getSampleStyleSheet()

        # Custom Styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10,
            alignment=1
        )
        
        heading_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2980b9'),
            spaceBefore=12,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            'BodyDark',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            spaceAfter=6
        )

        # 1. Title Banner
        story.append(Paragraph("📊 Autonomous MLOps: Data Health & EDA Report Card", title_style))
        story.append(Paragraph(f"<b>Dataset Target:</b> {os.path.basename(self.file_path)} | <b>Total Rows:</b> {total_rows:,} | <b>Total Columns:</b> {total_cols}", body_style))
        story.append(Spacer(1, 10))

        # 2. Executive Summary / Anomaly Analysis Section
        story.append(Paragraph("1. Automated Data Diagnostics & Anomaly Detection", heading_style))
        
        missing_data = df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        
        if len(missing_cols) > 0:
            missing_text = f"⚠️ <b>Missing Values Detected:</b> {len(missing_cols)} columns contain missing data requiring imputation before model ingestion."
        else:
            missing_text = "✅ <b>Data Completeness:</b> Zero missing values detected across all columns."
        story.append(Paragraph(missing_text, body_style))

        # Check for sharp spikes / outliers using IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_notes = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[df[col] > upper_bound]
            if len(outliers) > (0.05 * total_rows): # If > 5% are outliers
                outlier_notes.append(f"• <b>{col}:</b> Exhibits high variance / potential spike anomalies ({len(outliers):,} outlier points detected above normal statistical spread).")

        if outlier_notes:
            story.append(Paragraph("⚠️ <b>Distribution Spikes & Outliers:</b>", body_style))
            for note in outlier_notes:
                story.append(Paragraph(note, body_style))
        else:
            story.append(Paragraph("✅ <b>Distribution Spread:</b> Feature distributions lie within healthy parametric bounds.", body_style))

        story.append(Spacer(1, 10))

        # 3. Statistical Summary Table
        story.append(Paragraph("2. Statistical Picture & Distribution Metrics", heading_style))
        
        table_data = [["Feature", "Mean", "Std Dev", "Min", "Max"]]
        for col in numeric_cols[:5]: # Include up to 5 numeric columns
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            table_data.append([col, f"{mean_val:.2f}", f"{std_val:.2f}", f"{min_val:.2f}", f"{max_val:.2f}"])

        t = Table(table_data, colWidths=[120, 90, 90, 90, 90])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2980b9')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8f9f9')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7'))
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        # 4. Embedded Visual Charts (Histograms & Data Spread)
        story.append(Paragraph("3. Feature Distribution Visualizations", heading_style))
        story.append(Paragraph("Histograms generated below illustrate the data spreading, skewness, and frequency spikes across core numerical features.", body_style))
        story.append(Spacer(1, 5))

        for col, chart_path in self.charts:
            if os.path.exists(chart_path):
                story.append(Paragraph(f"<b>Distribution Histogram: {col}</b>", body_style))
                story.append(Image(chart_path, width=400, height=200))
                story.append(Spacer(1, 10))

        # Build PDF
        doc.build(story)
        print(f"🎉 PDF Report Card successfully generated and saved to: {self.output_pdf_path}")

if __name__ == "__main__":
    # Test agent generation
    profiler = PDFDataProfilerAgent("data/train.csv", output_pdf_path="data/data_health_report.pdf")
    profiler.generate_pdf_report()