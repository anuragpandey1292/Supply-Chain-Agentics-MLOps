from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os

# (Keep your existing app initialization and Pydantic models here)

@app.post("/profile-dataset/")
async def profile_uploaded_dataset(file: UploadFile = File(...)):
    """
    Accepts an uploaded CSV file, runs the automated data profiler and anomaly detection,
    and returns a downloadable PDF Health & EDA Report Card.
    """
    try:
        # 1. Save uploaded file temporarily into the data directory
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Import and run your PDF Data Profiler Agent
        from src.data.pdf_profiler import PDFDataProfilerAgent
        
        output_pdf_path = f"data/{file.filename.split('.')[0]}_health_report.pdf"
        
        profiler = PDFDataProfilerAgent(file_path=file_path, output_pdf_path=output_pdf_path)
        profiler.generate_pdf_report()
        
        # 3. Return the generated PDF as a downloadable response
        if os.path.exists(output_pdf_path):
            return FileResponse(
                path=output_pdf_path, 
                media_type='application/pdf', 
                filename=os.path.basename(output_pdf_path)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate PDF report.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling Error: {str(e)}")