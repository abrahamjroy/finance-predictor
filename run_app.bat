@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Finance Predictor...
streamlit run app.py
pause
