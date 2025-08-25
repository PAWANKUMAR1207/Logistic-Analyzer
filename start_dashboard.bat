@echo off
echo === Logistics Deliveries Analytics Dashboard ===
echo Installing dependencies...
pip install streamlit pandas plotly numpy

echo.
echo Creating config directory...
mkdir .streamlit 2>nul

echo.
echo Creating Streamlit config...
echo [server] > .streamlit\config.toml
echo headless = true >> .streamlit\config.toml
echo address = "0.0.0.0" >> .streamlit\config.toml
echo port = 7000 >> .streamlit\config.toml
echo. >> .streamlit\config.toml
echo [browser] >> .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml

echo.
echo Starting dashboard on port 7000...
echo Access at: http://localhost:7000
echo Press Ctrl+C to stop
python -m streamlit run app.py --server.port 7000

pause