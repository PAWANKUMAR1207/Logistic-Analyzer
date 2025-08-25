# Local Deployment Guide

## Files to Download
Download these files to your local machine:
- `app.py` (main dashboard application)
- `run_local.py` (startup script)
- `README.md` (documentation)

## Quick Setup (Option 1)

1. **Run the setup script:**
   ```bash
   python run_local.py
   ```
   This will automatically install dependencies and start the dashboard on port 7000.

## Manual Setup (Option 2)

1. **Install Python dependencies:**
   ```bash
   pip install streamlit pandas plotly numpy
   ```

2. **Create Streamlit config directory:**
   ```bash
   mkdir -p .streamlit
   ```

3. **Create config file `.streamlit/config.toml`:**
   ```toml
   [server]
   headless = true
   address = "0.0.0.0"
   port = 7000

   [theme]
   primaryColor = "#1f77b4"
   backgroundColor = "#ffffff"
   secondaryBackgroundColor = "#f0f2f6"
   textColor = "#262730"

   [browser]
   gatherUsageStats = false
   ```

4. **Run the dashboard:**
   ```bash
   python -m streamlit run app.py --server.port 7000
   ```

## Access Your Dashboard

Open your browser and go to:
- **Local access:** http://localhost:7000
- **Network access:** http://YOUR_IP_ADDRESS:7000

## Usage

1. Upload your CSV delivery data file
2. Optionally upload address book data for enhanced analytics
3. Use the sidebar filters to analyze your data
4. Explore the various analytics sections

## Data Format

Your CSV should include these columns:
- Waybill, Create Time, Last Update, Receiver Code
- Delivery City, Current Status, Direction, SKU
- Weight, No of Pieces, Actual Delivered QTY
- Delivery Date, Rejected QTY, Rejection Status
- Rejected Reason, Business Type

## Requirements

- Python 3.7+
- Internet connection for initial package installation
- Modern web browser
- 4GB+ RAM recommended for large datasets

## Troubleshooting

**Port already in use:**
```bash
python -m streamlit run app.py --server.port 7001
```

**Permission errors:**
Run as administrator/sudo on Windows/Linux

**Package installation issues:**
```bash
pip install --upgrade pip
pip install --user streamlit pandas plotly numpy
```