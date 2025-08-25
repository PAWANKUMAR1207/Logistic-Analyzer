#!/bin/bash

echo "=== Logistics Deliveries Analytics Dashboard ==="
echo "Installing dependencies..."
pip3 install streamlit pandas plotly numpy

echo ""
echo "Creating config directory..."
mkdir -p .streamlit

echo ""
echo "Creating Streamlit config..."
cat > .streamlit/config.toml << EOF
[server]
headless = true
address = "0.0.0.0"
port = 7000

[browser]
gatherUsageStats = false
EOF

echo ""
echo "Starting dashboard on port 7000..."
echo "Access at: http://localhost:7000"
echo "Press Ctrl+C to stop"
python -m streamlit run app.py --server.port 7000