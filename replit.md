# Overview

This project is a Logistics Deliveries Analytics Dashboard built with Streamlit. It provides comprehensive analytics and visualizations for delivery data, enabling users to track performance metrics, analyze delivery patterns, and gain insights into logistics operations. The dashboard offers interactive charts, KPI tracking, and data filtering capabilities for delivery management and optimization.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web framework for rapid dashboard development
- **Visualization**: Plotly Express and Plotly Graph Objects for interactive charts and graphs
- **Layout**: Wide layout configuration with expandable sidebar for controls
- **Caching**: Streamlit's @st.cache_data decorator for performance optimization of data processing functions

## Data Processing Architecture
- **Data Ingestion**: CSV file upload mechanism through Streamlit's file uploader
- **Data Processing**: Pandas for data manipulation, cleaning, and transformation
- **Data Validation**: Automatic type conversion with error handling for numeric and datetime fields
- **Missing Data Handling**: Strategic filling of NaN values for delivery quantities

## Analytics Components
- **RFM Analysis**: Customer segmentation functionality (partially implemented)
- **KPI Calculations**: Delivery performance metrics and success rates
- **Time Series Analysis**: Date-based filtering and trend analysis capabilities
- **Interactive Filtering**: Dynamic data exploration through Streamlit widgets

## Application Structure
- **Single-file Application**: Monolithic app.py structure for simplicity
- **Modular Functions**: Helper functions for data processing and analysis
- **Error Handling**: Comprehensive exception handling for data loading operations

# External Dependencies

## Python Libraries
- **streamlit**: Web application framework for dashboard interface
- **pandas**: Data manipulation and analysis library
- **plotly.express & plotly.graph_objects**: Interactive visualization libraries
- **numpy**: Numerical computing for data operations
- **datetime**: Date and time handling for temporal analysis
- **io**: Input/output operations for file handling

## Data Requirements
- **CSV Data Format**: Expects structured delivery data with specific columns
- **Required Fields**: Create Time, Last Update, Delivery Date, Weight, No of Pieces, Actual Delivered QTY, Rejected QTY
- **Date Format Compatibility**: Flexible datetime parsing with error tolerance