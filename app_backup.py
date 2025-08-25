import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io

# Set page configuration
st.set_page_config(
    page_title="Logistics Deliveries Analytics Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translation dictionary for Arabic rejection reasons
ARABIC_TRANSLATIONS = {
    'رفض الاستلام بسبب مشكلة في المنتج': 'Product quality issue - receipt refused',
    'رفض الاستلام بسبب كمية غير مطلوبة': 'Wrong quantity - receipt refused', 
    'رفض الاستلام بسبب مشاكل في المنتج': 'Product issues - receipt refused',
    'رفض الاستلام بسبب عدم مطابقة المنتج': 'Product mismatch - receipt refused',
    'رفض الاستلام بسبب التأخير': 'Delivery delay - receipt refused',
    'العميل غير قادر على الاستلام في الوقت المحدد': 'Customer unavailable at scheduled time',
    'العميل غير قادر على الاستلام في الوقت الحالي': 'Customer unavailable currently',
    'رفض الاستلام بسبب مشكلة في التسليم': 'Delivery issue - receipt refused',
    'رفض الاستلام بسبب مشكلة في العبوة': 'Packaging issue - receipt refused',
    'تأخير في توصيل الطلبية عن الموعد المحدد': 'Delayed delivery beyond scheduled time',
    'اختلاف المر المعلق على القيمة عن المرود للعميل': 'Value discrepancy from customer record',
    'رفض الاستلام بسبب وجود عيوب على القيمة': 'Receipt refused due to value defects',
    'رفض الاستلام بسبب اختلاف في السعر للقيمة': 'Price difference for value - receipt refused',
    'توزيع البضائع مختلفة في القيمة': 'Different goods distribution in value',
    # Additional common Arabic rejection reasons
    'العميل لا يريد المنتج': 'Customer does not want the product',
    'العميل غير متاح لاستلام الطلب': 'Customer not available to receive order',
    'رفض الاستلام': 'Receipt refused',
    'المنتج لم يجمع على المستلم': 'Product did not meet customer expectations',
    'رفض الاستلام بسبب متطلبات مختلفة': 'Receipt refused due to different requirements'
}

# Main title
st.title("📦 Logistics Deliveries Analytics Dashboard")
st.markdown("---")

def translate_arabic_reason(reason):
    """Translate Arabic rejection reasons to English"""
    if pd.isna(reason) or reason == '':
        return reason
    
    # Clean the text
    cleaned_reason = str(reason).strip()
    
    # Return translation if available, otherwise return original
    return ARABIC_TRANSLATIONS.get(cleaned_reason, cleaned_reason)

# Helper functions
@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Convert date columns to datetime
        date_columns = ['Create Time', 'Last Update', 'Delivery Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean and process data
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        df['No of Pieces'] = pd.to_numeric(df['No of Pieces'], errors='coerce')
        df['Actual Delivered QTY'] = pd.to_numeric(df['Actual Delivered QTY'], errors='coerce')
        df['Rejected QTY'] = pd.to_numeric(df['Rejected QTY'], errors='coerce')
        
        # Fill NaN values
        df['Actual Delivered QTY'] = df['Actual Delivered QTY'].fillna(0)
        df['Rejected QTY'] = df['Rejected QTY'].fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_address_data(address_file):
    """Load and process the address book CSV file"""
    try:
        df = pd.read_csv(address_file)
        
        # Clean and standardize the address data
        if 'code' in df.columns:
            df['code'] = df['code'].astype(str).str.strip()
        
        return df
    except Exception as e:
        st.error(f"Error loading address data: {str(e)}")
        return None

def merge_with_address_data(deliveries_df, address_df):
    """Merge deliveries data with address book data"""
    try:
        if address_df is not None and not address_df.empty:
            # Try to match Receiver Code with code column
            merged_df = deliveries_df.merge(
                address_df[['code', 'name', 'location.address', 'location.city', 'location.region']],
                left_on='Receiver Code',
                right_on='code',
                how='left'
            )
            
            # Add customer information columns
            merged_df['Customer_Name'] = merged_df['name']
            merged_df['Customer_Address'] = merged_df['location.address']
            merged_df['Customer_Region'] = merged_df['location.region']
            
            # Update city information if available from address book
            merged_df['Enhanced_City'] = merged_df['location.city'].fillna(merged_df['Delivery City'])
            
            return merged_df
        else:
            return deliveries_df
    except Exception as e:
        st.warning(f"Could not merge address data: {str(e)}")
        return deliveries_df

def calculate_rfm_analysis(df):
    """Calculate RFM analysis for customers"""
    try:
        # Filter for delivered orders with delivery dates
        delivered_df = df[
            (df['Current Status'].isin(['DELIVERED', 'PARTIALLY_DELIVERED'])) &
            (df['Delivery Date'].notna())
        ].copy()
        
        if delivered_df.empty:
            return pd.DataFrame()
        
        # Calculate RFM metrics by Receiver Code
        today = datetime.now()
        
        rfm_data = []
        for receiver in delivered_df['Receiver Code'].unique():
            customer_data = delivered_df[delivered_df['Receiver Code'] == receiver]
            
            # Recency: Days since last delivery
            last_delivery = customer_data['Delivery Date'].max()
            recency = (today - last_delivery).days
            
            # Frequency: Number of unique waybills (orders)
            frequency = customer_data['Waybill'].nunique()
            
            # Monetary: Total weight delivered (using weight as monetary proxy)
            monetary = customer_data['Weight'].sum()
            
            # Get enhanced customer information if available
            customer_name = customer_data['Customer_Name'].iloc[0] if 'Customer_Name' in customer_data.columns else receiver
            customer_city = customer_data['Enhanced_City'].iloc[0] if 'Enhanced_City' in customer_data.columns else customer_data['Delivery City'].iloc[0]
            customer_region = customer_data['Customer_Region'].iloc[0] if 'Customer_Region' in customer_data.columns else 'Unknown'
            
            rfm_data.append({
                'Receiver Code': receiver,
                'Customer Name': customer_name,
                'City': customer_city,
                'Region': customer_region,
                'Business Type': customer_data['Business Type'].iloc[0],
                'Recency': recency,
                'Frequency': frequency,
                'Monetary': monetary,
                'Last Delivery': last_delivery,
                'Total Pieces': customer_data['Actual Delivered QTY'].sum(),
                'Return Rate': (customer_data['Rejected QTY'].sum() / customer_data['No of Pieces'].sum() * 100) if customer_data['No of Pieces'].sum() > 0 else 0
            })
        
        rfm_df = pd.DataFrame(rfm_data)
        
        # Calculate RFM scores (1-5 scale)
        rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        rfm_df['R_Score'] = rfm_df['R_Score'].astype(int)
        rfm_df['F_Score'] = rfm_df['F_Score'].astype(int)
        rfm_df['M_Score'] = rfm_df['M_Score'].astype(int)
        
        # Calculate RFM Score
        rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
        
        # Segment customers
        def segment_customers(row):
            if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
                return 'Champions'
            elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
                return 'Loyal Customers'
            elif row['R_Score'] >= 3 and row['F_Score'] >= 2:
                return 'Potential Loyalists'
            elif row['R_Score'] >= 4:
                return 'New Customers'
            elif row['R_Score'] >= 2 and row['F_Score'] >= 2:
                return 'At Risk'
            elif row['F_Score'] >= 2:
                return 'Cannot Lose Them'
            else:
                return 'Lost Customers'
        
        rfm_df['Segment'] = rfm_df.apply(segment_customers, axis=1)
        
        return rfm_df
        
    except Exception as e:
        st.error(f"Error in RFM analysis: {str(e)}")
        return pd.DataFrame()

def create_summary_cards(df):
    """Create enhanced summary metric cards"""
    # Calculate key metrics
    total_orders = df['Waybill'].nunique()
    delivered_orders = df[df['Current Status'].isin(['DELIVERED', 'PARTIALLY_DELIVERED'])]['Waybill'].nunique()
    delivery_rate = (delivered_orders / total_orders * 100) if total_orders > 0 else 0
    
    total_weight = df['Weight'].sum()
    total_returns = df['Rejected QTY'].sum()
    total_pieces = df['No of Pieces'].sum()
    return_rate = (total_returns / total_pieces * 100) if total_pieces > 0 else 0
    
    # Create metrics with better context
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Orders", 
            f"{total_orders:,}",
            help="Total unique waybills/orders in the data"
        )
    
    with col2:
        color = "normal"
        if delivery_rate >= 95:
            color = "normal"
        elif delivery_rate < 85:
            color = "inverse"
        
        st.metric(
            "Delivery Success", 
            f"{delivery_rate:.1f}%",
            delta=f"{delivered_orders:,} orders",
            help="Percentage of orders successfully delivered"
        )
    
    with col3:
        st.metric(
            "Total Weight", 
            f"{total_weight:,.0f} kg" if total_weight >= 1000 else f"{total_weight:.1f} kg",
            help="Total weight of all shipments"
        )
    
    with col4:
        return_color = "normal" if return_rate <= 5 else "inverse"
        st.metric(
            "Return Rate", 
            f"{return_rate:.1f}%",
            delta=f"{total_returns:,.0f} items",
            delta_color=return_color,
            help="Percentage of items returned/rejected"
        )
    
    with col5:
        cities_served = df['Delivery City'].nunique()
        st.metric(
            "Cities Served", 
            f"{cities_served:,}",
            help="Number of unique delivery cities"
        )

def create_filters(df):
    """Create interactive filters"""
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    if 'Create Time' in df.columns and df['Create Time'].notna().any():
        min_date = df['Create Time'].min().date()
        max_date = df['Create Time'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        elif hasattr(date_range, '__len__') and len(date_range) >= 1:
            start_date = end_date = date_range[0] if len(date_range) > 0 else min_date
        else:
            start_date = end_date = min_date
    else:
        start_date = end_date = None
    
    # City filter
    cities = ['All'] + sorted(df['Delivery City'].dropna().unique().tolist())
    selected_cities = st.sidebar.multiselect(
        "Select Cities",
        cities,
        default=['All']
    )
    
    # Status filter
    statuses = ['All'] + sorted(df['Current Status'].dropna().unique().tolist())
    selected_statuses = st.sidebar.multiselect(
        "Select Status",
        statuses,
        default=['All']
    )
    
    # Business type filter
    business_types = ['All'] + sorted(df['Business Type'].dropna().unique().tolist())
    selected_business_types = st.sidebar.multiselect(
        "Select Business Type",
        business_types,
        default=['All']
    )
    
    return {
        'date_range': (start_date, end_date),
        'cities': selected_cities,
        'statuses': selected_statuses,
        'business_types': selected_business_types
    }

def apply_filters(df, filters):
    """Apply selected filters to the dataframe"""
    filtered_df = df.copy()
    
    # Apply date filter
    if filters['date_range'][0] and filters['date_range'][1] and 'Create Time' in df.columns:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['Create Time'].dt.date >= start_date) &
            (filtered_df['Create Time'].dt.date <= end_date)
        ]
    
    # Apply city filter
    if 'All' not in filters['cities']:
        filtered_df = filtered_df[filtered_df['Delivery City'].isin(filters['cities'])]
    
    # Apply status filter
    if 'All' not in filters['statuses']:
        filtered_df = filtered_df[filtered_df['Current Status'].isin(filters['statuses'])]
    
    # Apply business type filter
    if 'All' not in filters['business_types']:
        filtered_df = filtered_df[filtered_df['Business Type'].isin(filters['business_types'])]
    
    return filtered_df

# File upload
st.subheader("📁 Upload Your Data Files")

col1, col2 = st.columns(2)

with col1:
    st.write("**Deliveries Data**")
    uploaded_file = st.file_uploader(
        "Choose deliveries CSV file",
        type="csv",
        help="Upload your logistics deliveries CSV file",
        key="deliveries"
    )

with col2:
    st.write("**Address Book Data (Optional)**")
    address_file = st.file_uploader(
        "Choose address book CSV file",
        type="csv", 
        help="Upload customer address book for enhanced analysis",
        key="addresses"
    )

if uploaded_file is not None:
    # Load data
    df = load_and_process_data(uploaded_file)
    
    # Load address data if provided
    address_df = None
    if address_file is not None:
        address_df = load_address_data(address_file)
        if address_df is not None:
            st.success(f"Address book loaded: {len(address_df)} records")
    
    if df is not None:
        # Merge with address data if available
        if address_df is not None:
            df = merge_with_address_data(df, address_df)
            st.info("Deliveries data enhanced with customer information")
        # Data preview section
        st.subheader("📋 Data Overview")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.write("**Sample Data (First 5 rows):**")
            # Show only key columns for preview
            key_columns = ['Waybill', 'Receiver Code', 'Delivery City', 'Current Status', 'Weight', 'Delivery Date']
            preview_columns = [col for col in key_columns if col in df.columns]
            st.dataframe(df[preview_columns].head(5), use_container_width=True)
        
        with col2:
            st.write("**Data Summary:**")
            
            # Create clean summary metrics
            total_records = len(df)
            date_range = ""
            if 'Create Time' in df.columns and df['Create Time'].notna().any():
                min_date = df['Create Time'].min().strftime('%Y-%m-%d')
                max_date = df['Create Time'].max().strftime('%Y-%m-%d')
                date_range = f"{min_date} to {max_date}"
            
            # Display key metrics in a clean format
            metrics_data = {
                "Total Records": f"{total_records:,}",
                "Unique Orders": f"{df['Waybill'].nunique():,}",
                "Unique Customers": f"{df['Receiver Code'].nunique():,}",
                "Cities": f"{df['Delivery City'].nunique()}",
                "Date Range": date_range if date_range else "Not available"
            }
            
            for metric, value in metrics_data.items():
                if value != "Not available":
                    st.metric(metric.replace("_", " "), value)
                elif metric == "Date Range":
                    st.text(f"{metric}: {value}")
        
        st.markdown("---")
        
        # Filters
        filters = create_filters(df)
        filtered_df = apply_filters(df, filters)
        
        # Summary cards
        st.subheader("📊 Performance Dashboard")
        create_summary_cards(filtered_df)
        
        # Quick status overview
        st.markdown("### 📈 Quick Status Overview")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            delivered_count = len(filtered_df[filtered_df['Current Status'] == 'DELIVERED'])
            st.info(f"✅ **Delivered**: {delivered_count:,} orders")
        
        with status_col2:
            planned_count = len(filtered_df[filtered_df['Current Status'] == 'PLANNED'])
            st.warning(f"📋 **Planned**: {planned_count:,} orders")
        
        with status_col3:
            partial_count = len(filtered_df[filtered_df['Current Status'] == 'PARTIALLY_DELIVERED'])
            st.error(f"🔄 **Partial**: {partial_count:,} orders") if partial_count > 0 else st.success("🔄 **Partial**: 0 orders")
        
        st.markdown("---")
        
        # Orders by City
        st.subheader("🏙️ Orders by City")
        city_orders = filtered_df['Delivery City'].value_counts().head(15)
        fig_city = px.bar(
            x=city_orders.values,
            y=city_orders.index,
            orientation='h',
            title="Top 15 Cities by Order Count",
            labels={'x': 'Number of Orders', 'y': 'City'},
            color=city_orders.values,
            color_continuous_scale='Blues'
        )
        fig_city.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_city, use_container_width=True)
        
        # Sales trends over time
        st.subheader("📈 Sales Trends Over Time")
        if 'Create Time' in filtered_df.columns and filtered_df['Create Time'].notna().any():
            daily_orders = filtered_df.groupby(filtered_df['Create Time'].dt.date).size().reset_index()
            daily_orders.columns = ['Date', 'Orders']
            
            fig_trend = px.line(
                daily_orders,
                x='Date',
                y='Orders',
                title="Daily Order Trends",
                labels={'Orders': 'Number of Orders'}
            )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Date information not available for trend analysis")
        
        # Order Status Breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Order Status Breakdown")
            status_counts = filtered_df['Current Status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Distribution of Order Status"
            )
            fig_status.update_layout(height=400)
            st.plotly_chart(fig_status, use_container_width=True)
        
        with col2:
            st.subheader("🏢 Business Type Distribution")
            business_counts = filtered_df['Business Type'].value_counts()
            fig_business = px.pie(
                values=business_counts.values,
                names=business_counts.index,
                title="Distribution of Business Types"
            )
            fig_business.update_layout(height=400)
            st.plotly_chart(fig_business, use_container_width=True)
        
        # Top performing products and customers
        st.subheader("🏆 Top Performers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Products by Volume**")
            top_products = filtered_df.groupby('SKU').agg({
                'Weight': 'sum',
                'No of Pieces': 'sum',
                'Waybill': 'nunique'
            }).sort_values('Weight', ascending=False).head(10)
            top_products.columns = ['Total Weight', 'Total Pieces', 'Orders']
            st.dataframe(top_products, use_container_width=True)
        
        with col2:
            st.write("**Top 10 Customers by Volume**")
            top_customers = filtered_df.groupby('Receiver Code').agg({
                'Weight': 'sum',
                'Waybill': 'nunique',
                'Delivery City': 'first'
            }).sort_values('Weight', ascending=False).head(10)
            top_customers.columns = ['Total Weight', 'Orders', 'City']
            st.dataframe(top_customers, use_container_width=True)
        
        # Returns Analysis
        st.subheader("↩️ Returns Analysis")
        
        returns_df = filtered_df[filtered_df['Rejected QTY'] > 0].copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_items = filtered_df['No of Pieces'].sum()
            total_returns = filtered_df['Rejected QTY'].sum()
            return_rate = (total_returns / total_items * 100) if total_items > 0 else 0
            st.metric("Return Rate", f"{return_rate:.2f}%")
        
        with col2:
            total_return_orders = len(returns_df['Waybill'].unique()) if not returns_df.empty else 0
            st.metric("Orders with Returns", f"{total_return_orders:,}")
        
        with col3:
            avg_return_qty = returns_df['Rejected QTY'].mean() if not returns_df.empty else 0
            st.metric("Avg Return Quantity", f"{avg_return_qty:.2f}")
        
        if not returns_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Return Reasons**")
                if 'Rejected Reason' in returns_df.columns:
                    # Apply translation to reasons
                    returns_df_translated = returns_df.copy()
                    returns_df_translated['Rejected Reason Translated'] = returns_df_translated['Rejected Reason'].apply(translate_arabic_reason)
                    
                    # Check if any translations were applied
                    has_arabic = any(reason in ARABIC_TRANSLATIONS for reason in returns_df_translated['Rejected Reason'].dropna())
                    if has_arabic:
                        st.info("🌐 Arabic rejection reasons have been automatically translated to English")
                    
                    return_reasons = returns_df_translated['Rejected Reason Translated'].value_counts().head(10)
                    return_reasons_df = return_reasons.to_frame('Count')
                    return_reasons_df.index.name = 'Return Reason'
                    st.dataframe(return_reasons_df, use_container_width=True)
                else:
                    st.info("Return reason data not available")
            
            with col2:
                st.write("**Returns by City**")
                returns_by_city = returns_df.groupby('Delivery City')['Rejected QTY'].sum().sort_values(ascending=False).head(10)
                st.dataframe(returns_by_city.to_frame('Total Returns'), use_container_width=True)
        
        # RFM Analysis
        st.subheader("🎯 RFM Customer Analysis")
        
        rfm_df = calculate_rfm_analysis(filtered_df)
        
        if not rfm_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Customer Segments Distribution**")
                segment_counts = rfm_df['Segment'].value_counts()
                fig_segments = px.bar(
                    x=segment_counts.index,
                    y=segment_counts.values,
                    title="Customer Segments",
                    color=segment_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_segments.update_layout(height=400, showlegend=False, xaxis_tickangle=45)
                st.plotly_chart(fig_segments, use_container_width=True)
            
            with col2:
                st.write("**RFM Scatter Plot**")
                fig_rfm = px.scatter(
                    rfm_df,
                    x='Frequency',
                    y='Monetary',
                    size='Recency',
                    color='Segment',
                    hover_data=['Receiver Code', 'City'],
                    title="RFM Analysis - Frequency vs Monetary Value"
                )
                fig_rfm.update_layout(height=400)
                st.plotly_chart(fig_rfm, use_container_width=True)
            
            # RFM table
            st.write("**Top 20 Customers by RFM Score**")
            rfm_sorted = rfm_df.sort_values(['R_Score', 'F_Score', 'M_Score'], ascending=False).head(20)
            
            if 'Customer Name' in rfm_df.columns:
                rfm_display_cols = ['Customer Name', 'Receiver Code', 'City', 'Segment', 'RFM_Score', 'Frequency', 'Monetary', 'Recency', 'Return Rate']
            else:
                rfm_display_cols = ['Receiver Code', 'City', 'Business Type', 'Segment', 'RFM_Score', 'Frequency', 'Monetary', 'Recency']
            
            # Filter columns that actually exist in the dataframe
            available_cols = [col for col in rfm_display_cols if col in rfm_sorted.columns]
            st.dataframe(rfm_sorted[available_cols], use_container_width=True)
        else:
            st.info("Unable to perform RFM analysis. Please ensure your data contains delivered orders with delivery dates.")
        
        # Actionable Insights
        st.subheader("💡 Actionable Insights & Recommendations")
        
        insights = []
        
        # Delivery performance insights
        delivered_rate = (filtered_df['Current Status'].isin(['DELIVERED', 'PARTIALLY_DELIVERED']).sum() / len(filtered_df)) * 100
        if delivered_rate < 85:
            insights.append(f"⚠️ **Delivery Performance**: Current delivery rate is {delivered_rate:.1f}%. Consider investigating delivery bottlenecks.")
        
        # City performance insights
        city_performance = filtered_df.groupby('Delivery City').agg({
            'Current Status': lambda x: (x.isin(['DELIVERED', 'PARTIALLY_DELIVERED']).sum() / len(x)) * 100
        }).sort_values('Current Status')
        
        if not city_performance.empty:
            worst_city = city_performance.index[0]
            worst_rate = city_performance.iloc[0, 0]
            if worst_rate < 80:
                insights.append(f"🏙️ **City Alert**: {worst_city} has the lowest delivery success rate at {worst_rate:.1f}%. Focus on improving logistics in this area.")
        
        # Return rate insights
        if return_rate > 5:
            insights.append(f"↩️ **High Returns**: Return rate of {return_rate:.1f}% is concerning. Investigate quality issues or customer expectations.")
        
        # Product insights
        if not filtered_df.empty:
            product_performance = filtered_df.groupby('SKU').agg({
                'Rejected QTY': 'sum',
                'No of Pieces': 'sum'
            })
            product_performance['Return Rate'] = (product_performance['Rejected QTY'] / product_performance['No of Pieces']) * 100
            high_return_products = product_performance[product_performance['Return Rate'] > 10].sort_values('Return Rate', ascending=False)
            
            if not high_return_products.empty:
                worst_product = high_return_products.index[0]
                worst_product_rate = high_return_products.iloc[0]['Return Rate']
                insights.append(f"📦 **Product Alert**: SKU {worst_product} has a return rate of {worst_product_rate:.1f}%. Review product quality or specifications.")
        
        # RFM insights
        if not rfm_df.empty:
            at_risk_customers = len(rfm_df[rfm_df['Segment'] == 'At Risk'])
            lost_customers = len(rfm_df[rfm_df['Segment'] == 'Lost Customers'])
            
            if at_risk_customers > 0:
                insights.append(f"🎯 **Customer Retention**: {at_risk_customers} customers are at risk. Implement retention campaigns.")
            
            if lost_customers > 0:
                insights.append(f"💔 **Lost Customers**: {lost_customers} customers appear to be lost. Consider win-back campaigns.")
        
        # Return reason insights with translations
        if not returns_df.empty and 'Rejected Reason' in returns_df.columns:
            returns_translated = returns_df.copy()
            returns_translated['Rejected Reason Translated'] = returns_translated['Rejected Reason'].apply(translate_arabic_reason)
            top_return_reason = returns_translated['Rejected Reason Translated'].value_counts().head(1)
            
            if not top_return_reason.empty:
                reason = top_return_reason.index[0]
                count = top_return_reason.iloc[0]
                
                # Provide specific recommendations based on top return reason
                if 'product' in reason.lower() or 'quality' in reason.lower():
                    insights.append(f"🔍 **Product Quality Focus**: Top return reason is '{reason}' ({count} cases). Review product quality control processes.")
                elif 'quantity' in reason.lower() or 'wrong' in reason.lower():
                    insights.append(f"📊 **Inventory Accuracy**: Top return reason is '{reason}' ({count} cases). Improve order picking and inventory management.")
                elif 'unavailable' in reason.lower() or 'customer' in reason.lower():
                    insights.append(f"📞 **Customer Communication**: Top return reason is '{reason}' ({count} cases). Enhance delivery scheduling and customer notifications.")
                elif 'delay' in reason.lower():
                    insights.append(f"⏰ **Delivery Speed**: Top return reason is '{reason}' ({count} cases). Optimize delivery routes and scheduling.")
                else:
                    insights.append(f"🎯 **Return Analysis**: Top return reason is '{reason}' ({count} cases). Address this specific issue to reduce returns.")
        
        # Business type insights
        business_performance = filtered_df.groupby('Business Type').agg({
            'Current Status': lambda x: (x.isin(['DELIVERED', 'PARTIALLY_DELIVERED']).sum() / len(x)) * 100,
            'Weight': 'mean'
        })
        
        if not business_performance.empty:
            best_business = business_performance['Current Status'].idxmax()
            best_rate = business_performance['Current Status'].max()
            insights.append(f"🏢 **Best Performing Segment**: {best_business} has the highest delivery success rate at {best_rate:.1f}%.")
        
        if not insights:
            insights.append("✅ **Good Performance**: Your logistics operations are performing well across all key metrics!")
        
        for insight in insights:
            st.markdown(insight)
        
        # Export functionality
        st.subheader("📥 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Filtered Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"filtered_logistics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if not rfm_df.empty and st.button("Export RFM Analysis"):
                rfm_csv = rfm_df.to_csv(index=False)
                st.download_button(
                    label="Download RFM CSV",
                    data=rfm_csv,
                    file_name=f"rfm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("Export Insights"):
                insights_text = "\n".join(insights)
                st.download_button(
                    label="Download Insights",
                    data=insights_text,
                    file_name=f"logistics_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

else:
    st.info("👆 Please upload a CSV file to begin the analysis")
    
    # Sample data format information
    st.subheader("📋 Expected Data Format")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Deliveries CSV (Required):**")
        expected_columns = {
            'Waybill': 'Unique order identifier',
            'Create Time': 'Order creation timestamp',
            'Receiver Code': 'Customer identifier',
            'Delivery City': 'Destination city',
            'Current Status': 'Order status (DELIVERED, PLANNED, etc.)',
            'SKU': 'Product identifier',
            'Weight': 'Package weight',
            'No of Pieces': 'Number of items',
            'Actual Delivered QTY': 'Quantity delivered',
            'Delivery Date': 'Actual delivery date',
            'Rejected QTY': 'Quantity rejected/returned',
            'Business Type': 'Customer segment'
        }
        
        for col, desc in expected_columns.items():
            st.write(f"• **{col}**: {desc}")
    
    with col2:
        st.write("**Address Book CSV (Optional):**")
        address_columns = {
            'code': 'Customer code (matches Receiver Code)',
            'name': 'Customer name',
            'location.city': 'Customer city',
            'location.address': 'Customer address',
            'location.region': 'Customer region',
            'type': 'Customer type'
        }
        
        for col, desc in address_columns.items():
            st.write(f"• **{col}**: {desc}")
        
        st.info("💡 The address book enhances RFM analysis by providing customer names and detailed location information.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>📦 Logistics Deliveries Analytics Dashboard | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
