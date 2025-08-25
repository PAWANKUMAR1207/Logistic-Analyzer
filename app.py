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
    'العميل لا يريد المنتج': 'Customer does not want the product',
    'العميل غير متاح لاستلام الطلب': 'Customer not available to receive order',
    'رفض الاستلام': 'Receipt refused',
    'المنتج لم يجمع على المستلم': 'Product did not meet customer expectations',
    'رفض الاستلام بسبب متطلبات مختلفة': 'Receipt refused due to different requirements'
}

# Dynamic title based on filters
def get_dynamic_title(filters=None):
    base_title = "📦 Logistics Deliveries Analytics Dashboard"
    
    if filters is None:
        return base_title
    
    filter_parts = []
    
    # Add city filter info
    if filters.get('cities') and 'All' not in filters['cities']:
        if len(filters['cities']) == 1:
            filter_parts.append(f"📍 {filters['cities'][0]}")
        else:
            filter_parts.append(f"📍 {len(filters['cities'])} Cities")
    
    # Add business type filter info
    if filters.get('business_types') and 'All' not in filters['business_types']:
        if len(filters['business_types']) == 1:
            filter_parts.append(f"🏢 {filters['business_types'][0]}")
        else:
            filter_parts.append(f"🏢 {len(filters['business_types'])} Business Types")
    
    # Add status filter info
    if filters.get('statuses') and 'All' not in filters['statuses']:
        if len(filters['statuses']) == 1:
            filter_parts.append(f"📋 {filters['statuses'][0]}")
        else:
            filter_parts.append(f"📋 {len(filters['statuses'])} Statuses")
    
    if filter_parts:
        return f"{base_title} - {' | '.join(filter_parts)}"
    else:
        return base_title

# Initial title
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
        
        # **NEW: Standardize city names to fix case sensitivity issues**
        if 'Delivery City' in df.columns:
            df['Delivery City'] = df['Delivery City'].str.title()
            # Fix specific known variations
            city_mappings = {
                'Riyadh': 'Riyadh',
                'Jeddah': 'Jeddah', 
                'Dammam': 'Dammam',
                'Makkah': 'Makkah',
                'Madinah': 'Madinah',
                'Khamis Mushait': 'Khamis Mushait',
                'Hasa Industrial City': 'Hasa Industrial City'
            }
            df['Delivery City'] = df['Delivery City'].replace(city_mappings)
        
        # **NEW: Calculate delivery time in days**
        if 'Create Time' in df.columns and 'Delivery Date' in df.columns:
            df['Delivery_Days'] = (df['Delivery Date'] - df['Create Time']).dt.days
        
        # **NEW: Calculate fulfillment rate**
        df['Fulfillment_Rate'] = np.where(
            df['No of Pieces'] > 0,
            (df['Actual Delivered QTY'] / df['No of Pieces']) * 100,
            0
        )
        
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
        
        # Create RFM scores (1-5 scale, 5 being best) with duplicate handling
        try:
            rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1], duplicates='drop')
        except ValueError:
            # If still issues with duplicates, use simple ranking
            rfm_df['R_Score'] = pd.cut(rfm_df['Recency'], bins=5, labels=[5,4,3,2,1], duplicates='drop')
            
        try:
            rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        except ValueError:
            rfm_df['F_Score'] = pd.cut(rfm_df['Frequency'], bins=5, labels=[1,2,3,4,5], duplicates='drop')
            
        try:
            rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        except ValueError:
            rfm_df['M_Score'] = pd.cut(rfm_df['Monetary'], bins=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Create RFM segment with better business names
        def segment_customers(row):
            if row['R_Score'] >= 4 and row['F_Score'] >= 4:
                return 'VIP Customers'  # High frequency, recent orders
            elif row['R_Score'] >= 3 and row['F_Score'] >= 3:
                return 'Regular Customers'  # Consistent, reliable customers
            elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
                return 'Recent New Customers'  # New but not frequent yet
            elif row['F_Score'] >= 4 and row['R_Score'] <= 2:
                return 'Inactive Frequent Customers'  # Used to order a lot, now quiet
            elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
                return 'Lost Customers'  # Haven't ordered recently or frequently
            elif row['R_Score'] >= 3 and row['F_Score'] <= 2:
                return 'Occasional Customers'  # Order recently but not often
            else:
                return 'Developing Customers'  # Customers with potential to grow
        
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
    
    # **FIXED: Return rate calculation - use rejected items vs delivered items, not total pieces**
    total_returns = df['Rejected QTY'].sum()
    total_delivered_items = df['Actual Delivered QTY'].sum()
    # Calculate return rate as: rejected items / (delivered + rejected) items
    total_processed_items = total_delivered_items + total_returns
    return_rate = (total_returns / total_processed_items * 100) if total_processed_items > 0 else 0
    
    # Create metrics with better context
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Orders", 
            f"{total_orders:,}",
            help="Total unique waybills/orders in the data"
        )
    
    with col2:
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
    """Create sidebar filters"""
    st.sidebar.header("🔍 Filter Options")
    
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
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        elif len(date_range) == 1:
            start_date = end_date = date_range[0]
        else:
            start_date = end_date = None
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
        "Select Delivery Status",
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
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    # Apply date filter
    if filters['date_range'][0] and filters['date_range'][1] and 'Create Time' in df.columns:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['Create Time'].dt.date >= start_date) &
            (filtered_df['Create Time'].dt.date <= end_date)
        ]
    
    # Apply city filter (case-insensitive)
    if 'All' not in filters['cities']:
        # Create a case-insensitive filter, handling NaN values
        mask = filtered_df['Delivery City'].notna() & filtered_df['Delivery City'].str.upper().isin([city.upper() for city in filters['cities']])
        filtered_df = filtered_df[mask]
    
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
        
        # Check if filtered data is empty
        if filtered_df.empty:
            st.error("🚫 **No data matches your current filters**")
            st.info("Try adjusting your filter settings or uploading data with the selected criteria.")
            st.stop()
        
        # Update title with filter information and show active filters
        dynamic_title = get_dynamic_title(filters)
        
        # Show active filters in a nice container
        active_filters = []
        if filters.get('cities') and 'All' not in filters['cities']:
            active_filters.append(f"Cities: {', '.join(filters['cities'][:3])}{'...' if len(filters['cities']) > 3 else ''}")
        if filters.get('business_types') and 'All' not in filters['business_types']:
            active_filters.append(f"Business: {', '.join(filters['business_types'][:2])}{'...' if len(filters['business_types']) > 2 else ''}")
        if filters.get('statuses') and 'All' not in filters['statuses']:
            active_filters.append(f"Status: {', '.join(filters['statuses'][:2])}{'...' if len(filters['statuses']) > 2 else ''}")
        if filters.get('date_range') and filters['date_range'][0] and filters['date_range'][1]:
            start_date, end_date = filters['date_range']
            active_filters.append(f"Date: {start_date} to {end_date}")
            
        if active_filters:
            st.markdown("---")
            st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{dynamic_title}</h1>", unsafe_allow_html=True)
            with st.container():
                st.markdown("""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 10px 0;'>
                    <h4 style='margin: 0; color: #0066cc;'>🔍 Active Filters</h4>
                    <p style='margin: 5px 0; color: #333;'>{}</p>
                </div>
                """.format(" | ".join(active_filters)), unsafe_allow_html=True)
            st.markdown("---")
        
        # **NEW: Enhanced Performance Dashboard with Direction Analysis**
        st.subheader("📊 Performance Dashboard")
        create_summary_cards(filtered_df)
        
        # **FIXED: Direction Analysis with Clear Explanations**
        if 'Direction' in filtered_df.columns:
            st.markdown("### 🔄 Forward vs Reverse Orders Analysis")
            
            # Show clear explanation of what these mean
            with st.expander("📖 What do Forward and Reverse orders mean?"):
                st.markdown("""
                **🚛 FORWARD Orders**: Regular deliveries TO customers
                - Products being delivered from warehouse to customer
                - Normal outbound logistics operations
                - Status examples: DELIVERED, PLANNED, IN_ROUTE
                
                **🔄 REVERSE Orders**: Returns/pickups FROM customers  
                - Products being returned by customers
                - Pickup operations from customer locations
                - Usually have UNASSIGNED status (waiting for pickup)
                - No delivery dates (because they're pickups, not deliveries)
                """)
            
            direction_col1, direction_col2, direction_col3 = st.columns(3)
            
            forward_count = len(filtered_df[filtered_df['Direction'] == 'FORWARD'])
            reverse_count = len(filtered_df[filtered_df['Direction'] == 'REVERSE'])
            total_orders = len(filtered_df)
            
            with direction_col1:
                forward_pct = (forward_count/total_orders*100) if total_orders > 0 else 0
                st.metric(
                    "📦 Forward Orders", 
                    f"{forward_count:,}",
                    delta=f"{forward_pct:.1f}% of filtered data",
                    help="Regular deliveries TO customers"
                )
            
            with direction_col2:
                reverse_pct = (reverse_count/total_orders*100) if total_orders > 0 else 0
                st.metric(
                    "🔄 Reverse Orders", 
                    f"{reverse_count:,}",
                    delta=f"{reverse_pct:.1f}% of filtered data",
                    delta_color="inverse" if reverse_pct > 15 else "normal",
                    help="Returns/pickups FROM customers"
                )
            
            with direction_col3:
                reverse_rate = (reverse_count / total_orders * 100) if total_orders > 0 else 0
                rate_status = "normal" if reverse_rate <= 10 else "inverse"
                st.metric(
                    "🔄 Return Rate", 
                    f"{reverse_rate:.1f}%",
                    delta="Normal" if reverse_rate <= 10 else "High - investigate!",
                    delta_color=rate_status,
                    help="Percentage of orders that are returns (normal: <10%)"
                )
            
            # Add warning if return rate seems too high
            if reverse_rate > 30:
                st.warning(f"⚠️ **High Return Rate Alert**: {reverse_rate:.1f}% return rate may indicate filter settings or data issues. Check your active filters above.")
        
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
            if partial_count > 0:
                st.error(f"🔄 **Partial**: {partial_count:,} orders")
            else:
                st.success("🔄 **Partial**: 0 orders")
        
        st.markdown("---")
        
        # Orders by City (only show if multiple cities or no city filter active)
        if 'All' in filters.get('cities', ['All']) or len(filters.get('cities', [])) > 1:
            st.subheader("🏙️ Orders by City")
            city_orders = filtered_df['Delivery City'].value_counts().head(15)
            
            if not city_orders.empty and len(city_orders) > 1:
                city_df = pd.DataFrame({
                    'City': city_orders.index,
                    'Orders': city_orders.values
                })
                
                fig_city = px.bar(
                    city_df,
                    x='Orders',
                    y='City',
                    orientation='h',
                    title="Top 15 Cities by Order Count",
                    labels={'Orders': 'Number of Orders', 'City': 'City'},
                    color='Orders',
                    color_continuous_scale='Blues'
                )
                fig_city.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_city, use_container_width=True)
            else:
                st.info("Only one city in current selection")
        
        # **NEW: Enhanced Business Type Analysis**
        st.subheader("🏢 Business Type Performance")
        if 'Business Type' in filtered_df.columns:
            business_analysis = filtered_df.groupby('Business Type').agg({
                'Waybill': 'nunique',
                'Weight': 'sum',
                'Current Status': lambda x: (x.isin(['DELIVERED', 'PARTIALLY_DELIVERED'])).sum(),
                'Rejected QTY': 'sum'
            }).round(2)
            business_analysis.columns = ['Orders', 'Total Weight (kg)', 'Delivered', 'Rejected Qty']
            business_analysis['Success Rate %'] = (business_analysis['Delivered'] / business_analysis['Orders'] * 100).round(1)
            business_analysis = business_analysis.sort_values('Orders', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Business Type Summary**")
                st.dataframe(business_analysis, use_container_width=True)
            
            with col2:
                # Business type pie chart
                fig_business = px.pie(
                    values=business_analysis['Orders'],
                    names=business_analysis.index,
                    title="Orders by Business Type"
                )
                st.plotly_chart(fig_business, use_container_width=True)
        
        # Sales trends over time
        st.subheader("📈 Sales Trends Over Time")
        if 'Create Time' in filtered_df.columns and filtered_df['Create Time'].notna().any():
            daily_orders = filtered_df.groupby(filtered_df['Create Time'].dt.date).size().reset_index()
            daily_orders.columns = ['Date', 'Orders']
            
            # **NEW: Add business type breakdown in trends**
            if 'Business Type' in filtered_df.columns:
                daily_business = filtered_df.groupby([filtered_df['Create Time'].dt.date, 'Business Type']).size().reset_index()
                daily_business.columns = ['Date', 'Business Type', 'Orders']
                
                fig_trend = px.line(
                    daily_business, 
                    x='Date', 
                    y='Orders',
                    color='Business Type',
                    title='Daily Order Trends by Business Type',
                    labels={'Orders': 'Number of Orders', 'Date': 'Date'}
                )
            else:
                fig_trend = px.line(
                    daily_orders, 
                    x='Date', 
                    y='Orders',
                    title='Daily Order Trends',
                    labels={'Orders': 'Number of Orders', 'Date': 'Date'}
                )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Date information not available for trend analysis")
        
        # **NEW: Enhanced Delivery Performance Analysis**
        st.subheader("📊 Enhanced Delivery Status Analysis")
        status_counts = filtered_df['Current Status'].value_counts()
        
        # Status performance metrics
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            delivered_total = status_counts.get('DELIVERED', 0)
            st.metric("Fully Delivered", f"{delivered_total:,}", help="Successfully completed orders")
        
        with perf_col2:
            partial_total = status_counts.get('PARTIALLY_DELIVERED', 0)
            st.metric("Partial Delivery", f"{partial_total:,}", delta_color="inverse", help="Partially completed orders")
        
        with perf_col3:
            rejected_total = status_counts.get('TOTAL_REJECTION', 0)
            st.metric("Rejected Orders", f"{rejected_total:,}", delta_color="inverse", help="Completely rejected orders")
        
        with perf_col4:
            planned_total = status_counts.get('PLANNED', 0) + status_counts.get('UNASSIGNED', 0)
            st.metric("Pending Orders", f"{planned_total:,}", help="Orders waiting for delivery")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Delivery Status Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart with better formatting
            status_df = pd.DataFrame({
                'Status': status_counts.index,
                'Count': status_counts.values
            })
            
            fig_bar = px.bar(
                status_df,
                x='Count',
                y='Status',
                orientation='h',
                title="Orders by Status (Detailed View)",
                labels={'Count': 'Number of Orders', 'Status': 'Status'},
                color='Count',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # RFM Analysis
        st.subheader("🎯 RFM Customer Analysis")
        
        # Check if we have enough data for RFM analysis
        delivered_orders = len(filtered_df[filtered_df['Current Status'].isin(['DELIVERED', 'PARTIALLY_DELIVERED'])])
        unique_customers = filtered_df['Receiver Code'].nunique()
        
        # Initialize rfm_df to avoid scope issues
        rfm_df = pd.DataFrame()
        
        if delivered_orders < 10:
            st.info(f"📊 **RFM Analysis requires at least 10 delivered orders**. Current delivered orders: {delivered_orders}. Upload more data or adjust filters.")
        elif unique_customers < 5:
            st.info(f"📊 **RFM Analysis requires at least 5 unique customers**. Current unique customers: {unique_customers}. Upload more data or adjust filters.")
        else:
            rfm_df = calculate_rfm_analysis(filtered_df)
            
            if not rfm_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Customer segments
                    segment_counts = rfm_df['Segment'].value_counts()
                    fig_segments = px.pie(
                        values=segment_counts.values,
                        names=segment_counts.index,
                        title="Customer Segments Distribution"
                    )
                    st.plotly_chart(fig_segments, use_container_width=True)
                
                with col2:
                    # Top customers by monetary value
                    top_customers = rfm_df.nlargest(10, 'Monetary')[['Customer Name', 'Monetary', 'Frequency', 'Segment']]
                    st.write("**Top 10 Customers by Total Weight**")
                    st.dataframe(top_customers, use_container_width=True)
                
                # Segment analysis with explanations
                st.write("**Customer Segment Summary**")
                
                # Add segment explanations
                with st.expander("📖 What do these customer segments mean?"):
                    st.markdown("""
                    **🏆 VIP Customers**: Order frequently and recently - your most valuable customers  
                    **👥 Regular Customers**: Consistent buyers who order regularly - reliable revenue source  
                    **🌟 Recent New Customers**: Just started ordering - high potential for growth  
                    **💤 Inactive Frequent Customers**: Used to order a lot but haven't recently - need re-engagement  
                    **🆘 Lost Customers**: Haven't ordered recently or frequently - require win-back campaigns  
                    **🔄 Occasional Customers**: Order recently but infrequently - potential to increase frequency  
                    **📈 Developing Customers**: Show potential for growth with the right nurturing
                    """)
                
                segment_summary = rfm_df.groupby('Segment').agg({
                    'Receiver Code': 'count',
                    'Monetary': 'mean',
                    'Frequency': 'mean',
                    'Recency': 'mean'
                }).round(2)
                segment_summary.columns = ['Customer Count', 'Avg Weight (kg)', 'Avg Orders', 'Days Since Last Order']
                st.dataframe(segment_summary, use_container_width=True)
            else:
                st.info("RFM analysis could not be completed with current data")
        
        # If not enough data, show what's needed
        if delivered_orders < 10 or unique_customers < 5:
            st.markdown("""
            **RFM Analysis Requirements:**
            - Minimum 10 delivered orders
            - Minimum 5 unique customers  
            - Delivery dates must be available
            - Multiple orders per customer recommended for better insights
            """)
        
        # **NEW: SKU/Product Performance Analysis**
        if 'SKU' in filtered_df.columns:
            st.subheader("📦 Product (SKU) Performance")
            
            # Top performing SKUs
            sku_analysis = filtered_df.groupby('SKU').agg({
                'Waybill': 'nunique',
                'Weight': 'sum',
                'Actual Delivered QTY': 'sum',
                'Rejected QTY': 'sum',
                'No of Pieces': 'sum'
            }).round(2)
            sku_analysis.columns = ['Orders', 'Total Weight', 'Delivered Qty', 'Rejected Qty', 'Total Pieces']
            sku_analysis['Success Rate %'] = np.where(
                sku_analysis['Total Pieces'] > 0,
                (sku_analysis['Delivered Qty'] / sku_analysis['Total Pieces'] * 100).round(1),
                0
            )
            sku_analysis = sku_analysis.sort_values('Orders', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 10 Products by Order Count**")
                top_skus = sku_analysis.head(10)[['Orders', 'Total Weight', 'Success Rate %']]
                st.dataframe(top_skus, use_container_width=True)
            
            with col2:
                st.write("**Products with Highest Rejection Rates**")
                problem_skus = sku_analysis[sku_analysis['Rejected Qty'] > 0].sort_values('Success Rate %').head(10)
                if not problem_skus.empty:
                    st.dataframe(problem_skus[['Orders', 'Rejected Qty', 'Success Rate %']], use_container_width=True)
                else:
                    st.info("No products with rejections in current filter")
        
        # **NEW: Delivery Time Analysis**
        if 'Delivery_Days' in filtered_df.columns:
            st.subheader("⏰ Delivery Time Performance")
            
            # Filter for delivered orders with valid delivery times
            delivered_with_time = filtered_df[
                (filtered_df['Current Status'].isin(['DELIVERED', 'PARTIALLY_DELIVERED'])) & 
                (filtered_df['Delivery_Days'].notna()) & 
                (filtered_df['Delivery_Days'] >= 0)
            ]
            
            if not delivered_with_time.empty:
                time_col1, time_col2, time_col3 = st.columns(3)
                
                avg_delivery_time = delivered_with_time['Delivery_Days'].mean()
                median_delivery_time = delivered_with_time['Delivery_Days'].median()
                fast_deliveries = len(delivered_with_time[delivered_with_time['Delivery_Days'] <= 1])
                
                with time_col1:
                    st.metric("Average Delivery Time", f"{avg_delivery_time:.1f} days")
                
                with time_col2:
                    st.metric("Median Delivery Time", f"{median_delivery_time:.1f} days")
                
                with time_col3:
                    same_day_rate = (fast_deliveries / len(delivered_with_time) * 100)
                    st.metric("Same/Next Day Delivery", f"{same_day_rate:.1f}%", help="Orders delivered within 1 day")
                
                # Delivery time by city
                city_delivery_time = delivered_with_time.groupby('Delivery City')['Delivery_Days'].agg(['mean', 'count']).round(1)
                city_delivery_time = city_delivery_time[city_delivery_time['count'] >= 5].sort_values('mean')
                
                if not city_delivery_time.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Fastest Delivery Cities**")
                        fastest_cities = city_delivery_time.head(10)
                        fastest_cities.columns = ['Avg Days', 'Orders']
                        st.dataframe(fastest_cities, use_container_width=True)
                    
                    with col2:
                        st.write("**Slowest Delivery Cities**")
                        slowest_cities = city_delivery_time.tail(10)
                        slowest_cities.columns = ['Avg Days', 'Orders']
                        st.dataframe(slowest_cities, use_container_width=True)
            else:
                st.info("Delivery time analysis requires delivered orders with valid dates")
        
        # Returns Analysis
        st.subheader("↩️ Returns Analysis")
        returns_df = filtered_df[filtered_df['Rejected QTY'] > 0]
        
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
                
                if not returns_by_city.empty:
                    returns_city_df = pd.DataFrame({
                        'City': returns_by_city.index,
                        'Rejected_Quantity': returns_by_city.values
                    })
                    
                    fig_returns = px.bar(
                        returns_city_df,
                        x='Rejected_Quantity',
                        y='City',
                        orientation='h',
                        title="Top 10 Cities by Return Quantity",
                        labels={'Rejected_Quantity': 'Rejected Quantity', 'City': 'City'}
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
                else:
                    st.info("No returns by city data available")
        else:
            st.info("No returns found in the selected data")
        
        # **NEW: Enhanced Business Insights**
        st.subheader("💡 Advanced Business Insights & Recommendations")
        
        insights = []
        
        # Delivery performance insights
        if len(filtered_df) > 0:
            delivery_rate = (len(filtered_df[filtered_df['Current Status'].isin(['DELIVERED', 'PARTIALLY_DELIVERED'])]) / len(filtered_df)) * 100
        else:
            delivery_rate = 0
        
        if delivery_rate < 85:
            insights.append(f"⚠️ **Delivery Rate Alert**: Current delivery success rate is {delivery_rate:.1f}%. Consider reviewing logistics operations.")
        elif delivery_rate > 95:
            insights.append(f"🎉 **Excellent Performance**: Delivery success rate of {delivery_rate:.1f}% is outstanding!")
        
        # **NEW: Direction-based insights**
        if 'Direction' in filtered_df.columns:
            reverse_orders = len(filtered_df[filtered_df['Direction'] == 'REVERSE'])
            total_orders = len(filtered_df)
            reverse_rate = (reverse_orders / total_orders * 100) if total_orders > 0 else 0
            
            if reverse_rate > 15:
                insights.append(f"🔄 **High Return Volume**: {reverse_rate:.1f}% of orders are returns. Investigate product quality or customer satisfaction issues.")
            elif reverse_rate > 0:
                insights.append(f"📊 **Return Monitoring**: {reverse_rate:.1f}% return rate is within normal range, but keep monitoring trends.")
        
        # **NEW: Business type insights** 
        if 'Business Type' in filtered_df.columns:
            business_dist = filtered_df['Business Type'].value_counts(normalize=True) * 100
            
            if business_dist.get('PRE_SALE', 0) > 50:
                insights.append(f"🏪 **Pre-Sale Dominance**: {business_dist.get('PRE_SALE', 0):.1f}% of orders are PRE_SALE. Consider expanding other business channels.")
            
            if business_dist.get('KEY_ACCOUNT', 0) > 25:
                insights.append(f"🔑 **Strong Key Accounts**: {business_dist.get('KEY_ACCOUNT', 0):.1f}% from key accounts. Maintain these relationships closely.")
        
        # **NEW: Delivery time insights**
        if 'Delivery_Days' in filtered_df.columns:
            delivered_with_time = filtered_df[
                (filtered_df['Current Status'].isin(['DELIVERED', 'PARTIALLY_DELIVERED'])) & 
                (filtered_df['Delivery_Days'].notna()) & 
                (filtered_df['Delivery_Days'] >= 0)
            ]
            
            if not delivered_with_time.empty:
                avg_delivery = delivered_with_time['Delivery_Days'].mean()
                if avg_delivery > 3:
                    insights.append(f"⏰ **Slow Delivery Alert**: Average delivery time is {avg_delivery:.1f} days. Consider optimizing routes or increasing capacity.")
                elif avg_delivery <= 1:
                    insights.append(f"⚡ **Excellent Speed**: Average delivery time of {avg_delivery:.1f} days is outstanding!")
        
        # Returns insights
        if not returns_df.empty:
            return_rate = (returns_df['Rejected QTY'].sum() / filtered_df['No of Pieces'].sum()) * 100
            if return_rate > 5:
                insights.append(f"📉 **High Return Rate**: {return_rate:.1f}% return rate needs attention. Focus on quality control.")
        
        # RFM insights with improved naming
        if not rfm_df.empty:
            vip_customers = len(rfm_df[rfm_df['Segment'] == 'VIP Customers'])
            lost_customers = len(rfm_df[rfm_df['Segment'] == 'Lost Customers'])
            inactive_frequent = len(rfm_df[rfm_df['Segment'] == 'Inactive Frequent Customers'])
            recent_new = len(rfm_df[rfm_df['Segment'] == 'Recent New Customers'])
            
            if vip_customers > 0:
                insights.append(f"⭐ **VIP Customers**: {vip_customers} high-value customers who order frequently. Provide premium service and exclusive offers.")
            
            if lost_customers > 0:
                insights.append(f"🚨 **Lost Customers**: {lost_customers} customers haven't ordered recently. Launch win-back campaigns immediately.")
                
            if inactive_frequent > 0:
                insights.append(f"💤 **Inactive Frequent Customers**: {inactive_frequent} previously active customers are now quiet. Investigate why and re-engage them.")
                
            if recent_new > 0:
                insights.append(f"🌟 **Recent New Customers**: {recent_new} new customers to nurture. Focus on building loyalty with follow-up service.")
        
        # **NEW: SKU performance insights**
        if 'SKU' in filtered_df.columns:
            sku_performance = filtered_df.groupby('SKU').agg({
                'Waybill': 'nunique',
                'Rejected QTY': 'sum',
                'No of Pieces': 'sum'
            })
            sku_performance['Rejection_Rate'] = np.where(
                sku_performance['No of Pieces'] > 0,
                (sku_performance['Rejected QTY'] / sku_performance['No of Pieces'] * 100),
                0
            )
            
            high_rejection_skus = sku_performance[
                (sku_performance['Rejection_Rate'] > 10) & 
                (sku_performance['Waybill'] >= 5)
            ]
            
            if not high_rejection_skus.empty:
                insights.append(f"📦 **Problem Products**: {len(high_rejection_skus)} SKUs have rejection rates >10%. Focus on quality control for these products.")
        
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
        
        st.markdown("---")
        st.markdown("*Dashboard last updated: {}*".format(datetime.now().strftime("%Y-%m-%d %H:%M")))

else:
    st.info("Please upload your deliveries CSV file to begin analysis.")
    
    # Show sample data format
    st.subheader("📋 Expected Data Format")
    st.markdown("""
    Your CSV file should contain the following columns:
    - **Waybill**: Unique order identifier
    - **Receiver Code**: Customer identifier
    - **Delivery City**: Destination city
    - **Current Status**: Delivery status (DELIVERED, PLANNED, etc.)
    - **Weight**: Package weight
    - **No of Pieces**: Number of items
    - **Actual Delivered QTY**: Quantity delivered
    - **Rejected QTY**: Quantity returned/rejected
    - **Create Time**: Order creation date
    - **Delivery Date**: Actual delivery date
    - **Business Type**: Type of business/customer
    - **Rejected Reason**: Reason for rejection (supports Arabic)
    """)