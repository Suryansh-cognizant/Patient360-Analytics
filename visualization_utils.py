#!/usr/bin/env python3
"""
Patient360 Analytics - Enhanced Visualization Utilities

Multi-library visualization framework with:
- Altair interactive statistical charts
- Bokeh advanced interactive dashboards
- Seaborn statistical plotting integration
- Folium geographic mapping
- Robust fallback system for library compatibility

Built for healthcare analytics with clinical context and interactivity.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Primary visualization libraries (with fallbacks)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import altair as alt
    alt.data_transformers.enable('json')
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

try:
    import bokeh.plotting as bp
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper, ColumnDataSource
    from bokeh.palettes import Viridis256, Category20
    from bokeh.embed import file_html
    from bokeh.resources import CDN
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('default')  # Clean style for healthcare
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# =============================================================================
# VISUALIZATION LIBRARY STATUS AND FALLBACK MANAGEMENT
# =============================================================================

def get_available_libraries():
    """Get status of all visualization libraries"""
    return {
        'plotly': PLOTLY_AVAILABLE,
        'altair': ALTAIR_AVAILABLE,
        'bokeh': BOKEH_AVAILABLE,
        'seaborn': SEABORN_AVAILABLE,
        'folium': FOLIUM_AVAILABLE
    }

def show_library_status():
    """Display library availability status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Visualization Libraries")
    
    libraries = get_available_libraries()
    for lib, available in libraries.items():
        status = "✅" if available else "❌"
        st.sidebar.text(f"{status} {lib.capitalize()}")
    
    total_available = sum(libraries.values())
    st.sidebar.info(f"{total_available}/5 libraries available")

# =============================================================================
# ALTAIR INTERACTIVE STATISTICAL CHARTS
# =============================================================================

def create_altair_mpr_distribution(df: pd.DataFrame, title: str = "MPR Distribution") -> Optional[alt.Chart]:
    """Create interactive MPR distribution line chart with Altair"""
    if not ALTAIR_AVAILABLE or 'MPR' not in df.columns:
        return None
    
    try:
        # Create binned data for line chart
        mpr_data = df['MPR'].dropna()
        bins = np.linspace(mpr_data.min(), mpr_data.max(), 30)
        hist_data = []
        
        for i in range(len(bins)-1):
            bin_center = (bins[i] + bins[i+1]) / 2
            count = ((mpr_data >= bins[i]) & (mpr_data < bins[i+1])).sum()
            hist_data.append({'MPR': bin_center, 'Count': count})
        
        # Add the last bin
        if len(bins) > 1:
            bin_center = bins[-1]
            count = (mpr_data >= bins[-1]).sum()
            hist_data.append({'MPR': bin_center, 'Count': count})
        
        line_df = pd.DataFrame(hist_data)
        
        # Base selection for interactivity
        brush = alt.selection_interval(bind='scales')
        
        # Main line chart
        line_chart = alt.Chart(line_df).add_selection(brush).mark_line(
            point=True, color='steelblue', strokeWidth=3
        ).encode(
            x=alt.X('MPR:Q', title='Medication Possession Ratio (%)', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('Count:Q', title='Number of Patients'),
            tooltip=['MPR:Q', 'Count:Q']
        )
        
        # Area under the curve for better visualization
        area_chart = alt.Chart(line_df).mark_area(
            opacity=0.3, color='steelblue'
        ).encode(
            x=alt.X('MPR:Q'),
            y=alt.Y('Count:Q')
        )
        
        # Adherence threshold line
        threshold_line = alt.Chart(pd.DataFrame({'x': [80]})).mark_rule(
            color='red', strokeDash=[5, 5], size=2
        ).encode(x='x:Q')
        
        # Text annotation for threshold
        max_count = line_df['Count'].max() if not line_df.empty else 0
        threshold_text = alt.Chart(pd.DataFrame({
            'x': [82], 
            'y': [max_count * 0.8], 
            'text': ['80% Adherence Threshold']
        })).mark_text(
            align='left', color='red', fontSize=11, fontWeight='bold'
        ).encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )
        
        # Combine all elements
        chart = (area_chart + line_chart + threshold_line + threshold_text).resolve_scale(
            y='independent'
        ).properties(
            title=title,
            width=600,
            height=300
        )
        
        return chart
        
    except Exception as e:
        st.warning(f"Altair line chart error: {e}")
        return None

def create_altair_disease_demographics(df: pd.DataFrame) -> Optional[alt.Chart]:
    """Create interactive disease demographics chart with Altair"""
    if not ALTAIR_AVAILABLE or 'Disease_Type' not in df.columns:
        return None
    
    try:
        # Brush selection
        brush = alt.selection_interval()
        
        # Disease distribution
        disease_chart = alt.Chart(df).add_selection(brush).mark_bar().encode(
            x=alt.X('count()', title='Number of Patients'),
            y=alt.Y('Disease_Type:N', sort='-x', title='Disease Type'),
            color=alt.condition(brush, alt.value('steelblue'), alt.value('lightgray')),
            tooltip=['Disease_Type:N', 'count()']
        ).properties(
            title='Disease Distribution (Interactive)',
            width=500,
            height=400
        )
        
        # Age distribution for selected diseases
        age_chart = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('Age:Q', scale=alt.Scale(zero=False)),
            y=alt.Y('Disease_Type:N', sort='-x'),
            color=alt.Color('Insurance_Type:N', legend=alt.Legend(title="Insurance")),
            opacity=alt.condition(brush, alt.value(0.8), alt.value(0.1)),
            tooltip=['Patient_ID:N', 'Age:Q', 'Disease_Type:N', 'Insurance_Type:N']
        ).properties(
            title='Age Distribution by Disease',
            width=400,
            height=400
        )
        
        return disease_chart | age_chart
        
    except Exception as e:
        st.warning(f"Altair demographics chart error: {e}")
        return None

def create_altair_correlation_matrix(df: pd.DataFrame) -> Optional[alt.Chart]:
    """Create interactive correlation heatmap with Altair"""
    if not ALTAIR_AVAILABLE:
        return None
    
    try:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return None
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().reset_index().melt('index')
        corr_matrix.columns = ['Variable1', 'Variable2', 'Correlation']
        
        # Create heatmap
        heatmap = alt.Chart(corr_matrix).mark_rect().encode(
            x=alt.X('Variable1:N', title=None),
            y=alt.Y('Variable2:N', title=None),
            color=alt.Color('Correlation:Q', 
                          scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                          legend=alt.Legend(title="Correlation")),
            tooltip=['Variable1:N', 'Variable2:N', 'Correlation:Q']
        ).properties(
            title='Healthcare Metrics Correlation Matrix',
            width=300,
            height=300
        )
        
        return heatmap
        
    except Exception as e:
        st.warning(f"Altair correlation chart error: {e}")
        return None

# =============================================================================
# BOKEH ADVANCED INTERACTIVE DASHBOARDS
# =============================================================================

def create_bokeh_adherence_dashboard(df: pd.DataFrame) -> Optional[str]:
    """Create comprehensive adherence dashboard with Bokeh"""
    if not BOKEH_AVAILABLE or 'MPR' not in df.columns:
        return None
    
    try:
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource, Tabs, Panel
        
        # Data source
        source = ColumnDataSource(df)
        
        # MPR histogram
        hist, edges = np.histogram(df['MPR'].dropna(), bins=20)
        p1 = bp.figure(title="MPR Distribution with Interactive Selection",
                      x_axis_label="MPR (%)", y_axis_label="Count",
                      width=800, height=400)
        p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="steelblue", line_color="white", alpha=0.7)
        
        # Add adherence threshold line
        p1.line([80, 80], [0, max(hist)], line_color="red", line_dash="dashed", line_width=2)
        
        # Scatter plot: Age vs MPR
        p2 = bp.figure(title="Age vs MPR (Interactive)",
                      x_axis_label="Age", y_axis_label="MPR (%)",
                      width=800, height=400)
        
        if 'Age' in df.columns:
            # Color by disease type if available
            if 'Disease_Type' in df.columns:
                diseases = df['Disease_Type'].unique()
                from bokeh.palettes import Category10
                if len(diseases) == 1:
                    colors = ["#1f77b4"]  # A default blue
                elif len(diseases) == 2:
                    colors = Category10[10][:2]
                else:
                    colors = Category20[20][:len(diseases)]
                for i, disease in enumerate(diseases[:20]):
                    disease_data = df[df['Disease_Type'] == disease]
                    p2.scatter('Age', 'MPR', size=8, alpha=0.6, 
                             color=colors[i], legend_label=disease,
                             source=ColumnDataSource(disease_data))
            else:
                p2.scatter('Age', 'MPR', size=8, alpha=0.6, source=source)
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Patient ID", "@Patient_ID"),
            ("Age", "@Age"),
            ("MPR", "@MPR"),
            ("Disease", "@Disease_Type")
        ])
        p2.add_tools(hover)
        
        # Layout: stack vertically for larger plots
        layout = column(p1, p2)
        
        # Generate HTML
        html = file_html(layout, CDN, "Bokeh Adherence Dashboard")
        return html
        
    except Exception as e:
        st.warning(f"Bokeh dashboard error: {e}")
        return None

def create_bokeh_time_series(df: pd.DataFrame) -> Optional[str]:
    """Create time series analysis with Bokeh"""
    if not BOKEH_AVAILABLE:
        return None
    
    try:
        # Look for actual date columns with more precise detection
        date_col = None
        available_date_cols = []
        
        # Check specific date column patterns (more precise)
        for col in df.columns:
            col_lower = col.lower()
            # Only consider columns that are likely to be actual dates
            if ('date' in col_lower and 'visit' not in col_lower) or \
               (col_lower in ['last_visit_date', 'prescription_date', 'date_of_birth', 'visit_date']) or \
               ('birth' in col_lower and 'date' in col_lower):
                # Additional check: see if the column actually contains date-like values
                sample_vals = df[col].dropna().head(5).astype(str)
                if any(val for val in sample_vals if '-' in str(val) or '/' in str(val) or len(str(val)) >= 8):
                    available_date_cols.append(col)
                    print(f"Found potential date column: {col} with sample values: {list(sample_vals)}")
        
        # Priority order for date columns
        priority_cols = ['Last_Visit_Date', 'Prescription_Date', 'Date_of_Birth', 'Visit_Date']
        
        # Select the best available date column
        for col in priority_cols:
            if col in available_date_cols:  # Only from validated date columns
                date_col = col
                break
        
        # If no priority column found, use any available date column
        if not date_col and available_date_cols:
            date_col = available_date_cols[0]
        
        if not date_col:
            print(f"No date columns found. Available columns: {list(df.columns)}")
            return None
        
        print(f"Using date column: {date_col}")
        
        # Process dates with better error handling
        df_copy = df.copy()
        
        # Clean and convert dates with multiple format attempts
        print(f"Original column sample values: {df_copy[date_col].head().tolist()}")
        
        # Try multiple date parsing approaches
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        # Remove rows with invalid dates or empty dates
        original_count = len(df_copy)
        df_copy = df_copy.dropna(subset=[date_col])
        df_copy = df_copy[df_copy[date_col].notna()]
        
        print(f"Data after date cleaning: {len(df_copy)} rows (was {original_count})")
        print(f"Sample parsed dates: {df_copy[date_col].head().tolist()}")
        
        if df_copy.empty:
            print("No valid dates found after cleaning - check your date formats")
            return None
        
        # More flexible date filtering - only filter obviously bad dates
        current_date = pd.Timestamp.now()
        date_range_start = df_copy[date_col].min()
        date_range_end = df_copy[date_col].max()
        
        print(f"Date range in data: {date_range_start} to {date_range_end}")
        
        # Only filter out clearly impossible dates
        if date_col == 'Date_of_Birth':
            # For birth dates, reasonable range is 1900 to current year
            before_filter = len(df_copy)
            df_copy = df_copy[
                (df_copy[date_col] >= pd.Timestamp('1900-01-01')) & 
                (df_copy[date_col] <= current_date)
            ]
            print(f"Birth date filtering: {before_filter} -> {len(df_copy)} rows")
        else:
            # For visit dates, be very flexible: 1990 to 2050
            before_filter = len(df_copy)
            df_copy = df_copy[
                (df_copy[date_col] >= pd.Timestamp('1990-01-01')) & 
                (df_copy[date_col] <= pd.Timestamp('2050-12-31'))
            ]
            print(f"Visit date filtering: {before_filter} -> {len(df_copy)} rows")
        
        print(f"Data after date range filtering: {len(df_copy)} rows")
        
        if df_copy.empty:
            print(f"No data remaining after date range filtering. Original range was {date_range_start} to {date_range_end}")
            return None
        
        # Group by month for better visualization - but handle small datasets
        if len(df_copy) > 50:
            # Monthly grouping for larger datasets
            df_copy['TimePeriod'] = df_copy[date_col].dt.to_period('M')
        elif len(df_copy) > 20:
            # Weekly grouping for medium datasets  
            df_copy['TimePeriod'] = df_copy[date_col].dt.to_period('W')
        else:
            # Daily grouping for small datasets
            df_copy['TimePeriod'] = df_copy[date_col].dt.to_period('D')
            
        period_counts = df_copy.groupby('TimePeriod').size().reset_index(name='Count')
        period_counts['Date'] = period_counts['TimePeriod'].dt.to_timestamp()
        
        print(f"Time series data points: {len(period_counts)}")
        
        if period_counts.empty or len(period_counts) < 2:
            print("Insufficient time series data points")
            return None
        
        # Create Bokeh plot
        from bokeh.models import HoverTool
        
        p = bp.figure(
            title=f"Patient Activity Timeline ({date_col.replace('_', ' ')})",
            x_axis_label="Date", 
            y_axis_label="Patient Count",
            x_axis_type="datetime", 
            width=700, 
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ('Date', '@Date{%F}'),
            ('Count', '@Count'),
        ], formatters={'@Date': 'datetime'})
        p.add_tools(hover)
        
        # Create line and circle plots
        source = ColumnDataSource(period_counts)
        p.line('Date', 'Count', source=source, line_color="#1f77b4", line_width=3)
        p.scatter('Date', 'Count', source=source, size=8, color="#1f77b4", alpha=0.8)
        
        # Style the plot
        p.title.text_color = "#2F4F4F"
        p.title.text_font_size = "16pt"
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        
        html = file_html(p, CDN, "Patient Timeline Analysis")
        print("Time series chart created successfully")
        return html
        
    except Exception as e:
        print(f"Bokeh time series error: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# SEABORN STATISTICAL PLOTTING
# =============================================================================

def create_seaborn_statistical_plots(df: pd.DataFrame) -> Dict[str, plt.Figure]:
    """Create comprehensive statistical plots with Seaborn"""
    if not SEABORN_AVAILABLE:
        return {}
    
    plots = {}
    
    try:
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # 1. MPR by Disease Type (if available)
        if 'MPR' in df.columns and 'Disease_Type' in df.columns:
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            # Get top 10 diseases by count
            top_diseases = df['Disease_Type'].value_counts().head(10).index
            df_filtered = df[df['Disease_Type'].isin(top_diseases)]
            
            sns.boxplot(data=df_filtered, x='Disease_Type', y='MPR', ax=ax1)
            ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
            ax1.set_title('MPR Distribution by Disease Type (Top 10)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Disease Type', fontweight='bold')
            ax1.set_ylabel('Medication Possession Ratio (%)', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            plt.tight_layout()
            plots['mpr_by_disease'] = fig1
        
        # 2. Age distribution by Insurance Type
        if 'Age' in df.columns and 'Insurance_Type' in df.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df, x='Insurance_Type', y='Age', ax=ax2)
            ax2.set_title('Age Distribution by Insurance Type', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Insurance Type', fontweight='bold')
            ax2.set_ylabel('Age (years)', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plots['age_by_insurance'] = fig2
        
        # 3. Correlation heatmap of numeric variables
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=ax3, square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
            ax3.set_title('Healthcare Metrics Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plots['correlation_heatmap'] = fig3
        
        # 4. Patient flow analysis (if visit data available)
        if 'Total_Visits' in df.columns:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='Total_Visits', bins=20, kde=True, ax=ax4)
            ax4.set_title('Patient Visit Frequency Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Total Visits', fontweight='bold')
            ax4.set_ylabel('Number of Patients', fontweight='bold')
            plt.tight_layout()
            plots['visit_distribution'] = fig4
        
        return plots
        
    except Exception as e:
        st.warning(f"Seaborn plots error: {e}")
        return {}

# =============================================================================
# FOLIUM GEOGRAPHIC MAPPING
# =============================================================================

def create_folium_geographic_analysis(df: pd.DataFrame) -> Optional[folium.Map]:
    """Create geographic analysis with Folium"""
    if not FOLIUM_AVAILABLE or 'Region' not in df.columns:
        return None
    
    try:
        # Regional patient counts
        region_counts = df['Region'].value_counts().to_dict()
        
        # Create base map (centered on US)
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, 
                      tiles='OpenStreetMap')
        
        # Define comprehensive coordinates for US states and regions
        region_coords = {
            # Major US States
            'California': [36.7783, -119.4179],
            'Texas': [31.9686, -99.9018],
            'Florida': [27.7663, -82.6404],
            'New York': [40.7128, -74.0060],
            'Illinois': [40.6331, -89.3985],
            'Pennsylvania': [41.2033, -77.1945],
            'Ohio': [40.4173, -82.9071],
            'Georgia': [32.1656, -82.9001],
            'Michigan': [44.3148, -85.6024],
            'Virginia': [37.4316, -78.6569],
            'North Carolina': [35.7596, -79.0193],
            'New Jersey': [40.0583, -74.4057],
            'Washington': [47.7511, -120.7401],
            'Arizona': [34.0489, -111.0937],
            'Massachusetts': [42.4072, -71.3824],
            'Tennessee': [35.5175, -86.5804],
            'Indiana': [40.2732, -86.1349],
            'Missouri': [37.9643, -91.8318],
            'Maryland': [39.0458, -76.6413],
            'Wisconsin': [43.7844, -88.7879],
            'Colorado': [39.7392, -104.9903],
            'Minnesota': [46.7296, -94.6859],
            'South Carolina': [33.8361, -81.1637],
            'Alabama': [32.3668, -86.7999],
            'Louisiana': [30.9843, -91.9623],
            'Kentucky': [37.8393, -84.2700],
            'Oregon': [43.8041, -120.5542],
            'Oklahoma': [35.0078, -97.0929],
            'Connecticut': [41.6032, -73.0877],
            'Utah': [39.3210, -111.0937],
            
            # International Regions (exact format from user's data)
            'Ontario, Canada': [51.2538, -85.3232],
            'Quebec, Canada': [52.9399, -73.5491],
            'British Columbia, Canada': [53.7267, -127.6476],
            'Manchester, UK': [53.4808, -2.2426],
            'Berlin, Germany': [52.5200, 13.4050],
            'Paris, France': [48.8566, 2.3522],
            'Madrid, Spain': [40.4168, -3.7038],
            'Rome, Italy': [41.9028, 12.4964],
            'Tokyo, Japan': [35.6762, 139.6503],
            'Sydney, Australia': [-33.8688, 151.2093],
            'Melbourne, Australia': [-37.8136, 144.9631],
            'Mumbai, India': [19.0760, 72.8777],
            'Dubai, UAE': [25.2048, 55.2708],
            'Lyon, France': [45.7640, 4.8357],
            'Barcelona, Spain': [41.3851, 2.1734],
            'Munich, Germany': [48.1351, 11.5820],
            'Osaka, Japan': [34.6937, 135.5023],
            
            # Major US Cities (if used as regions)
            'Chicago': [41.8781, -87.6298],
            'Los Angeles': [34.0522, -118.2437],
            'Houston': [29.7604, -95.3698],
            'Phoenix': [33.4484, -112.0740],
            'Philadelphia': [39.9526, -75.1652],
            'San Antonio': [29.4241, -98.4936],
            'San Diego': [32.7157, -117.1611],
            'Dallas': [32.7767, -96.7970],
            'San Francisco': [37.7749, -122.4194],
            'Austin': [30.2672, -97.7431],
            'Jacksonville': [30.3322, -81.6557],
            'Columbus': [39.9612, -82.9988],
            'Charlotte': [35.2271, -80.8431],
            'Indianapolis': [39.7684, -86.1581],
            'Seattle': [47.6062, -122.3321],
            'Denver': [39.7392, -104.9903],
            'Boston': [42.3601, -71.0589],
            'Nashville': [36.1627, -86.7816],
            'Miami': [25.7617, -80.1918],
            'Atlanta': [33.7490, -84.3880]
        }
        
        # Add markers for each region with patient counts
        for region, count in region_counts.items():
            if region in region_coords:
                coords = region_coords[region]
                
                # Calculate MPR stats for region if available
                region_data = df[df['Region'] == region]
                popup_text = f"<b>{region}</b><br>Patients: {count}"
                
                if 'MPR' in df.columns:
                    avg_mpr = region_data['MPR'].mean()
                    popup_text += f"<br>Avg MPR: {avg_mpr:.1f}%"
                
                # Color code by patient count
                if count > 100:
                    color = 'red'
                elif count > 50:
                    color = 'orange'
                else:
                    color = 'green'
                
                folium.CircleMarker(
                    location=coords,
                    radius=min(count/10, 50),  # Scale radius by patient count
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillOpacity=0.6
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Patient Distribution</h4>
        <p><i style="color:red">●</i> >100 patients</p>
        <p><i style="color:orange">●</i> 50-100 patients</p>
        <p><i style="color:green">●</i> <50 patients</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
        
    except Exception as e:
        st.warning(f"Folium map error: {e}")
        return None

def create_folium_disease_hotspots(df: pd.DataFrame) -> Optional[folium.Map]:
    """Create disease hotspot analysis with Folium"""
    if not FOLIUM_AVAILABLE or 'Region' not in df.columns or 'Disease_Type' not in df.columns:
        return None
    
    try:
        # Get disease distribution by region
        disease_by_region = df.groupby(['Region', 'Disease_Type']).size().reset_index(name='Count')
        
        # Create map centered for global view
        m = folium.Map(location=[20.0, 0.0], zoom_start=2)
        
        # Use comprehensive region coordinates (same as patient distribution)
        region_coords = {
            # Major US States
            'California': [36.7783, -119.4179],
            'Texas': [31.9686, -99.9018],
            'Florida': [27.7663, -82.6404],
            'New York': [40.7128, -74.0060],
            'Illinois': [40.6331, -89.3985],
            'Pennsylvania': [41.2033, -77.1945],
            'Ohio': [40.4173, -82.9071],
            'Georgia': [32.1656, -82.9001],
            'Michigan': [44.3148, -85.6024],
            'Virginia': [37.4316, -78.6569],
            'North Carolina': [35.7596, -79.0193],
            'New Jersey': [40.0583, -74.4057],
            'Washington': [47.7511, -120.7401],
            'Arizona': [34.0489, -111.0937],
            'Massachusetts': [42.4072, -71.3824],
            'Tennessee': [35.5175, -86.5804],
            'Indiana': [40.2732, -86.1349],
            'Missouri': [37.9643, -91.8318],
            'Maryland': [39.0458, -76.6413],
            'Wisconsin': [43.7844, -88.7879],
            'Colorado': [39.7392, -104.9903],
            'Minnesota': [46.7296, -94.6859],
            'South Carolina': [33.8361, -81.1637],
            'Alabama': [32.3668, -86.7999],
            'Louisiana': [30.9843, -91.9623],
            'Kentucky': [37.8393, -84.2700],
            'Oregon': [43.8041, -120.5542],
            'Oklahoma': [35.0078, -97.0929],
            'Connecticut': [41.6032, -73.0877],
            'Utah': [39.3210, -111.0937],
            
            # International Regions (exact format from user's data)
            'Ontario, Canada': [51.2538, -85.3232],
            'Quebec, Canada': [52.9399, -73.5491],
            'British Columbia, Canada': [53.7267, -127.6476],
            'Manchester, UK': [53.4808, -2.2426],
            'Berlin, Germany': [52.5200, 13.4050],
            'Paris, France': [48.8566, 2.3522],
            'Madrid, Spain': [40.4168, -3.7038],
            'Rome, Italy': [41.9028, 12.4964],
            'Tokyo, Japan': [35.6762, 139.6503],
            'Sydney, Australia': [-33.8688, 151.2093],
            'Melbourne, Australia': [-37.8136, 144.9631],
            'Mumbai, India': [19.0760, 72.8777],
            'Dubai, UAE': [25.2048, 55.2708],
            'Lyon, France': [45.7640, 4.8357],
            'Barcelona, Spain': [41.3851, 2.1734],
            'Munich, Germany': [48.1351, 11.5820],
            'Osaka, Japan': [34.6937, 135.5023],
        }
        
        # Add disease hotspots as heatmap data
        heat_data = []
        region_disease_totals = df.groupby('Region')['Disease_Type'].count().to_dict()
        
        # Create heat data based on disease burden per region
        for region, count in region_disease_totals.items():
            if region in region_coords:
                coords = region_coords[region]
                # Normalize weight (disease burden intensity)
                weight = min(count / 5.0, 1.0)  # Scale to 0-1 range
                heat_data.append([coords[0], coords[1], weight])
        
        print(f"Disease hotspot data points: {len(heat_data)}")
        
        if heat_data:
            # Add heatmap with disease intensity
            from folium import plugins
            plugins.HeatMap(
                heat_data, 
                radius=60, 
                blur=35, 
                gradient={
                    0.1: 'blue', 
                    0.3: 'cyan',
                    0.5: 'lime', 
                    0.7: 'yellow',
                    0.9: 'orange', 
                    1.0: 'red'
                },
                min_opacity=0.3,
                max_zoom=18
            ).add_to(m)
            
            # Add region markers with disease details
            for region in region_disease_totals.keys():
                if region in region_coords:
                    coords = region_coords[region]
                    
                    # Get top diseases for this region
                    region_diseases = df[df['Region'] == region]['Disease_Type'].value_counts().head(3)
                    disease_list = '<br>'.join([f"• {disease}: {count} patients" for disease, count in region_diseases.items()])
                    
                    popup_text = f"""
                    <b>{region}</b><br>
                    Total Patients: {region_disease_totals[region]}<br><br>
                    <b>Top Diseases:</b><br>
                    {disease_list}
                    """
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=8,
                        popup=folium.Popup(popup_text, max_width=300),
                        color='darkred',
                        fillColor='red',
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(m)
        else:
            print("No heat data generated - no matching regions found")
        
        return m
        
    except Exception as e:
        print(f"Folium hotspot map error: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# INTEGRATED VISUALIZATION SELECTOR
# =============================================================================

def create_enhanced_visualization(df: pd.DataFrame, viz_type: str, **kwargs) -> Any:
    """
    Create enhanced visualizations with automatic fallback
    
    Args:
        df: DataFrame with patient data
        viz_type: Type of visualization to create
        **kwargs: Additional parameters for specific visualizations
    
    Returns:
        Visualization object or None if creation fails
    """
    
    available_libs = get_available_libraries()
    
    if viz_type == "mpr_distribution":
        # Try Altair first, then fallback to Plotly
        if available_libs['altair']:
            chart = create_altair_mpr_distribution(df, kwargs.get('title', 'MPR Distribution'))
            if chart: return chart
        
        if available_libs['plotly'] and 'MPR' in df.columns:
            fig = px.histogram(df, x='MPR', nbins=20, title='MPR Distribution (Plotly Fallback)')
            fig.add_vline(x=80, line_dash="dash", line_color="red")
            return fig
    
    elif viz_type == "disease_demographics":
        if available_libs['altair']:
            chart = create_altair_disease_demographics(df)
            if chart: return chart
    
    elif viz_type == "correlation_matrix":
        if available_libs['altair']:
            chart = create_altair_correlation_matrix(df)
            if chart: return chart
    
    elif viz_type == "adherence_dashboard":
        if available_libs['bokeh']:
            html = create_bokeh_adherence_dashboard(df)
            if html: return html
    
    elif viz_type == "time_series":
        if available_libs['bokeh']:
            html = create_bokeh_time_series(df)
            if html: return html
    
    elif viz_type == "statistical_plots":
        if available_libs['seaborn']:
            plots = create_seaborn_statistical_plots(df)
            if plots: return plots
    
    elif viz_type == "geographic_map":
        if available_libs['folium']:
            map_obj = create_folium_geographic_analysis(df)
            if map_obj: return map_obj
    
    elif viz_type == "disease_hotspots":
        if available_libs['folium']:
            map_obj = create_folium_disease_hotspots(df)
            if map_obj: return map_obj
    
    return None

# =============================================================================
# VISUALIZATION ANALYSIS AND INSIGHTS
# =============================================================================

def generate_mpr_analysis_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive MPR analysis with clinical insights"""
    if 'MPR' not in df.columns:
        return {}
    
    try:
        mpr_data = pd.to_numeric(df['MPR'], errors='coerce').dropna()
        
        summary = {
            'total_patients': len(mpr_data),
            'mean_mpr': mpr_data.mean(),
            'median_mpr': mpr_data.median(),
            'std_mpr': mpr_data.std(),
            'adherent_patients': (mpr_data >= 80).sum(),
            'adherence_rate': (mpr_data >= 80).mean() * 100,
            'non_adherent_patients': (mpr_data < 80).sum(),
            'critical_patients': (mpr_data < 50).sum(),
            'excellent_adherence': (mpr_data >= 95).sum(),
            'quartiles': mpr_data.quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
        # Clinical risk stratification
        summary['risk_categories'] = {
            'Critical Risk (MPR < 50%)': (mpr_data < 50).sum(),
            'High Risk (50-65%)': ((mpr_data >= 50) & (mpr_data < 65)).sum(),
            'Moderate Risk (65-80%)': ((mpr_data >= 65) & (mpr_data < 80)).sum(),
            'Good Adherence (80-95%)': ((mpr_data >= 80) & (mpr_data < 95)).sum(),
            'Excellent Adherence (≥95%)': (mpr_data >= 95).sum()
        }
        
        # Clinical interpretations
        if summary['adherence_rate'] >= 85:
            summary['clinical_status'] = "✅ Excellent Population Adherence"
            summary['clinical_action'] = "Continue current strategies. Focus on maintaining high adherence."
        elif summary['adherence_rate'] >= 70:
            summary['clinical_status'] = "⚠️ Moderate Population Adherence"
            summary['clinical_action'] = "Implement targeted interventions for non-adherent patients."
        else:
            summary['clinical_status'] = "🚨 Poor Population Adherence"
            summary['clinical_action'] = "Urgent intervention required. Review prescribing practices and patient education."
        
        # Economic impact estimation
        if 'total_patients' in summary:
            estimated_savings = summary['adherent_patients'] * 2500  # Average annual savings per adherent patient
            potential_loss = summary['non_adherent_patients'] * 8000  # Average annual cost of non-adherence
            summary['economic_impact'] = {
                'estimated_savings': estimated_savings,
                'potential_loss': potential_loss,
                'net_impact': estimated_savings - potential_loss
            }
        
        return summary
        
    except Exception as e:
        return {'error': str(e)}

def generate_disease_demographics_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive disease demographics analysis"""
    if 'Disease_Type' not in df.columns:
        return {}
    
    try:
        summary = {
            'total_patients': len(df),
            'total_diseases': df['Disease_Type'].nunique(),
            'top_5_diseases': df['Disease_Type'].value_counts().head(5).to_dict(),
            'disease_distribution': df['Disease_Type'].value_counts().to_dict()
        }
        
        # Age analysis by disease
        if 'Age' in df.columns:
            age_by_disease = df.groupby('Disease_Type')['Age'].agg(['mean', 'median', 'std']).round(1)
            summary['age_by_disease'] = age_by_disease.to_dict('index')
            
            # Identify age-related patterns
            elderly_diseases = age_by_disease[age_by_disease['mean'] >= 65].index.tolist()
            young_diseases = age_by_disease[age_by_disease['mean'] < 40].index.tolist()
            
            summary['clinical_patterns'] = {
                'elderly_predominant': elderly_diseases,
                'younger_predominant': young_diseases,
                'overall_mean_age': df['Age'].mean()
            }
        
        # MPR analysis by disease
        if 'MPR' in df.columns:
            mpr_by_disease = df.groupby('Disease_Type')['MPR'].agg(['mean', 'median', 'count']).round(1)
            summary['adherence_by_disease'] = mpr_by_disease.to_dict('index')
            
            # Identify diseases with adherence issues
            low_adherence_diseases = mpr_by_disease[mpr_by_disease['mean'] < 70].index.tolist()
            high_adherence_diseases = mpr_by_disease[mpr_by_disease['mean'] >= 85].index.tolist()
            
            summary['adherence_patterns'] = {
                'problematic_diseases': low_adherence_diseases,
                'well_managed_diseases': high_adherence_diseases
            }
        
        # Generate clinical recommendations
        top_disease = max(summary['top_5_diseases'], key=summary['top_5_diseases'].get)
        summary['clinical_insights'] = {
            'primary_focus': f"{top_disease} represents {summary['top_5_diseases'][top_disease]} patients ({summary['top_5_diseases'][top_disease]/summary['total_patients']*100:.1f}%)",
            'diversity_index': summary['total_diseases'],
            'specialization_needed': summary['total_diseases'] > 15
        }
        
        return summary
        
    except Exception as e:
        return {'error': str(e)}

def generate_geographic_analysis_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive geographic analysis"""
    if 'Region' not in df.columns:
        return {}
    
    try:
        summary = {
            'total_regions': df['Region'].nunique(),
            'patients_by_region': df['Region'].value_counts().to_dict(),
            'regional_distribution': df['Region'].value_counts(normalize=True).mul(100).round(1).to_dict()
        }
        
        # MPR analysis by region
        if 'MPR' in df.columns:
            mpr_by_region = df.groupby('Region')['MPR'].agg(['mean', 'median', 'count']).round(1)
            summary['adherence_by_region'] = mpr_by_region.to_dict('index')
            
            # Identify regional patterns
            high_performing_regions = mpr_by_region[mpr_by_region['mean'] >= 80].index.tolist()
            underperforming_regions = mpr_by_region[mpr_by_region['mean'] < 70].index.tolist()
            
            summary['regional_performance'] = {
                'high_performing': high_performing_regions,
                'needs_improvement': underperforming_regions,
                'national_average': df['MPR'].mean()
            }
        
        # Disease distribution by region
        if 'Disease_Type' in df.columns:
            disease_by_region = df.groupby('Region')['Disease_Type'].apply(lambda x: x.value_counts().head(3)).to_dict()
            summary['top_diseases_by_region'] = disease_by_region
        
        # Resource allocation insights
        total_patients = len(df)
        top_3_regions = sorted(summary['patients_by_region'].items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary['resource_insights'] = {
            'concentration': f"Top 3 regions ({', '.join([r[0] for r in top_3_regions])}) contain {sum([r[1] for r in top_3_regions])} patients ({sum([r[1] for r in top_3_regions])/total_patients*100:.1f}%)",
            'geographic_spread': 'High' if summary['total_regions'] > 20 else 'Moderate' if summary['total_regions'] > 10 else 'Concentrated',
            'top_regions': [{'region': r[0], 'patients': r[1], 'percentage': r[1]/total_patients*100} for r in top_3_regions]
        }
        
        return summary
        
    except Exception as e:
        return {'error': str(e)}

def generate_correlation_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate insights from correlation analysis"""
    try:
        # Clean the data first - remove obvious data quality issues
        df_clean = df.copy()
        
        # Clean age column (remove unrealistic values like 200)
        if 'Age' in df_clean.columns:
            df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
            df_clean = df_clean[(df_clean['Age'] >= 0) & (df_clean['Age'] <= 120)]
        
        # Clean MPR column
        if 'MPR' in df_clean.columns:
            df_clean['MPR'] = pd.to_numeric(df_clean['MPR'], errors='coerce')
            df_clean = df_clean[(df_clean['MPR'] >= 0) & (df_clean['MPR'] <= 100)]
        
        # Get numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Also include boolean and categorical columns that can be converted
        for col in df_clean.columns:
            if col not in numeric_cols:
                if df_clean[col].dtype == 'object' or df_clean[col].dtype == 'bool':
                    # Try to convert to numeric if possible
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                        if not df_clean[col].isna().all():
                            numeric_cols.append(col)
                    except:
                        pass
        
        if len(numeric_cols) < 2:
            # Try to create numeric encodings for categorical variables
            categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                if df_clean[col].nunique() < 20:  # Only if not too many categories
                    df_clean[f'{col}_encoded'] = pd.Categorical(df_clean[col]).codes
                    numeric_cols.append(f'{col}_encoded')
        
        if len(numeric_cols) < 2:
            return {'error': 'Insufficient numeric columns for correlation analysis'}
        
        # Clean numeric data
        df_numeric = df_clean[numeric_cols].copy()
        
        # Convert all to numeric and handle errors
        for col in numeric_cols:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        
        # Remove columns with all NaN values or single unique value
        df_numeric = df_numeric.dropna(axis=1, how='all')
        for col in df_numeric.columns:
            if df_numeric[col].nunique() <= 1:
                df_numeric = df_numeric.drop(columns=[col])
        
        if df_numeric.shape[1] < 2:
            return {'error': 'Insufficient valid numeric data for correlation analysis'}
        
        # Fill NaN values with column medians
        df_numeric = df_numeric.fillna(df_numeric.median())
        
        corr_matrix = df_numeric.corr()
        
        # Find all correlations (lowered threshold even more)
        all_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not pd.isna(corr_val) and abs(corr_val) > 0.05:  # Very low threshold to catch any correlations
                    all_correlations.append({
                        'variables': (corr_matrix.columns[i], corr_matrix.columns[j]),
                        'correlation': corr_val,
                        'strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.4 else 'Weak'
                    })
        
        # Sort by absolute correlation value
        all_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        summary = {
            'total_variables': len(df_numeric.columns),
            'variables_analyzed': df_numeric.columns.tolist(),
            'strong_correlations': all_correlations[:20],  # Top 20
            'correlation_matrix_shape': corr_matrix.shape,
            'total_correlations_found': len(all_correlations),
            'data_rows_analyzed': len(df_numeric)
        }
        
        # Generate clinical interpretations based on actual correlations found
        clinical_insights = []
        
        for corr in all_correlations[:8]:  # Check top 8 correlations
            var1, var2 = corr['variables']
            corr_val = corr['correlation']
            
            # Clean variable names for display
            clean_var1 = var1.replace('_encoded', '').replace('_', ' ')
            clean_var2 = var2.replace('_encoded', '').replace('_', ' ')
            
            if 'MPR' in [var1, var2] and 'Age' in [var1, var2]:
                if corr_val > 0:
                    clinical_insights.append(f"📈 Older patients tend to have better medication adherence (r={corr_val:.3f})")
                else:
                    clinical_insights.append(f"📉 Younger patients show better adherence patterns (r={corr_val:.3f})")
            
            elif 'Total_Visits' in [var1, var2] and 'MPR' in [var1, var2]:
                if corr_val > 0:
                    clinical_insights.append(f"🏥 Higher visit frequency correlates with better adherence (r={corr_val:.3f})")
                else:
                    clinical_insights.append(f"⚠️ More visits may indicate adherence problems (r={corr_val:.3f})")
            
            elif 'Age' in [var1, var2] and 'Total_Visits' in [var1, var2]:
                if corr_val > 0:
                    clinical_insights.append(f"👴 Older patients tend to have more frequent visits (r={corr_val:.3f})")
                else:
                    clinical_insights.append(f"👶 Younger patients have fewer routine visits (r={corr_val:.3f})")
            
            else:
                # Generic insight for other correlations
                if abs(corr_val) > 0.3:
                    clinical_insights.append(f"🔗 {clean_var1} and {clean_var2} show {'positive' if corr_val > 0 else 'negative'} correlation (r={corr_val:.3f})")
        
        if not clinical_insights:
            clinical_insights.append("📊 Correlations found but no specific clinical patterns identified")
            
        summary['clinical_insights'] = clinical_insights
        
        return summary
        
    except Exception as e:
        return {'error': f'Correlation analysis failed: {str(e)}'}

# =============================================================================
# VISUALIZATION TESTING AND DEMO
# =============================================================================

def display_analysis_summary(summary_data: Dict[str, Any], analysis_type: str):
    """Display comprehensive analysis summary with clinical insights"""
    
    if not summary_data or 'error' in summary_data:
        st.warning(f"Unable to generate {analysis_type} analysis summary")
        return
    
    st.markdown("---")
    st.subheader(f"📊 {analysis_type} Analysis Summary")
    
    # Create summary tabs
    tabs = st.tabs(["📈 Key Metrics", "🎯 Clinical Insights", "📋 Recommendations", "💡 Detailed Analysis"])
    
    with tabs[0]:  # Key Metrics
        if analysis_type == "MPR":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patients", f"{summary_data.get('total_patients', 0):,}")
            with col2:
                st.metric("Adherence Rate", f"{summary_data.get('adherence_rate', 0):.1f}%", 
                         delta=f"{summary_data.get('adherence_rate', 0) - 80:.1f}%")
            with col3:
                st.metric("Avg MPR", f"{summary_data.get('mean_mpr', 0):.1f}%")
            with col4:
                st.metric("Critical Patients", summary_data.get('critical_patients', 0), 
                         delta="Risk < 50%")
            
            # Risk stratification chart
            if 'risk_categories' in summary_data:
                st.subheader("🎯 Risk Stratification")
                risk_data = pd.DataFrame(list(summary_data['risk_categories'].items()), 
                                       columns=['Risk Level', 'Patients'])
                fig = px.bar(risk_data, x='Patients', y='Risk Level', 
                            title="Patient Distribution by Adherence Risk",
                            color='Patients', color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Disease Demographics":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Diseases", summary_data.get('total_diseases', 0))
            with col2:
                st.metric("Total Patients", f"{summary_data.get('total_patients', 0):,}")
            with col3:
                if 'clinical_patterns' in summary_data:
                    st.metric("Avg Age", f"{summary_data['clinical_patterns'].get('overall_mean_age', 0):.1f} years")
            
            # Top diseases chart
            if 'top_5_diseases' in summary_data:
                st.subheader("🏥 Top 5 Disease Types")
                disease_data = pd.DataFrame(list(summary_data['top_5_diseases'].items()), 
                                          columns=['Disease', 'Patients'])
                fig = px.pie(disease_data, values='Patients', names='Disease', 
                            title="Disease Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Geographic":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Regions", summary_data.get('total_regions', 0))
            with col2:
                if 'regional_performance' in summary_data:
                    st.metric("National Avg MPR", f"{summary_data['regional_performance'].get('national_average', 0):.1f}%")
            with col3:
                high_perf = len(summary_data.get('regional_performance', {}).get('high_performing', []))
                st.metric("High Performing Regions", high_perf)
    
    with tabs[1]:  # Clinical Insights
        st.subheader("🔍 Clinical Insights")
        
        if analysis_type == "MPR":
            st.markdown(f"**Population Status:** {summary_data.get('clinical_status', 'Unknown')}")
            st.info(summary_data.get('clinical_action', 'No specific action recommended'))
            
            if 'economic_impact' in summary_data:
                st.subheader("💰 Economic Impact Analysis")
                econ = summary_data['economic_impact']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Estimated Savings", f"${econ.get('estimated_savings', 0):,}")
                with col2:
                    st.metric("Potential Loss", f"${econ.get('potential_loss', 0):,}")
                with col3:
                    net_impact = econ.get('net_impact', 0)
                    st.metric("Net Impact", f"${net_impact:,}", 
                             delta="Positive" if net_impact > 0 else "Negative")
        
        elif analysis_type == "Disease Demographics":
            if 'clinical_patterns' in summary_data:
                patterns = summary_data['clinical_patterns']
                if patterns.get('elderly_predominant'):
                    st.info(f"👴 **Elderly-Predominant Diseases:** {', '.join(patterns['elderly_predominant'][:3])}")
                if patterns.get('younger_predominant'):
                    st.info(f"👶 **Younger-Predominant Diseases:** {', '.join(patterns['younger_predominant'][:3])}")
            
            if 'adherence_patterns' in summary_data:
                adh_patterns = summary_data['adherence_patterns']
                if adh_patterns.get('problematic_diseases'):
                    st.warning(f"⚠️ **Low Adherence Diseases:** {', '.join(adh_patterns['problematic_diseases'][:3])}")
                if adh_patterns.get('well_managed_diseases'):
                    st.success(f"✅ **Well-Managed Diseases:** {', '.join(adh_patterns['well_managed_diseases'][:3])}")
        
        elif analysis_type == "Geographic":
            if 'resource_insights' in summary_data:
                insights = summary_data['resource_insights']
                st.info(f"📍 **Geographic Concentration:** {insights.get('concentration')}")
                st.info(f"🗺️ **Coverage Pattern:** {insights.get('geographic_spread')} geographic spread")
        
        elif analysis_type == "Correlation":
            if 'clinical_insights' in summary_data:
                for insight in summary_data['clinical_insights']:
                    st.info(insight)
    
    with tabs[2]:  # Recommendations
        st.subheader("💡 Clinical Recommendations")
        
        if analysis_type == "MPR":
            adherence_rate = summary_data.get('adherence_rate', 0)
            critical_patients = summary_data.get('critical_patients', 0)
            
            recommendations = []
            if adherence_rate < 70:
                recommendations.append("🚨 **Priority:** Implement population-wide adherence improvement program")
                recommendations.append("📚 **Education:** Develop patient education materials on medication importance")
                recommendations.append("📱 **Technology:** Consider medication reminder apps or systems")
            
            if critical_patients > 0:
                recommendations.append(f"⚠️ **Critical:** {critical_patients} patients need immediate intervention (MPR < 50%)")
                recommendations.append("🏥 **Outreach:** Schedule clinical consultations for high-risk patients")
            
            recommendations.append("📊 **Monitoring:** Establish monthly MPR tracking and reporting")
            recommendations.append("💊 **Pharmacy:** Collaborate with pharmacies for adherence support programs")
            
            for rec in recommendations:
                st.markdown(rec)
        
        elif analysis_type == "Disease Demographics":
            total_diseases = summary_data.get('total_diseases', 0)
            
            recommendations = []
            if total_diseases > 20:
                recommendations.append("🏥 **Specialization:** Consider establishing disease-specific care teams")
            
            recommendations.append("📋 **Protocols:** Develop standardized care protocols for top 5 diseases")
            recommendations.append("👥 **Resources:** Allocate clinical resources based on patient volume")
            recommendations.append("📚 **Training:** Provide disease-specific training for healthcare staff")
            
            for rec in recommendations:
                st.markdown(rec)
        
        elif analysis_type == "Geographic":
            recommendations = [
                "🗺️ **Coverage:** Ensure adequate healthcare coverage in all regions",
                "🚐 **Mobile Services:** Consider mobile clinics for underserved areas",
                "💻 **Telemedicine:** Implement telehealth for remote regions",
                "📊 **Regional Analysis:** Establish region-specific quality metrics"
            ]
            
            for rec in recommendations:
                st.markdown(rec)
    
    with tabs[3]:  # Detailed Analysis
        st.subheader("📋 Detailed Statistical Analysis")
        
        if analysis_type == "MPR" and 'quartiles' in summary_data:
            st.subheader("📊 MPR Distribution Quartiles")
            quartiles_df = pd.DataFrame(list(summary_data['quartiles'].items()), 
                                      columns=['Quartile', 'MPR Value'])
            quartiles_df['Quartile'] = ['25th Percentile', '50th Percentile (Median)', '75th Percentile']
            st.dataframe(quartiles_df, use_container_width=True)
        
        elif analysis_type == "Disease Demographics":
            if 'age_by_disease' in summary_data:
                st.subheader("👥 Age Statistics by Disease Type")
                age_df = pd.DataFrame.from_dict(summary_data['age_by_disease'], orient='index')
                age_df = age_df.round(1).reset_index()
                age_df.columns = ['Disease Type', 'Mean Age', 'Median Age', 'Std Dev']
                st.dataframe(age_df.head(10), use_container_width=True)
        
        elif analysis_type == "Geographic":
            if 'adherence_by_region' in summary_data:
                st.subheader("🗺️ Regional Performance Analysis")
                region_df = pd.DataFrame.from_dict(summary_data['adherence_by_region'], orient='index')
                region_df = region_df.round(1).reset_index()
                region_df.columns = ['Region', 'Mean MPR', 'Median MPR', 'Patient Count']
                region_df = region_df.sort_values('Mean MPR', ascending=False)
                st.dataframe(region_df.head(15), use_container_width=True)
        
        elif analysis_type == "Correlation":
            if 'strong_correlations' in summary_data:
                st.subheader("🔗 Strong Correlations Found")
                corr_data = []
                for corr in summary_data['strong_correlations'][:10]:
                    corr_data.append({
                        'Variable 1': corr['variables'][0],
                        'Variable 2': corr['variables'][1],
                        'Correlation': f"{corr['correlation']:.3f}",
                        'Strength': corr['strength']
                    })
                
                if corr_data:
                    corr_df = pd.DataFrame(corr_data)
                    st.dataframe(corr_df, use_container_width=True)

def show_visualization_showcase(df: pd.DataFrame):
    """Display a showcase of all available enhanced visualizations"""
    
    st.header("🎨 Enhanced Visualization Showcase")
    
    # Library status
    show_library_status()
    
    available_libs = get_available_libraries()
    total_libs = sum(available_libs.values())
    
    if total_libs == 0:
        st.error("No visualization libraries available. Please install: altair, bokeh, seaborn, folium")
        return
    
    st.success(f"✅ {total_libs}/5 enhanced visualization libraries available")
    
    # Visualization options
    viz_options = st.multiselect(
        "Select Visualizations to Display:",
        [
            "Interactive MPR Distribution (Altair)",
            "Disease Demographics (Altair)", 
            "Correlation Matrix (Altair)",
            "Adherence Dashboard (Bokeh)",
            "Statistical Plots (Seaborn)",
            "Geographic Analysis (Folium)",
            "Disease Hotspots (Folium)"
        ]
    )
    
    # Create and display selected visualizations
    for option in viz_options:
        st.subheader(f"📊 {option}")
        
        if "MPR Distribution" in option and available_libs['altair']:
            chart = create_enhanced_visualization(df, "mpr_distribution")
            if chart:
                st.altair_chart(chart, use_container_width=True)
        
        elif "Disease Demographics" in option and available_libs['altair']:
            chart = create_enhanced_visualization(df, "disease_demographics") 
            if chart:
                st.altair_chart(chart, use_container_width=True)
        
        elif "Correlation Matrix" in option and available_libs['altair']:
            chart = create_enhanced_visualization(df, "correlation_matrix")
            if chart:
                st.altair_chart(chart, use_container_width=True)
        
        elif "Adherence Dashboard" in option and available_libs['bokeh']:
            html = create_enhanced_visualization(df, "adherence_dashboard")
            if html:
                st.components.v1.html(html, height=600)
        
        elif "Statistical Plots" in option and available_libs['seaborn']:
            plots = create_enhanced_visualization(df, "statistical_plots")
            for plot_name, fig in plots.items():
                st.pyplot(fig)
        
        elif "Geographic Analysis" in option and available_libs['folium']:
            map_obj = create_enhanced_visualization(df, "geographic_map")
            if map_obj:
                st.components.v1.html(map_obj._repr_html_(), height=600, width=1200, scrolling=False)
        
        elif "Disease Hotspots" in option and available_libs['folium']:
            map_obj = create_enhanced_visualization(df, "disease_hotspots")
            if map_obj:
                st.components.v1.html(map_obj._repr_html_(), height=600, width=1200, scrolling=False)
