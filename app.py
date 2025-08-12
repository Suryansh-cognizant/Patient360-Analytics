"""
Patient360 Analytics Dashboard
A comprehensive healthcare analytics platform with data quality transparency

Built according to new_plan.md specifications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import our analytics utilities
from analytics_utils import (
    load_data, filter_data, calculate_mpr, flag_overdue_visits,
    generate_data_quality_dashboard, create_missing_value_heatmap,
    get_sample_for_visualization, patient_persona_clustering,
    detect_outliers, adherence_prediction, cohort_trends, 
    custom_segmentation, calculate_overall_quality_score,
    clean_data, calculate_balance_skewness_metrics, display_balance_skewness_dashboard
)

# Import enhanced visualization utilities
try:
    from visualization_utils import (
        create_enhanced_visualization, show_visualization_showcase,
        show_library_status, get_available_libraries,
        display_analysis_summary, generate_mpr_analysis_summary,
        generate_disease_demographics_summary, generate_geographic_analysis_summary,
        generate_correlation_insights
    )
    ENHANCED_VIZ_AVAILABLE = True
except ImportError:
    ENHANCED_VIZ_AVAILABLE = False

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Patient360 Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ADAPTIVE CLINICAL CONTEXT SYSTEM
# =============================================================================

def assess_data_quality(df):
    """Assess overall data quality and return context level"""
    if df is None or df.empty:
        return "no_data"
    
    # Calculate missing value percentage
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    
    # Assess critical data completeness
    critical_cols = ['Patient_ID', 'Age', 'Gender']
    critical_missing = 0
    for col in critical_cols:
        if col in df.columns:
            critical_missing += df[col].isnull().sum()
    
    critical_missing_pct = (critical_missing / (len(df) * len([c for c in critical_cols if c in df.columns]))) * 100 if len(df) > 0 else 0
    
    # Determine quality level
    if missing_pct < 5 and critical_missing_pct < 1:
        return "high_quality"
    elif missing_pct < 15 and critical_missing_pct < 5:
        return "moderate_quality"
    else:
        return "low_quality"

def get_clinical_context(quality_level, missing_pct=0):
    """Return appropriate clinical context based on data quality"""
    contexts = {
        "high_quality": {
            "status": "✅ High-Quality Dataset",
            "message": "Excellent data completeness detected. Analytics results are highly reliable for clinical decision-making and population health insights.",
            "color": "success"
        },
        "moderate_quality": {
            "status": "⚠️ Moderate Data Quality",
            "message": f"Some missing values detected (~{missing_pct:.1f}% overall) and cleaned automatically. Results are generally reliable but interpret with appropriate clinical context.",
            "color": "warning"
        },
        "low_quality": {
            "status": "🚨 Significant Data Quality Issues",
            "message": f"Substantial missing data (~{missing_pct:.1f}% overall) detected and cleaned. Results should be validated with clinical expertise before making care decisions.",
            "color": "error"
        },
        "no_data": {
            "status": "📋 No Data Loaded",
            "message": "Please upload patient data to begin analysis.",
            "color": "info"
        }
    }
    return contexts.get(quality_level, contexts["no_data"])

def display_clinical_context_banner():
    """Display adaptive clinical context banner based on current data quality"""
    if 'data_quality_level' in st.session_state and st.session_state.data_quality_level:
        quality_level = st.session_state.data_quality_level
        missing_pct = st.session_state.get('missing_percentage', 0)
        context = get_clinical_context(quality_level, missing_pct)
        
        if context["color"] == "success":
            st.success(f"**{context['status']}**: {context['message']}")
        elif context["color"] == "warning":
            st.warning(f"**{context['status']}**: {context['message']}")
        elif context["color"] == "error":
            st.error(f"**{context['status']}**: {context['message']}")
        else:
            st.info(f"**{context['status']}**: {context['message']}")

# =============================================================================
# SIDEBAR NAVIGATION AND FILTERS
# =============================================================================

def create_sidebar():
    """Create sidebar with navigation and filters"""
    
    st.sidebar.title("🏥 Patient360 Analytics")
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.subheader("📍 Navigation")
    sections = ["Home", "Cleaned Data Overview", "Demographics Overview", "Adherence Analysis", 
               "Recency Analysis", "Advanced Analytics"]
    
    if ENHANCED_VIZ_AVAILABLE:
        sections.append("Enhanced Visualizations")
    
    section = st.sidebar.selectbox(
        "Select Section",
        sections,
        key="section_selector"
    )
    
    # Enhanced visualization library status
    #if ENHANCED_VIZ_AVAILABLE:
    #    show_library_status()

    
    # Data filters (only show if data is loaded and not on Home page)
    filters = {}
    if 'data' in st.session_state and st.session_state.data is not None and section != "Home":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Data Filters")
        
        df = st.session_state.data
        
        # Disease Type filter
        if 'Disease_Type' in df.columns:
            disease_options = df['Disease_Type'].dropna().unique().tolist()
            filters['diseases'] = st.sidebar.multiselect(
                "Disease Type", disease_options, default=disease_options[:5] if len(disease_options) > 5 else disease_options
            )
        
        # Region filter
        if 'Region' in df.columns:
            region_options = df['Region'].dropna().unique().tolist()
            filters['regions'] = st.sidebar.multiselect(
                "Region", region_options, default=region_options[:10] if len(region_options) > 10 else region_options
            )
        
        # Insurance Type filter
        if 'Insurance_Type' in df.columns:
            insurance_options = df['Insurance_Type'].dropna().unique().tolist()
            filters['insurance_types'] = st.sidebar.multiselect(
                "Insurance Type", insurance_options, default=insurance_options
            )
    
    return section, filters

# =============================================================================
# HOME SECTION - DATA UPLOAD AND QUALITY DASHBOARD
# =============================================================================

def show_home_section():
    """Home section with data upload and comprehensive data quality dashboard"""
    
    st.title("🏥 Patient360 Analytics Dashboard")
    st.markdown("### Comprehensive Healthcare Analytics with Data Quality Transparency")

    # --- Sample Data Download Options ---
    import os
    st.subheader("📥 Download Sample Data")
    
    # User guidance
    st.info(
        "📋 **For best results and most accurate analytics, use the clean sample file.** "
        "You may also use the uncleaned dataset with missing values to see how the app's data cleaning and quality checks work, "
        "but results and accuracy may vary depending on data quality."
    )
    
    col1, col2 = st.columns(2)
    
    # Clean dataset option
    with col1:
        clean_paths = [
            os.path.join(os.path.dirname(__file__), "data", "sample_data_10k_cleaned.csv"),
            "data/sample_data_10k_cleaned.csv"
        ]
        
        clean_path = None
        for path in clean_paths:
            if os.path.exists(path):
                clean_path = path
                break
        
        if clean_path:
            try:
                with open(clean_path, "rb") as f:
                    st.download_button(
                        label="✅ Download Clean Sample Data",
                        data=f,
                        file_name="sample_data_10k_cleaned.csv",
                        mime="text/csv",
                        help="Download cleaned sample data with no missing values (recommended)"
                    )
            except Exception as e:
                st.error(f"Error loading clean sample file: {e}")
        else:
            st.warning("⚠️ Clean sample data not found.")
    
    # Uncleaned dataset option
    with col2:
        uncleaned_paths = [
            os.path.join(os.path.dirname(__file__), "data", "sample_data_10k_uncleaned.csv"),
            "data/sample_data_10k_uncleaned.csv"
        ]
        
        uncleaned_path = None
        for path in uncleaned_paths:
            if os.path.exists(path):
                uncleaned_path = path
                break
        
        if uncleaned_path:
            try:
                with open(uncleaned_path, "rb") as f:
                    st.download_button(
                        label="🔧 Download Uncleaned Sample Data",
                        data=f,
                        file_name="sample_data_10k_uncleaned.csv",
                        mime="text/csv",
                        help="Download uncleaned sample data with missing values to test cleaning features"
                    )
            except Exception as e:
                st.error(f"Error loading uncleaned sample file: {e}")
        else:
            st.warning("⚠️ Uncleaned sample data not found.")
    
    # Data Upload Section
    st.subheader("📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Patient Data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload your patient dataset. The system will automatically assess data quality and provide transparency reports."
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner("Loading and processing data..."):
                if uploaded_file.name.endswith('.csv'):
                    df_original = pd.read_csv(uploaded_file)
                else:
                    df_original = pd.read_excel(uploaded_file)
                
                # Store original data
                st.session_state.original_data = df_original.copy()
                
                # Assess data quality of original data (before cleaning)
                quality_level = assess_data_quality(df_original)
                missing_pct = (df_original.isnull().sum().sum() / (len(df_original) * len(df_original.columns))) * 100
                
                # Store quality assessment in session state
                st.session_state.data_quality_level = quality_level
                st.session_state.missing_percentage = missing_pct
            
            st.success(f"✅ Data loaded successfully! {len(df_original):,} records with {len(df_original.columns)} columns")
            
            # Display adaptive clinical context banner
            display_clinical_context_banner()
            
            # Automatic Data Cleaning (silent)
            with st.spinner("Cleaning data..."):
                df_cleaned = clean_data(df_original, show_report=False)
                st.session_state.data = df_cleaned  # Store cleaned data for all sections
            
            # Data Quality Dashboard (on cleaned data)
            st.markdown("---")
            generate_data_quality_dashboard(df_cleaned)
            
            # Data Preview (Cleaned Data)
            st.subheader("👀 Data Preview (Cleaned)")
            
            # Show sample data
            preview_df = df_cleaned.head(20)
            st.dataframe(preview_df, use_container_width=True)
            
            # Comparison metrics: Original vs Cleaned
            st.subheader("📈 Data Statistics (Original vs Cleaned)")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                original_count = len(df_original)
                cleaned_count = len(df_cleaned)
                delta = cleaned_count - original_count
                st.metric("Total Records", f"{cleaned_count:,}", delta=f"{delta:,} from original")
            with col2:
                st.metric("Total Columns", len(df_cleaned.columns))
            with col3:
                numeric_cols = len(df_cleaned.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_cols)
            with col4:
                memory_mb = df_cleaned.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            
            # Geographic Distribution (if Region column exists)
            if 'Region' in df_cleaned.columns:
                st.subheader("🗺️ Geographic Distribution")
                
                region_counts = df_cleaned['Region'].value_counts().head(10)
                
                if not region_counts.empty:
                    fig = px.bar(
                        x=region_counts.index,
                        y=region_counts.values,
                        title="Patient Distribution by Region (Top 10)",
                        labels={'x': 'Region', 'y': 'Patient Count'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Missing Value Heatmap
            st.subheader("🌡️ Data Quality Visualization")
            create_missing_value_heatmap(df_cleaned)
            
            # Navigation Instructions
            st.info("""
            **Next Steps:**
            1. Review the data quality dashboard above
            2. Use the sidebar to navigate to different analysis sections
            3. Apply filters to focus on specific patient populations
            4. Each section provides detailed methodology and clinical context
            """)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please ensure your file is in the correct format with appropriate column names.")
    
    else:
        # Instructions when no data is uploaded
        st.info("""
        **Welcome to Patient360 Analytics!**
        
        This dashboard provides comprehensive healthcare analytics with full data transparency:
        
        **🔍 Data Quality Features:**
        - Automatic data quality assessment
        - Missing value analysis and visualization  
        - Duplicate detection and reporting
        - Data cleaning transparency
        
        **📊 Analytics Sections:**
        - **Demographics**: Population analysis with clinical context
        - **Adherence**: Medication adherence monitoring (MPR calculation)
        - **Recency**: Care gap analysis and visit patterns
        - **Advanced**: ML-driven insights and predictions
        
        **🎯 Built for Healthcare Professionals:**
        - Plain-English explanations for all metrics
        - Clinical context and business impact
        - Actionable recommendations
        - Transparent methodology
        
        **📁 To Get Started:**
        Upload your patient data file (CSV or Excel) above to begin analysis.
        """)

# =============================================================================
# CLEANED DATA OVERVIEW SECTION
# =============================================================================

def show_cleaned_data_section():
    """Section to display cleaned data, cleaning report, balance/skewness, and comparison with original data"""
    st.title("🧼 Cleaned Data Overview")
    st.markdown("### Latest Cleaned Data, Cleaning Report, and Data Balance/Skewness")

    df_original = st.session_state.get('original_data', None)
    df_cleaned = st.session_state.get('data', None)

    if df_original is None or df_cleaned is None:
        st.warning("Please upload data in the Home section first.")
        return

    # Side-by-side preview
    st.subheader("👀 Side-by-Side Preview: Original vs Cleaned Data")
    n = min(10, len(df_original), len(df_cleaned))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Data (first 10 rows):**")
        st.dataframe(df_original.head(n), use_container_width=True)
    with col2:
        st.markdown("**Cleaned Data (first 10 rows):**")
        st.dataframe(df_cleaned.head(n), use_container_width=True)

    # Download cleaned data
    st.markdown("---")
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=df_cleaned.to_csv(index=False),
        file_name="cleaned_patient_data.csv",
        mime="text/csv"
    )

    # Cleaning report (rerun cleaning logic for display only)
    st.markdown("---")
    st.subheader("🧹 Cleaning Report (from last upload)")
    clean_data(df_original, show_report=True)

    # Balance/skewness metrics
    st.markdown("---")
    st.subheader("⚖️ Data Balance & Skewness (from last upload)")
    metrics = calculate_balance_skewness_metrics(df_cleaned)
    display_balance_skewness_dashboard(metrics)

# =============================================================================
# DEMOGRAPHICS OVERVIEW SECTION
# =============================================================================

def show_demographics_section(df):
    """Demographics analysis with data transparency"""
    
    st.title("👥 Demographics Overview")
    st.markdown("### Comprehensive Population Demographics Analysis")
    
    if df is None or df.empty:
        st.warning("Please upload data in the Home section first.")
        return
    
    # Sample data for performance if needed
    viz_df = get_sample_for_visualization(df)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        if 'Age' in df.columns:
            avg_age = df['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f}" if not pd.isna(avg_age) else "N/A")
        else:
            st.metric("Average Age", "N/A")
    with col3:
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            if not gender_counts.empty:
                dominant_gender = gender_counts.index[0]
                st.metric("Most Common Gender", dominant_gender)
            else:
                st.metric("Most Common Gender", "N/A")
        else:
            st.metric("Most Common Gender", "N/A")
    with col4:
        if 'Disease_Type' in df.columns:
            disease_count = df['Disease_Type'].nunique()
            st.metric("Disease Types", disease_count)
        else:
            st.metric("Disease Types", "N/A")
    
    # Age Distribution - Histogram
    if 'Age' in df.columns:
        st.subheader("👥 Age Distribution (Histogram)")
        fig = px.histogram(viz_df, x='Age', nbins=20, title="Patient Age Distribution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Graph Explanation:** Shows the frequency of patients by age group.")
        st.info("""
        **Description:**
        This histogram displays the distribution of patient ages in the dataset. It helps identify the most represented age groups, spot potential outliers, and guide age-specific screening or prevention programs.
        """)

    # Gender/Insurance/Region - Stacked Bar Chart
    if all(col in df.columns for col in ['Gender', 'Insurance_Type', 'Region']):
        st.subheader("📊 Gender, Insurance & Region Distribution (Stacked Bar Chart)")
        stacked_df = viz_df.groupby(['Region', 'Gender', 'Insurance_Type']).size().reset_index(name='Count')
        fig = px.bar(stacked_df, x='Region', y='Count', color='Gender', barmode='stack',
                     facet_col='Insurance_Type', title="Distribution by Region, Gender, and Insurance")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Graph Explanation:** Stacked bars show gender breakdown per region, split by insurance type.")
        st.info("""
        **Description:**
        This stacked bar chart visualizes the distribution of patients by region, gender, and insurance type. It highlights demographic imbalances and helps target regions or groups for outreach or support.
        """)

    # Multi-category - Treemap
    if all(col in df.columns for col in ['Region', 'Insurance_Type', 'Gender']):
        st.subheader("🌳 Multi-Category Demographic Breakdown (Treemap)")
        treemap_df = viz_df.groupby(['Region', 'Insurance_Type', 'Gender']).size().reset_index(name='Count')
        fig = px.treemap(treemap_df, path=['Region', 'Insurance_Type', 'Gender'], values='Count',
                         title="Demographic Breakdown Treemap")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Graph Explanation:** Each rectangle represents a subgroup; size shows patient count.")
        st.info("""
        **Description:**
        The treemap provides a hierarchical view of the patient population by region, insurance, and gender. It helps identify dominant subgroups and diversity across locations and payers.
        """)
    
    # Gender Breakdown - Horizontal Stacked Bar Chart
    if 'Gender' in df.columns:
        st.subheader("⚖️ Gender Distribution (Horizontal Stacked Bar)")
        gender_counts = viz_df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        fig = px.bar(
            gender_counts,
            x='Count',
            y='Gender',
            orientation='h',
            title="Patient Distribution by Gender",
            color='Gender',
            text='Count',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Graph Explanation:** Horizontal stacked bar chart shows patient counts for each gender category, making comparisons intuitive and accessible.")
    
    # Disease Type Distribution
    if 'Disease_Type' in df.columns:
        st.subheader("🏥 Disease Type Distribution")
        
        disease_counts = viz_df['Disease_Type'].value_counts().head(10)
        
        if not disease_counts.empty:
            fig = px.bar(
                x=disease_counts.values,
                y=disease_counts.index,
                orientation='h',
                title="Top 10 Disease Types",
                labels={'x': 'Patient Count', 'y': 'Disease Type'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Business impact
            st.success(f"""
            **Business Impact:**
            - Top condition: {disease_counts.index[0]} ({disease_counts.iloc[0]:,} patients)
            - Consider specialized care programs for high-volume conditions
            - Resource allocation should prioritize top 5 conditions
            """)
    
    # Insurance Type Analysis
    if 'Insurance_Type' in df.columns:
        st.subheader("💳 Insurance Coverage Analysis")
        
        insurance_counts = viz_df['Insurance_Type'].value_counts()
        
        if not insurance_counts.empty:
            fig = px.pie(
                values=insurance_counts.values,
                names=insurance_counts.index,
                title="Patient Distribution by Insurance Type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Financial context
            st.info("""
            **Financial Planning Context:**
            - Monitor payer mix for revenue forecasting
            - Identify opportunities for value-based contracts
            - Plan for uninsured/self-pay patient support programs
            """)

# =============================================================================
# ADHERENCE ANALYSIS SECTION  
# =============================================================================

def show_adherence_section(df):
    """Medication adherence analysis with transparent MPR calculation"""
    
    st.title("💊 Adherence Analysis")
    st.markdown("### Medication Adherence Monitoring & Analysis")
    
    if df is None or df.empty:
        st.warning("Please upload data in the Home section first.")
        return
    
    # Use MPR column if present, otherwise calculate
    if 'MPR' in df.columns:
        mpr_df = df[['Patient_ID', 'MPR']].copy()
        # Add robust columns for compatibility
        mpr_df['Total_Days_Supplied'] = (
            df['Total_Days_Supplied'] if 'Total_Days_Supplied' in df.columns
            else (df['Days_Supplied'] if 'Days_Supplied' in df.columns else np.nan)
        )
        mpr_df['Prescription_Count'] = (
            df['Prescription_Count'] if 'Prescription_Count' in df.columns
            else (df['Total_Prescriptions'] if 'Total_Prescriptions' in df.columns else np.nan)
        )
    else:
        mpr_df = calculate_mpr(df, show_transparency=True)
        if mpr_df.empty:
            st.error("Unable to calculate MPR. Please check your data has required columns: Patient_ID, Prescription_Date, Days_Supplied")
            return
    
    # Key Adherence Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    adherent_patients = (mpr_df['MPR'] >= 80).sum()
    total_patients = len(mpr_df)
    adherence_rate = (adherent_patients / total_patients) * 100
    avg_mpr = mpr_df['MPR'].mean()
    
    with col1:
        st.metric("Adherent Patients", f"{adherent_patients:,}", delta=f"{adherence_rate:.1f}% of total")
    with col2:
        st.metric("Average MPR", f"{avg_mpr:.1f}%")
    with col3:
        st.metric("Non-Adherent", f"{total_patients - adherent_patients:,}", delta=f"{100-adherence_rate:.1f}%")
    with col4:
        median_mpr = mpr_df['MPR'].median()
        st.metric("Median MPR", f"{median_mpr:.1f}%")
    
    # MPR Distribution - Density Plot
    st.subheader("📊 MPR Distribution (Density Plot)")
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    sns.kdeplot(mpr_df['MPR'], bw_adjust=1, fill=True, color='blue')
    plt.axvline(80, color='red', linestyle='--', label='80% Adherence Threshold')
    plt.title('Medication Possession Ratio (MPR) Density Plot')
    plt.xlabel('MPR (%)')
    plt.ylabel('Density')
    plt.legend()
    st.pyplot(plt.gcf())
    st.caption("**Graph Explanation:** Shows the smoothed distribution of MPR across patients, with a clinical threshold.")
    st.info("""
    **Description:**
    This density plot visualizes the distribution of Medication Possession Ratio (MPR) in the population. The red line marks the 80% adherence threshold. Peaks show common adherence levels; left-skew indicates more non-adherence.
    """)



    # Adherence by Group - Grouped Bar Chart
    if 'Age' in df.columns:
        st.subheader("📊 Adherence by Age Group (Grouped Bar Chart)")
        age_bins = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 120], labels=["0-18","19-35","36-50","51-65","66+"])
        group_df = df.copy()
        group_df['Age_Group'] = age_bins
        group_adherence = group_df.groupby('Age_Group')['MPR'].mean().reset_index()
        fig = px.bar(group_adherence, x='Age_Group', y='MPR', title="Average MPR by Age Group")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Graph Explanation:** Bars compare average adherence across age groups.")
        st.info("""
        **Description:**
        This grouped bar chart shows average medication adherence (MPR) for different age groups. It helps identify which age segments need more adherence support.
        """)

    # Trends Over Time - Area Chart
    if 'Prescription_Date' in df.columns:
        st.subheader("📈 Adherence Trends Over Time (Area Chart)")
        df['Prescription_Date'] = pd.to_datetime(df['Prescription_Date'], errors='coerce')
        time_trend = df.groupby(df['Prescription_Date'].dt.to_period('M'))['MPR'].mean().reset_index()
        time_trend['Prescription_Date'] = time_trend['Prescription_Date'].dt.to_timestamp()
        fig = px.area(time_trend, x='Prescription_Date', y='MPR', title="Monthly Adherence Trend")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Graph Explanation:** Area shows average adherence per month.")
        st.info("""
        **Description:**
        This area chart displays how average MPR changes over time. It helps monitor the effectiveness of adherence programs and spot seasonal or policy-driven trends.
        """)
    # Non-adherent patients table
    st.subheader("⚠️ Non-Adherent Patients (MPR < 80%)")
    
    non_adherent = mpr_df[mpr_df['MPR'] < 80].sort_values('MPR')
    
    if not non_adherent.empty:
        st.dataframe(
            non_adherent[['Patient_ID', 'MPR', 'Total_Days_Supplied', 'Prescription_Count']].head(20),
            use_container_width=True
        )
        
        # Clinical recommendations
        st.error(f"""
        **Clinical Action Required:**
        - {len(non_adherent):,} patients need adherence intervention
        - Priority: Patients with MPR < 50% (highest risk)
        - Consider: Medication synchronization, patient education, adherence monitoring
        """)
    else:
        st.success("🎉 All patients meet adherence threshold!")
    
    # Adherence by Insurance Type (if available)
    if 'Insurance_Type' in df.columns:
        st.subheader("📈 Adherence by Insurance Type")
        
        # Create merged data with proper column handling
        if 'MPR' in df.columns:
            # Use original data directly since it already has MPR
            df_with_mpr = df.copy()
        else:
            # Merge calculated MPR with original data
            df_with_mpr = df.merge(mpr_df[['Patient_ID', 'MPR']], on='Patient_ID', how='inner')
        
        if not df_with_mpr.empty and 'MPR' in df_with_mpr.columns:
            fig = px.box(
                df_with_mpr, 
                x='Insurance_Type', 
                y='MPR',
                title="MPR Distribution by Insurance Type"
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Graph Explanation:** Each box shows the spread and central tendency of MPR for an insurance type. The red line marks the 80% adherence threshold.")
            st.info("""
            **Description:**
            This box plot compares medication adherence (MPR) across different insurance types. It helps identify which payer groups have better or worse adherence, supporting targeted payer interventions and contract negotiations.
            """)

# =============================================================================
# RECENCY ANALYSIS SECTION
# =============================================================================

def show_recency_section(df):
    """Patient visit recency and care gap analysis"""
    
    st.title("📅 Recency Analysis")
    st.markdown("### Care Gap Identification & Visit Pattern Analysis")
    
    if df is None or df.empty:
        st.warning("Please upload data in the Home section first.")
        return
    
    # Flag overdue visits with transparency
    overdue_df = flag_overdue_visits(df, days=180, show_transparency=True)
    
    # Key Recency Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_patients = len(df)
    overdue_count = len(overdue_df)
    overdue_rate = (overdue_count / total_patients) * 100 if total_patients > 0 else 0
    
    with col1:
        st.metric("Total Patients", f"{total_patients:,}")
    with col2:
        st.metric("Overdue Visits", f"{overdue_count:,}", delta=f"{overdue_rate:.1f}%")
    with col3:
        if 'Last_Visit_Date' in df.columns:
            recent_visits = df['Last_Visit_Date'].notna().sum()
            st.metric("Patients with Visit Data", f"{recent_visits:,}")
        else:
            st.metric("Patients with Visit Data", "N/A")
    with col4:
        missing_visits = df['Last_Visit_Date'].isna().sum() if 'Last_Visit_Date' in df.columns else 0
        st.metric("Missing Visit Dates", f"{missing_visits:,}")
    
    # Time Since Last Visit - Cumulative Distribution Plot
    if 'Last_Visit_Date' in df.columns and not df['Last_Visit_Date'].isna().all():
        st.subheader("📊 Time Since Last Visit (Cumulative Distribution)")
        df_copy = df.copy()
        df_copy['Last_Visit_Date'] = pd.to_datetime(df_copy['Last_Visit_Date'], errors='coerce')
        df_copy['Days_Since_Visit'] = (pd.Timestamp.today() - df_copy['Last_Visit_Date']).dt.days
        valid_visits = df_copy.dropna(subset=['Days_Since_Visit'])
        if not valid_visits.empty:
            sorted_days = np.sort(valid_visits['Days_Since_Visit'])
            cum_pct = np.arange(1, len(sorted_days)+1) / len(sorted_days)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,4))
            plt.plot(sorted_days, cum_pct, color='blue')
            plt.axvline(180, color='red', linestyle='--', label='180-Day Overdue Threshold')
            plt.xlabel('Days Since Last Visit')
            plt.ylabel('Cumulative % of Patients')
            plt.title('Cumulative Distribution: Days Since Last Visit')
            plt.legend()
            st.pyplot(plt.gcf())
            st.caption("**Graph Explanation:** Shows what % of patients had visits within a given number of days.")
            st.info("""
            **Description:**
            This cumulative distribution plot displays the percentage of patients by days since their last visit. The red line marks the clinical threshold for overdue care.
            """)

    # Get overdue data for metrics (but don't display the list to avoid duplication)
    overdue_df = flag_overdue_visits(df, days=180, show_transparency=True)

    # Recent Activity by Group - Dot Plot
    if 'Region' in df.columns and 'Last_Visit_Date' in df.columns:
        st.subheader("📈 Recent Activity by Region (Dot Plot)")
        df_copy = df.copy()
        df_copy['Last_Visit_Date'] = pd.to_datetime(df_copy['Last_Visit_Date'], errors='coerce')
        df_copy['Days_Since_Visit'] = (pd.Timestamp.today() - df_copy['Last_Visit_Date']).dt.days
        region_activity = df_copy.groupby('Region')['Days_Since_Visit'].mean().reset_index()
        fig = px.scatter(region_activity, x='Region', y='Days_Since_Visit',
                         title="Average Days Since Last Visit by Region", size='Days_Since_Visit', color='Days_Since_Visit')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Graph Explanation:** Each dot shows the average recency for a region.")
        st.info("""
        **Description:**
        This dot plot visualizes the average time since last visit for each region. Larger dots indicate longer gaps, helping to target regions with lower patient engagement.
        """)

    # Patient Recency Segments - Lollipop Chart
    if 'Last_Visit_Date' in df.columns:
        st.subheader("🍭 Patient Recency Segments (Lollipop Chart)")
        df_copy = df.copy()
        df_copy['Last_Visit_Date'] = pd.to_datetime(df_copy['Last_Visit_Date'], errors='coerce')
        df_copy['Days_Since_Visit'] = (pd.Timestamp.today() - df_copy['Last_Visit_Date']).dt.days
        bins = [0, 30, 90, 180, 365, np.inf]
        labels = ['<30d','30-89d','90-179d','180-365d','>365d']
        df_copy['Recency_Segment'] = pd.cut(df_copy['Days_Since_Visit'], bins=bins, labels=labels, right=False)
        seg_counts = df_copy['Recency_Segment'].value_counts().sort_index().reset_index()
        seg_counts.columns = ['Recency_Segment', 'Patient_Count']
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.stem(seg_counts['Recency_Segment'], seg_counts['Patient_Count'], basefmt=" ")
        plt.xlabel('Recency Segment')
        plt.ylabel('Number of Patients')
        plt.title('Patient Recency Segments (Lollipop Chart)')
        st.pyplot(plt.gcf())
        st.caption("**Graph Explanation:** Each lollipop shows the number of patients in each recency segment.")
        st.info("""
        **Description:**
        This lollipop chart segments patients by how recently they visited. It helps identify populations at risk for care gaps and target follow-up efforts.
        """)
    # Overdue patients by priority
    if not overdue_df.empty:
        st.subheader("🚨 Overdue Patients - Priority List")
        
        # Sort by days since last visit (most overdue first)
        priority_patients = overdue_df.sort_values('Days_Since_Last_Visit', ascending=False)
        
        # Display top 20 most overdue
        display_cols = ['Patient_ID', 'Days_Since_Last_Visit']
        if 'Disease_Type' in priority_patients.columns:
            display_cols.append('Disease_Type')
        if 'Insurance_Type' in priority_patients.columns:
            display_cols.append('Insurance_Type')
        
        st.dataframe(
            priority_patients[display_cols].head(20),
            use_container_width=True
        )
        
        # Care coordination recommendations
        st.warning(f"""
        **Care Coordination Actions:**
        - {len(priority_patients):,} patients need outreach for care continuity
        - Priority: Patients >365 days overdue
        - Consider: Automated appointment reminders, care coordinator calls
        - Monitor: Chronic disease patients requiring regular follow-up
        """)
    else:
        st.success("🎉 No patients are overdue for visits!")
    
    # Visit patterns by disease type
    if 'Disease_Type' in df.columns and 'Last_Visit_Date' in df.columns:
        st.subheader("🏥 Visit Patterns by Disease Type")
        
        df_with_days = df.copy()
        df_with_days['Last_Visit_Date'] = pd.to_datetime(df_with_days['Last_Visit_Date'], errors='coerce')
        df_with_days['Days_Since_Visit'] = (pd.Timestamp.today() - df_with_days['Last_Visit_Date']).dt.days
        
        valid_data = df_with_days.dropna(subset=['Days_Since_Visit', 'Disease_Type'])
        
        if not valid_data.empty:
            fig = px.box(
                valid_data,
                x='Disease_Type',
                y='Days_Since_Visit',
                title="Visit Recency by Disease Type"
            )
            fig.add_hline(y=180, line_dash="dash", line_color="red")
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Graph Explanation:** Each box shows the distribution of days since last visit for each disease type. The red line marks the 180-day overdue threshold.")
            st.info("""
            **Description:**
            This plot visualizes how recently patients with different diseases have visited. It highlights care gaps and helps prioritize outreach for conditions with longer average time since last visit.
            """)
        else:
            st.info("No valid data available for disease type comparison")

# =============================================================================
# ADVANCED ANALYTICS SECTION
# =============================================================================

def show_advanced_analytics_section(df):
    """Advanced ML-driven analytics with model transparency"""
    
    st.title("🧠 Advanced Analytics")
    st.markdown("### Machine Learning-Driven Insights & Predictive Analytics")
    
    if df is None or df.empty:
        st.warning("Please upload data in the Home section first.")
        return
    
    # Model transparency introduction
    st.info("""
    **Model Transparency Promise:**
    All machine learning models used here provide:
    - Plain-English algorithm explanations
    - Feature importance rankings  
    - Confidence scores and limitations
    - Training data characteristics
    - Bias detection and mitigation strategies
    """)
    
    # Analytics Options
    analytics_option = st.selectbox(
        "Select Advanced Analytics",
        ["Patient Persona Clustering", "Outlier Detection", "Adherence Risk Prediction", 
         "Cohort Trends", "Custom Segmentation"]
    )
    
    if analytics_option == "Patient Persona Clustering":
        show_clustering_analysis(df)
    elif analytics_option == "Outlier Detection":
        show_outlier_analysis(df)
    elif analytics_option == "Adherence Risk Prediction":
        show_prediction_analysis(df)
    elif analytics_option == "Cohort Trends":
        show_cohort_analysis(df)
    elif analytics_option == "Custom Segmentation":
        show_segmentation_analysis(df)

def show_clustering_analysis(df):
    """Patient persona clustering with transparency"""
    
    st.subheader("👥 Patient Persona Clustering")
    
    st.info("""
    **Algorithm:** K-Means Clustering

    **Purpose:** Group patients into distinct personas for targeted care strategies

    **Features Used:** Age, Medication Adherence (MPR), Comorbidities
    
    **Clinical Value:** Enables personalized care pathways and resource allocation
    """)
    
    # Number of clusters
    n_clusters = st.slider("Number of Patient Personas", 2, 8, 3)
    
    if st.button("Generate Patient Personas"):
        with st.spinner("Analyzing patient personas..."):
            try:
                # Perform clustering
                cluster_results = patient_persona_clustering(df, n_clusters=n_clusters)
                
                if cluster_results is not None and not cluster_results.empty:
                    st.success(f"✅ Successfully identified {n_clusters} patient personas!")
                    
                    # Display cluster summary
                    st.subheader("📊 Cluster Summary")
                    cluster_summary = cluster_results.groupby('Cluster').agg({
                        'Patient_ID': 'count',
                        'Age': 'mean',
                        'MPR': 'mean' if 'MPR' in cluster_results.columns else lambda x: 0
                    }).round(2)
                    cluster_summary.columns = ['Patient Count', 'Avg Age', 'Avg MPR']
                    st.dataframe(cluster_summary, use_container_width=True)
                    
                    # Show sample patients from each cluster
                    st.subheader("👥 Sample Patients by Persona")
                    for cluster_id in cluster_results['Cluster'].unique():
                        cluster_data = cluster_results[cluster_results['Cluster'] == cluster_id]
                        with st.expander(f"Persona {cluster_id} ({len(cluster_data)} patients)"):
                            st.dataframe(cluster_data.head(10), use_container_width=True)
                else:
                    st.error("Unable to perform clustering. Please ensure your data has Age and other required columns.")
                    
            except Exception as e:
                st.error(f"Error in clustering analysis: {e}")
                st.info("Please ensure your data has the required columns: Patient_ID, Age")

def show_outlier_analysis(df):
    """Outlier detection with transparency"""
    
    st.subheader("📊 Outlier Detection")
    
    st.info("""
    **Algorithm:** Statistical Outlier Detection (IQR Method)

    **Purpose:** Identify unusual patients requiring investigation

    **Features:** Age, MPR values, Days since last visit

    **Clinical Value:** Data quality assurance and exception-based care management
    """)
    
    if st.button("Detect Outliers"):
        with st.spinner("Detecting outliers..."):
            try:
                outliers = detect_outliers(df)
                
                if outliers is not None and not outliers.empty:
                    st.warning(f"⚠️ Found {len(outliers)} outlier patients")
                    
                    # Display outliers with explanation
                    st.subheader("🔍 Outlier Details")
                    st.dataframe(outliers.head(20), use_container_width=True)
                    
                    # Outlier statistics
                    st.subheader("📈 Outlier Statistics")
                    for col in outliers.columns:
                        if pd.api.types.is_numeric_dtype(outliers[col]):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{col} - Min", f"{outliers[col].min():.1f}")
                            with col2:
                                st.metric(f"{col} - Max", f"{outliers[col].max():.1f}")
                            with col3:
                                st.metric(f"{col} - Mean", f"{outliers[col].mean():.1f}")
                    
                    # Clinical recommendations
                    st.info("""
                    **Outlier Investigation Recommendations:**
                    - Review outlier patients for data entry errors
                    - Validate extreme age values (>100 or <0)
                    - Check for data quality issues in source systems
                    - Consider if outliers represent valid edge cases
                    """)
                else:
                    st.success("✅ No significant outliers detected!")
                    st.info("Your data appears to be within normal statistical ranges.")
                    
            except Exception as e:
                st.error(f"Error in outlier detection: {e}")
                st.info("Please ensure your data has numeric columns like Age for outlier detection.")

def show_prediction_analysis(df):
    """Adherence risk prediction with transparency"""
    
    st.subheader("🎯 Adherence Risk Prediction")
    
    st.info("""
    **Algorithm:** Random Forest Classifier

    **Purpose:** Predict which patients are at risk of non-adherence

    **Features:** Age, Gender, Region, Insurance Type, Comorbidities  

    **Clinical Value:** Proactive intervention targeting and resource prioritization
    """)
    
    if st.button("Generate Risk Predictions"):
        with st.spinner("Training prediction model..."):
            try:
                predictions = adherence_prediction(df)
                
                if predictions is not None and len(predictions) > 0:
                    st.success("✅ Risk predictions generated!")
                    
                    # Display prediction results
                    st.subheader("📊 Adherence Risk Predictions")
                    
                    # Create risk categories
                    high_risk = [p for p in predictions if p.get('risk_score', 0) > 0.7]
                    medium_risk = [p for p in predictions if 0.3 < p.get('risk_score', 0) <= 0.7]
                    low_risk = [p for p in predictions if p.get('risk_score', 0) <= 0.3]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("High Risk", len(high_risk), delta=f"{len(high_risk)/len(predictions)*100:.1f}%")
                    with col2:
                        st.metric("Medium Risk", len(medium_risk), delta=f"{len(medium_risk)/len(predictions)*100:.1f}%")
                    with col3:
                        st.metric("Low Risk", len(low_risk), delta=f"{len(low_risk)/len(predictions)*100:.1f}%")
                    
                    # Show high-risk patients
                    if high_risk:
                        st.subheader("🚨 High-Risk Patients (Risk Score > 0.7)")
                        high_risk_df = pd.DataFrame(high_risk[:20])  # Top 20
                        st.dataframe(high_risk_df, use_container_width=True)
                        
                        st.error(f"""
                        **Immediate Action Required:**
                        - {len(high_risk)} patients at high risk of non-adherence
                        - Prioritize for medication counseling and adherence programs
                        - Consider medication synchronization and simplified regimens
                        """)
                    
                else:
                    st.error("Unable to generate predictions. Please check your data has required columns.")
                    st.info("Required columns: Patient_ID, Age, and either prescription or visit data.")
                    
            except Exception as e:
                st.error(f"Error in prediction analysis: {e}")
                st.info("Please ensure your data has sufficient information for prediction modeling.")

def show_cohort_analysis(df):
    """Cohort trends analysis"""
    
    st.subheader("📈 Cohort Trends Analysis")
    
    st.info("""
    **Purpose:** Analyze temporal patterns in patient adherence and outcomes

    **Method:** Time-series analysis of patient cohorts

    **Clinical Value:** Seasonal pattern identification and program effectiveness tracking
    """)
    
    if st.button("Analyze Cohort Trends"):
        with st.spinner("Analyzing cohort trends..."):
            try:
                trends = cohort_trends(df)
                
                if trends is not None and not trends.empty:
                    st.success("✅ Cohort trends analyzed!")
                    
                    # Display trend results
                    st.subheader("📊 Cohort Analysis Results")
                    st.dataframe(trends.head(20), use_container_width=True)
                    
                    # Create trend visualization if possible
                    if 'Cohort' in trends.columns and len(trends) > 1:
                        fig = px.line(
                            trends, 
                            x='Cohort', 
                            y=trends.columns[1] if len(trends.columns) > 1 else None,
                            title="Cohort Trends Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Business insights
                    st.info("""
                    **Cohort Insights:**
                    - Monitor trends across different patient enrollment periods
                    - Identify seasonal patterns in adherence or outcomes
                    - Track effectiveness of program changes over time
                    """)
                    
                else:
                    st.error("Unable to analyze trends. Please check your data has date columns.")
                    st.info("Cohort analysis requires columns like Prescription_Date or Last_Visit_Date.")
                    
            except Exception as e:
                st.error(f"Error in cohort analysis: {e}")
                st.info("Please ensure your data has appropriate date columns for trend analysis.")

def show_segmentation_analysis(df):
    """Custom patient segmentation"""
    
    st.subheader("🎯 Custom Patient Segmentation")
    
    st.info("""
    **Purpose:** Flexible patient segmentation by any variable

    **Method:** Group patients by selected characteristics

    **Clinical Value:** Ad-hoc analysis and custom reporting capabilities
    """)
    
    # Segmentation options
    segment_options = [col for col in df.columns if col in ['Disease_Type', 'Insurance_Type', 'Region', 'Comorbidities', 'Physician_ID']]
    
    if segment_options:
        segment_by = st.selectbox("Segment Patients By", segment_options)
        
        if st.button("Generate Segmentation"):
            with st.spinner("Segmenting patients..."):
                try:
                    segmentation = custom_segmentation(df, by=segment_by)
                    
                    if segmentation is not None and not segmentation.empty:
                        st.success(f"✅ Patients segmented by {segment_by}!")
                        
                        # Display segmentation results
                        st.subheader(f"📊 Patient Segmentation by {segment_by}")
                        
                        # Summary table
                        segment_summary = segmentation[segment_by].value_counts()
                        summary_df = pd.DataFrame({
                            'Segment': segment_summary.index,
                            'Patient Count': segment_summary.values,
                            'Percentage': (segment_summary.values / len(segmentation) * 100).round(1)
                        })
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Visualization
                        if len(segment_summary) <= 20:  # Only create chart if not too many segments
                            fig = px.pie(
                                values=segment_summary.values,
                                names=segment_summary.index,
                                title=f"Patient Distribution by {segment_by}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Sample data by segment
                        st.subheader("📋 Sample Patients by Segment")
                        for segment in segment_summary.index[:5]:  # Top 5 segments
                            segment_data = segmentation[segmentation[segment_by] == segment]
                            with st.expander(f"{segment} ({len(segment_data)} patients)"):
                                st.dataframe(segment_data.head(10), use_container_width=True)
                        
                    else:
                        st.error("Unable to generate segmentation.")
                        
                except Exception as e:
                    st.error(f"Error in segmentation: {e}")
    else:
        st.warning("No suitable columns found for segmentation.")
        st.info("Upload data with columns like Disease_Type, Insurance_Type, or Region for segmentation analysis.")

# =============================================================================
# ENHANCED VISUALIZATIONS SECTION
# =============================================================================

def show_enhanced_visualizations_section(df):
    """Enhanced multi-library visualization showcase"""
    
    st.title("🎨 Enhanced Visualizations")
    st.markdown("### Multi-Library Interactive Analytics")
    
    if df is None or df.empty:
        st.warning("Please upload data in the Home section first.")
        return
    
    # Get library availability for functionality
    available_libs = get_available_libraries()
    # Enhanced Visualization Categories
    st.subheader("📊 Available Enhanced Visualizations")
    
    viz_categories = st.tabs([
        "📊 Interactive Charts", 
        "🗺️ Geographic Analysis", 
        "📈 Advanced Dashboards", 
        "📎 Statistical Analysis"
    ])
    
    with viz_categories[0]:  # Interactive Analytics
        st.subheader("📈 Interactive Analytics (Altair)")
        
        if available_libs['altair']:
            
            # Interactive MPR Distribution with Comprehensive Analysis
            if 'MPR' in df.columns:
                st.subheader("🎯 Interactive MPR Distribution")
                
                # Create the visualization
                altair_mpr = create_enhanced_visualization(df, "mpr_distribution", 
                                                         title="Interactive MPR Analysis")
                if altair_mpr:
                    st.altair_chart(altair_mpr, use_container_width=True)
                    
                    # Interactive features info
                    st.info("""
                    **Interactive Features:** 
                    - Zoom and pan with mouse/touch
                    - Hover for detailed patient counts
                    - Click and drag to select ranges
                    - Real-time threshold analysis
                    """)
                    
                    # Generate and display comprehensive analysis summary
                    mpr_summary = generate_mpr_analysis_summary(df)
                    if mpr_summary:
                        display_analysis_summary(mpr_summary, "MPR")
            
            # Interactive Disease Demographics with Analysis
            if 'Disease_Type' in df.columns:
                st.subheader("🏥 Interactive Disease Demographics")
                
                # Create the visualization
                altair_demo = create_enhanced_visualization(df, "disease_demographics")
                if altair_demo:
                    st.altair_chart(altair_demo, use_container_width=True)
                    
                    # Interactive features info
                    st.info("""
                    **Interactive Features:**
                    - Brush selection on disease chart affects age visualization
                    - Cross-filtering between charts
                    - Detailed patient tooltips
                    - Age distribution updates based on disease selection
                    """)
                    
                    # Generate and display comprehensive analysis summary
                    demo_summary = generate_disease_demographics_summary(df)
                    if demo_summary:
                        display_analysis_summary(demo_summary, "Disease Demographics")
        else:
            st.warning("Altair not available. Install with: pip install altair")
    
    with viz_categories[1]:  # Statistical Plots
        st.subheader("📊 Advanced Statistical Analysis (Seaborn)")
        
        if available_libs['seaborn']:
            seaborn_plots = create_enhanced_visualization(df, "statistical_plots")
            
            if seaborn_plots:
                for plot_name, fig in seaborn_plots.items():
                    if plot_name == 'mpr_by_disease':
                        st.subheader("💊 MPR by Disease Type (Box Plot)")
                    elif plot_name == 'age_by_insurance':
                        st.subheader("👥 Age Distribution by Insurance (Violin Plot)")
                    elif plot_name == 'correlation_heatmap':
                        st.subheader("🔗 Healthcare Metrics Correlation")
                    elif plot_name == 'visit_distribution':
                        st.subheader("🏥 Patient Visit Patterns")
                    
                    st.pyplot(fig)
                    
                st.success("✅ Statistical analysis complete with clinical insights")
            else:
                st.info("No statistical plots available. Requires numeric data and disease types.")
        else:
            st.warning("Seaborn not available. Install with: pip install seaborn matplotlib")
    
    with viz_categories[2]:  # Geographic Mapping
        st.subheader("🌍 Geographic Patient Analysis (Folium)")
        
        if available_libs['folium'] and 'Region' in df.columns:
            
            map_type = st.selectbox(
                "Choose Map Type:",
                ["Patient Distribution", "Disease Hotspots"],
                key="map_type_selector"
            )
            
            if map_type == "Patient Distribution":
                st.subheader("📍 Patient Distribution by Region")
                
                # Create the map
                geo_map = create_enhanced_visualization(df, "geographic_map")
                if geo_map:
                    st.components.v1.html(geo_map._repr_html_(), height=600)
                    
                    # Map features info
                    st.info("""
                    **Map Features:**
                    - Circle size = number of patients
                    - Color coding by patient volume
                    - Click markers for detailed regional stats
                    - Average MPR displayed per region
                    """)
                    
                    # Generate and display comprehensive geographic analysis
                    geo_summary = generate_geographic_analysis_summary(df)
                    if geo_summary:
                        display_analysis_summary(geo_summary, "Geographic")
            
            elif map_type == "Disease Hotspots":
                st.subheader("🔥 Disease Hotspot Analysis")
                hotspot_map = create_enhanced_visualization(df, "disease_hotspots")
                if hotspot_map:
                    st.components.v1.html(hotspot_map._repr_html_(), height=600)
                    st.info("""
                    **Hotspot Features:**
                    - Heat intensity = disease concentration
                    - Gradient colors show disease burden
                    - Identify high-prevalence regions
                    """)
        else:
            if not available_libs['folium']:
                st.warning("Folium not available. Install with: pip install folium")
            else:
                st.warning("Geographic analysis requires 'Region' column in your data")
    
    with viz_categories[3]:  # Advanced Dashboards
        st.subheader("🔧 Advanced Interactive Dashboards (Bokeh)")
        
        if available_libs['bokeh']:
            dashboard_type = st.selectbox(
                "Choose Dashboard:",
                ["Adherence Dashboard", "Time Series Analysis"],
                key="bokeh_dashboard_selector"
            )
            
            if dashboard_type == "Adherence Dashboard" and 'MPR' in df.columns:
                st.subheader("💊 Interactive Adherence Dashboard")
                bokeh_dashboard = create_enhanced_visualization(df, "adherence_dashboard")
                if bokeh_dashboard:
                    st.components.v1.html(bokeh_dashboard, height=700)
                    st.info("""
                    **Dashboard Features:**
                    - Interactive MPR histogram with selection tools
                    - Age vs MPR scatter plot with disease color coding
                    - Hover tooltips with patient details
                    - Linked brush selection across charts
                    """)
            
            elif dashboard_type == "Time Series Analysis":
                st.subheader("📈 Time Series Patient Activity Analysis")
                
                bokeh_timeseries = create_enhanced_visualization(df, "time_series")
                if bokeh_timeseries:
                    st.components.v1.html(bokeh_timeseries, height=500)
                    st.info("""
                    **Time Series Features:**
                    - Patient activity trends over time
                    - Interactive zoom and pan
                    - Hover for specific date details
                    - Seasonal pattern identification
                    """)
                else:
                    st.warning("⚠️ Unable to create time series analysis. Please ensure your data contains valid date columns like 'Last_Visit_Date' or 'Date_of_Birth'.")
            else:
                st.info("Additional Bokeh dashboards coming soon!")
        else:
            st.warning("Bokeh not available. Install with: pip install bokeh")
    


# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

def main():
    """Main application logic with proper navigation structure"""
    
    # Create sidebar and get navigation/filters
    section, filters = create_sidebar()
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Apply filters if data exists and not on Home page
    if st.session_state.data is not None and section != "Home":
        if filters:
            filtered_data = filter_data(
                st.session_state.data, 
                filters.get('diseases', []), 
                filters.get('regions', []), 
                filters.get('insurance_types', [])
            )
        else:
            filtered_data = st.session_state.data
    else:
        filtered_data = st.session_state.data
    
    # Navigation Logic - Clean if/elif/else structure
    if section == "Home":
        show_home_section()
    
    elif section == "Cleaned Data Overview":
        show_cleaned_data_section()
    
    elif section == "Demographics Overview":
        show_demographics_section(filtered_data)
    
    elif section == "Adherence Analysis":
        show_adherence_section(filtered_data)
    
    elif section == "Recency Analysis":
        show_recency_section(filtered_data)
    
    elif section == "Advanced Analytics":
        show_advanced_analytics_section(filtered_data)
    
    elif section == "Enhanced Visualizations":
        if ENHANCED_VIZ_AVAILABLE:
            show_enhanced_visualizations_section(filtered_data)
        else:
            st.error("Enhanced visualization libraries not available. Please install: altair, bokeh, seaborn, folium")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Patient360 Analytics v1.0**")
    st.sidebar.markdown("Built with ❤️ for Healthcare Professionals")

if __name__ == "__main__":
    main()
