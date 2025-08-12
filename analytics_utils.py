import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache
import time
from datetime import datetime, timedelta

# Machine Learning imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Visualization imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    sns = None
    plt = None

# =============================================================================
# CORE DATA LOADING AND PROCESSING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(file):
    """Load data with caching and memory optimization"""
    try:
        if file.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Memory optimization
        df = optimize_dataframe_memory(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def clean_data(df, show_report=True):
    """Automatic data cleaning with transparency reporting"""
    if df is None or df.empty:
        return df
    
    original_rows = len(df)
    original_cols = len(df.columns)
    
    cleaning_report = {
        'original_rows': original_rows,
        'original_cols': original_cols,
        'duplicates_removed': 0,
        'rows_dropped_missing_critical': 0,
        'missing_values_filled': {},
        'invalid_values_cleaned': 0
    }
    
    # 1. Remove duplicate rows
    df_clean = df.drop_duplicates()
    cleaning_report['duplicates_removed'] = original_rows - len(df_clean)
    
    # 2. Handle missing values in critical columns
    critical_columns = ['Patient_ID']
    missing_critical = df_clean[critical_columns].isnull().any(axis=1).sum() if any(col in df_clean.columns for col in critical_columns) else 0
    
    # Drop rows with missing critical data
    for col in critical_columns:
        if col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[col])
    
    cleaning_report['rows_dropped_missing_critical'] = missing_critical
    
    # 3. Fill missing values in non-critical columns with reasonable defaults
    fill_rules = {
        'Insurance_Type': 'Unknown',
        'Gender': 'Not Specified',
        'Region': 'Unknown',
        'Comorbidities': 'None Listed',
        'Diagnosis_Stage': 'Not Documented'
    }
    
    for col, fill_value in fill_rules.items():
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                df_clean[col] = df_clean[col].fillna(fill_value)
                cleaning_report['missing_values_filled'][col] = missing_count
    
    # 4. Clean invalid values
    invalid_count = 0
    
    # Clean Age column (remove unrealistic values)
    if 'Age' in df_clean.columns:
        df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
        invalid_ages = ((df_clean['Age'] < 0) | (df_clean['Age'] > 120)).sum()
        df_clean.loc[(df_clean['Age'] < 0) | (df_clean['Age'] > 120), 'Age'] = np.nan
        invalid_count += invalid_ages
    
    # Clean MPR column (ensure it's between 0-100)
    if 'MPR' in df_clean.columns:
        df_clean['MPR'] = pd.to_numeric(df_clean['MPR'], errors='coerce')
        invalid_mpr = ((df_clean['MPR'] < 0) | (df_clean['MPR'] > 100)).sum()
        df_clean['MPR'] = df_clean['MPR'].clip(lower=0, upper=100)
        invalid_count += invalid_mpr
    
    cleaning_report['invalid_values_cleaned'] = invalid_count
    cleaning_report['final_rows'] = len(df_clean)
    cleaning_report['final_cols'] = len(df_clean.columns)
    
    # Display cleaning report
    if show_report and (cleaning_report['duplicates_removed'] > 0 or 
                       cleaning_report['rows_dropped_missing_critical'] > 0 or 
                       cleaning_report['missing_values_filled'] or 
                       cleaning_report['invalid_values_cleaned'] > 0):
        
        st.subheader("🧹 Automatic Data Cleaning Report")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duplicates Removed", cleaning_report['duplicates_removed'])
        with col2:
            st.metric("Critical Missing Dropped", cleaning_report['rows_dropped_missing_critical'])
        with col3:
            filled_count = sum(cleaning_report['missing_values_filled'].values())
            st.metric("Missing Values Filled", filled_count)
        with col4:
            st.metric("Invalid Values Cleaned", cleaning_report['invalid_values_cleaned'])
        
        # Detailed cleaning actions
        if cleaning_report['missing_values_filled']:
            st.info("**Missing Values Filled:**")
            for col, count in cleaning_report['missing_values_filled'].items():
                st.write(f"• {col}: {count:,} missing values filled")
        
        rows_removed = original_rows - cleaning_report['final_rows']
        if rows_removed > 0:
            st.success(f"✅ Data cleaned: {original_rows:,} → {cleaning_report['final_rows']:,} rows ({rows_removed:,} removed)")
        else:
            st.success(f"✅ Data was already clean: {cleaning_report['final_rows']:,} rows")
    
    return df_clean


def calculate_balance_skewness_metrics(df):
    """Calculate and display data balance and skewness metrics"""
    if df is None or df.empty:
        return {}
    
    metrics = {
        'categorical_balance': {},
        'numerical_skewness': {},
        'warnings': []
    }
    
    # 1. Categorical Balance Analysis
    categorical_cols = ['Disease_Type', 'Insurance_Type', 'Region', 'Gender', 'Diagnosis_Stage']
    
    for col in categorical_cols:
        if col in df.columns:
            value_counts = df[col].value_counts(normalize=True) * 100
            max_percentage = value_counts.max()
            
            metrics['categorical_balance'][col] = {
                'max_class_percentage': max_percentage,
                'num_categories': len(value_counts),
                'top_3_categories': value_counts.head(3).to_dict(),
                'balance_status': (
                    'Highly Imbalanced' if max_percentage > 80 else
                    'Moderately Imbalanced' if max_percentage > 60 else
                    'Well Balanced'
                )
            }
            
            # Add warnings for highly imbalanced data
            if max_percentage > 80:
                metrics['warnings'].append(f"⚠️ {col}: {max_percentage:.1f}% of data is a single category")
    
    # 2. Numerical Skewness Analysis
    numerical_cols = ['Age', 'MPR', 'Days_Supplied', 'Total_Days_Supplied']
    
    for col in numerical_cols:
        if col in df.columns:
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(numeric_data) > 0:
                    skewness = numeric_data.skew()
                    
                    metrics['numerical_skewness'][col] = {
                        'skewness': skewness,
                        'interpretation': (
                            'Highly Right Skewed' if skewness > 1 else
                            'Moderately Right Skewed' if skewness > 0.5 else
                            'Highly Left Skewed' if skewness < -1 else
                            'Moderately Left Skewed' if skewness < -0.5 else
                            'Approximately Normal'
                        ),
                        'mean': numeric_data.mean(),
                        'median': numeric_data.median()
                    }
                    
                    # Add warnings for highly skewed data
                    if abs(skewness) > 1:
                        direction = 'right' if skewness > 0 else 'left'
                        metrics['warnings'].append(f"⚠️ {col}: Highly {direction}-skewed distribution (skewness: {skewness:.2f})")
            except Exception:
                continue
    
    return metrics


def display_balance_skewness_dashboard(metrics):
    """Display comprehensive balance and skewness dashboard"""
    if not metrics:
        return
    
    st.subheader("⚖️ Data Balance & Skewness Analysis")
    
    # Display warnings first if any
    if metrics.get('warnings'):
        st.warning("**Data Quality Alerts:**")
        for warning in metrics['warnings']:
            st.write(warning)
        st.markdown("---")
    
    # Categorical Balance Analysis
    if metrics.get('categorical_balance'):
        st.markdown("### 📊 Categorical Data Balance")
        
        balance_data = []
        for col, data in metrics['categorical_balance'].items():
            balance_data.append({
                'Column': col,
                'Categories': data['num_categories'],
                'Max Class %': f"{data['max_class_percentage']:.1f}%",
                'Balance Status': data['balance_status']
            })
        
        balance_df = pd.DataFrame(balance_data)
        st.dataframe(balance_df, use_container_width=True)
        
        # Show top categories for each column
        for col, data in metrics['categorical_balance'].items():
            with st.expander(f"📈 {col} Distribution"):
                top_cats = data['top_3_categories']
                for category, percentage in top_cats.items():
                    st.write(f"• **{category}**: {percentage:.1f}%")
    
    # Numerical Skewness Analysis
    if metrics.get('numerical_skewness'):
        st.markdown("### 📈 Numerical Data Skewness")
        
        skew_data = []
        for col, data in metrics['numerical_skewness'].items():
            skew_data.append({
                'Column': col,
                'Skewness': f"{data['skewness']:.3f}",
                'Interpretation': data['interpretation'],
                'Mean': f"{data['mean']:.2f}",
                'Median': f"{data['median']:.2f}"
            })
        
        skew_df = pd.DataFrame(skew_data)
        st.dataframe(skew_df, use_container_width=True)
        
        # Clinical context
        st.info("""
        **Clinical Context for Skewness:**
        - **Age**: Slight right skew is normal (more younger patients)
        - **MPR**: Left skew indicates most patients are adherent (good)
        - **Days_Supplied**: Right skew suggests varied prescription lengths
        """)

@st.cache_data
def load_and_process_large_dataset(file_path):
    """Load and perform initial processing with caching"""
    df = load_data(file_path)
    
    if df is not None:
        # Initial data quality assessment
        quality_report = generate_data_quality_report(df)
        return df, quality_report
    
    return None, None

def filter_data(df, diseases, regions, insurance_types):
    """Filter data with error handling and transparency"""
    try:
        if df is None or df.empty:
            return df
        
        # Create filter conditions with null-safe operations
        conditions = []
        
        if diseases and 'Disease_Type' in df.columns:
            conditions.append(df['Disease_Type'].isin(diseases))
        
        if regions and 'Region' in df.columns:
            conditions.append(df['Region'].isin(regions))
        
        if insurance_types and 'Insurance_Type' in df.columns:
            conditions.append(df['Insurance_Type'].isin(insurance_types))
        
        # Apply all conditions
        if conditions:
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition & condition
            
            filtered_df = df[combined_condition]
        else:
            filtered_df = df
        
        # Show filtering transparency
        if len(filtered_df) != len(df):
            st.info(f"Filtered from {len(df):,} to {len(filtered_df):,} records based on selected criteria")
        
        return filtered_df
        
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        return df

# =============================================================================
# MPR CALCULATION WITH TRANSPARENCY AND ERROR HANDLING
# =============================================================================

@st.cache_data
def calculate_mpr(df, show_transparency=True):
    """Calculate MPR with transparency and robust error handling"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Data cleaning transparency
        if show_transparency:
            st.subheader("📋 MPR Calculation Methodology")
            st.info("""
            **MPR (Medication Possession Ratio) Calculation:**
            - MPR = (Total Days Supplied / Days in Treatment Period) × 100
            - Treatment Period = Last Prescription Date - First Prescription Date + 1
            - Adherent patients typically have MPR ≥ 80%
            """)
        
        # Check required columns
        required_columns = ['Patient_ID', 'Prescription_Date', 'Days_Supplied']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns for MPR calculation: {missing_columns}")
            return pd.DataFrame()
        
        # Data cleaning steps
        original_count = len(df)
        
        # Convert Prescription_Date to datetime with error handling
        df['Prescription_Date'] = pd.to_datetime(df['Prescription_Date'], errors='coerce')
        
        # Remove records with invalid dates
        df = df.dropna(subset=['Prescription_Date'])
        invalid_dates = original_count - len(df)
        
        # Remove records with invalid Days_Supplied
        df = df[(df['Days_Supplied'] > 0) & (df['Days_Supplied'] <= 365)]
        invalid_days = original_count - invalid_dates - len(df)
        
        if show_transparency and (invalid_dates > 0 or invalid_days > 0):
            st.warning(f"⚠️ Data Cleaning Applied:")
            if invalid_dates > 0:
                st.write(f"• Removed {invalid_dates} records with invalid prescription dates")
            if invalid_days > 0:
                st.write(f"• Removed {invalid_days} records with invalid days supplied (≤0 or >365)")
        
        if df.empty:
            st.error("No valid data remaining after cleaning for MPR calculation")
            return pd.DataFrame()
        
        # Calculate MPR
        mpr_df = df.groupby('Patient_ID').agg({
            'Days_Supplied': 'sum',
            'Prescription_Date': ['min', 'max', 'count']
        })
        
        mpr_df.columns = ['Total_Days_Supplied', 'First_Prescription', 'Last_Prescription', 'Prescription_Count']
        
        # Calculate treatment period
        mpr_df['Days_In_Period'] = (
            pd.to_datetime(mpr_df['Last_Prescription']) - pd.to_datetime(mpr_df['First_Prescription'])
        ).dt.days + 1
        
        # Handle single prescription cases
        mpr_df.loc[mpr_df['Prescription_Count'] == 1, 'Days_In_Period'] = mpr_df.loc[mpr_df['Prescription_Count'] == 1, 'Total_Days_Supplied']
        
        # Calculate MPR
        mpr_df['MPR'] = (mpr_df['Total_Days_Supplied'] / mpr_df['Days_In_Period']) * 100
        
        # Cap MPR at 100% (oversupply cases)
        mpr_df['MPR'] = mpr_df['MPR'].clip(upper=100)
        
        # Add adherence classification
        mpr_df['Adherent'] = mpr_df['MPR'] >= 80
        
        mpr_df = mpr_df.reset_index()
        
        if show_transparency:
            adherent_count = mpr_df['Adherent'].sum()
            adherence_rate = (adherent_count / len(mpr_df)) * 100
            st.success(f"✅ MPR calculated for {len(mpr_df):,} patients. Adherence rate: {adherence_rate:.1f}%")
        
        return mpr_df
        
    except Exception as e:
        st.error(f"Error calculating MPR: {e}")
        return pd.DataFrame()

@st.cache_data
def calculate_mpr_cached(df):
    """Cached MPR calculation for large datasets"""
    return calculate_mpr(df, show_transparency=False)

def flag_overdue_visits(df, days=180, show_transparency=True):
    """Flag overdue visits with transparency and error handling"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        if show_transparency:
            st.subheader("🗓️ Care Gap Analysis Methodology")
            st.info(f"""
            **Overdue Visit Criteria:**
            - Patients with last visit > {days} days ago are flagged as overdue
            - Based on clinical guidelines for care continuity
            - Missing visit dates are handled separately
            """)
        
        # Check for required column
        if 'Last_Visit_Date' not in df.columns:
            st.error("Last_Visit_Date column not found for recency analysis")
            return pd.DataFrame()
        
        # Data cleaning
        original_count = len(df)
        
        # Convert Last_Visit_Date to datetime with error handling
        df['Last_Visit_Date'] = pd.to_datetime(df['Last_Visit_Date'], errors='coerce')
        
        # Track missing visit dates
        missing_visits = df['Last_Visit_Date'].isna().sum()
        
        # Calculate days since last visit
        today = pd.Timestamp.today()
        df['Days_Since_Last_Visit'] = (today - df['Last_Visit_Date']).dt.days
        
        # Flag overdue visits
        overdue = df[df['Days_Since_Last_Visit'] > days]
        
        if show_transparency:
            if missing_visits > 0:
                st.warning(f"⚠️ {missing_visits} patients have missing visit dates")
            
            overdue_count = len(overdue)
            overdue_rate = (overdue_count / (len(df) - missing_visits)) * 100 if (len(df) - missing_visits) > 0 else 0
            st.info(f"📊 {overdue_count:,} patients ({overdue_rate:.1f}%) are overdue for visits")
        
        return overdue
        
    except Exception as e:
        st.error(f"Error flagging overdue visits: {e}")
        return pd.DataFrame()

# =============================================================================
# PERFORMANCE OPTIMIZATION FUNCTIONS
# =============================================================================

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage for large datasets"""
    if df is None or df.empty:
        return df
    
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Convert object columns to category where appropriate
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
            df[col] = df[col].astype('category')
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype('int32')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    if original_memory > 0:
        reduction = ((original_memory - final_memory) / original_memory * 100)
        if reduction > 5:  # Only show if significant reduction
            st.info(f"🔧 Memory optimized: {original_memory:.1f}MB → {final_memory:.1f}MB ({reduction:.1f}% reduction)")
    
    return df

def get_sample_for_visualization(df, max_points=5000):
    """Smart sampling for visualization performance"""
    if df is None or len(df) <= max_points:
        return df
    
    # Stratified sampling to maintain distribution
    try:
        if 'Disease_Type' in df.columns and 'Insurance_Type' in df.columns:
            sample_df = df.groupby(['Disease_Type', 'Insurance_Type'], group_keys=False)\
                          .apply(lambda x: x.sample(min(len(x), max(1, max_points//20))))\
                          .reset_index(drop=True)
        else:
            # Simple random sampling if stratification columns don't exist
            sample_df = df.sample(n=max_points, random_state=42)
        
        st.info(f"📊 Displaying sample of {len(sample_df):,} records (from {len(df):,} total) for performance")
        return sample_df
        
    except Exception as e:
        # Fallback to simple random sampling
        sample_df = df.sample(n=min(max_points, len(df)), random_state=42)
        st.info(f"📊 Displaying random sample of {len(sample_df):,} records for performance")
        return sample_df

# =============================================================================
# DATA QUALITY AND TRANSPARENCY FUNCTIONS
# =============================================================================

@st.cache_data
def generate_data_quality_report(df):
    """Comprehensive data quality report optimized for large datasets"""
    if df is None or df.empty:
        return {}
    
    quality_report = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': {},
        'duplicates': {},
        'outliers': {},
        'data_types': {},
        'date_issues': {}
    }
    
    # Missing values analysis
    missing_stats = df.isnull().sum()
    quality_report['missing_values'] = {
        col: {
            'count': int(missing_stats[col]),
            'percentage': float(missing_stats[col] / len(df) * 100)
        }
        for col in df.columns if missing_stats[col] > 0
    }
    
    # Duplicate analysis
    duplicate_count = df.duplicated().sum()
    duplicate_subset_patient = df.duplicated(subset=['Patient_ID']).sum() if 'Patient_ID' in df.columns else 0
    
    quality_report['duplicates'] = {
        'total_duplicates': int(duplicate_count),
        'duplicate_patients': int(duplicate_subset_patient),
        'percentage': float(duplicate_count / len(df) * 100)
    }
    
    # Data type validation
    for col in df.columns:
        quality_report['data_types'][col] = {
            'dtype': str(df[col].dtype),
            'unique_values': int(df[col].nunique()),
            'most_frequent': str(df[col].mode().iloc[0]) if not df[col].mode().empty else 'N/A'
        }
    
    return quality_report

def calculate_overall_quality_score(df):
    """Calculate overall data quality score (0-100)"""
    if df is None or df.empty:
        return 0
    
    # Completeness score (60% weight)
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    # Uniqueness score (20% weight) - based on duplicates
    uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
    
    # Validity score (20% weight) - basic data type validation
    validity = calculate_validity_score(df)
    
    overall_score = (completeness * 0.6) + (uniqueness * 0.2) + (validity * 0.2)
    return overall_score

def calculate_validity_score(df):
    """Calculate validity score based on data type consistency"""
    if df is None or df.empty:
        return 0
    
    validity_issues = 0
    total_checks = 0
    
    # Check age values
    if 'Age' in df.columns:
        invalid_ages = df[(df['Age'] < 0) | (df['Age'] > 120)]['Age'].count()
        validity_issues += invalid_ages
        total_checks += len(df['Age'].dropna())
    
    # Check date formats (if any date columns exist)
    date_columns = [col for col in df.columns if 'Date' in col]
    for col in date_columns:
        try:
            pd.to_datetime(df[col], errors='coerce')
            invalid_dates = df[col][pd.to_datetime(df[col], errors='coerce').isna() & df[col].notna()]
            validity_issues += len(invalid_dates)
        except:
            validity_issues += df[col].count()
        total_checks += df[col].count()
    
    if total_checks == 0:
        return 100
    
    validity_score = (1 - validity_issues / total_checks) * 100
    return max(0, validity_score)

def generate_data_quality_dashboard(df):
    """Generate comprehensive data quality dashboard for healthcare professionals"""
    if df is None or df.empty:
        st.error("No data available for quality assessment")
        return
    
    st.subheader("🔍 Data Quality Dashboard")
    
    # Overall Quality Score
    quality_score = calculate_overall_quality_score(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score_delta = "✅ Excellent" if quality_score > 80 else "⚠️ Needs Review" if quality_score > 60 else "❌ Poor Quality"
        st.metric("Overall Quality Score", f"{quality_score:.1f}%", delta=score_delta)
    with col2:
        st.metric("Total Records", f"{len(df):,}")
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with col4:
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        st.metric("Duplicate Records", f"{duplicate_pct:.1f}%")
    
    # Data Quality Summary Table
    st.subheader("📊 Data Quality Summary by Column")
    
    quality_summary = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        
        quality_summary.append({
            'Column': col,
            'Missing Count': missing_count,
            'Missing %': f"{missing_pct:.1f}%",
            'Unique Values': unique_count,
            'Uniqueness %': f"{unique_pct:.1f}%",
            'Data Type': str(df[col].dtype)
        })
    
    quality_df = pd.DataFrame(quality_summary)
    st.dataframe(quality_df, use_container_width=True)
    
    # Clinical Context
    st.info("""
    **Clinical Data Quality Context:**
    - **Patient_ID, Age**: Should have 0% missing (core identifiers)
    - **Insurance_Type**: 5-10% missing is normal (self-pay patients)
    - **Last_Visit_Date**: 15-25% missing is common (new patients, scheduling gaps)
    - **Comorbidities**: High missing rates (20-30%) are typical in EHR systems
    - **Diagnosis_Stage**: Often missing (not always documented)
    """)

def create_missing_value_heatmap(df):
    """Create missing value heatmap visualization"""
    if df is None or df.empty:
        return
    
    # Sample data if too large for visualization
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
        st.info("Showing heatmap for 1,000 sample records for performance")
    else:
        df_sample = df
    
    # Create heatmap using matplotlib/seaborn if available
    if sns is not None and plt is not None:
        missing_data = df_sample.isnull()
        
        if missing_data.any().any():
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(missing_data.T, cbar=True, cmap='viridis', 
                       yticklabels=True, xticklabels=False, ax=ax)
            ax.set_title('Missing Value Heatmap (Yellow = Missing)')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No missing values to visualize")
    else:
        # Fallback to text-based missing value summary
        missing_summary = df_sample.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        
        if not missing_summary.empty:
            st.write("Missing values by column:")
            for col, missing_count in missing_summary.items():
                missing_pct = (missing_count / len(df_sample)) * 100
                st.write(f"- {col}: {missing_count} ({missing_pct:.1f}%)")
        else:
            st.success("✅ No missing values detected!")

# --- Advanced Analytics ---
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

def detect_outliers(df):
    """Enhanced outlier detection for multiple numeric columns"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        outlier_indices = set()
        outlier_reasons = {}
        
        # Check numeric columns for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add Age if it exists but isn't numeric
        if 'Age' in df.columns and 'Age' not in numeric_cols:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            if not df['Age'].isna().all():
                numeric_cols.append('Age')
        
        if not numeric_cols:
            return pd.DataFrame({'Info': ['No numeric columns found for outlier detection']})
        
        for col in numeric_cols:
            if df[col].dtype in ['int64', 'float64'] and not df[col].isna().all():
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                    outlier_indices.update(col_outliers)
                    
                    # Track reasons
                    for idx in col_outliers:
                        if idx not in outlier_reasons:
                            outlier_reasons[idx] = []
                        
                        value = df.loc[idx, col]
                        if value < lower_bound:
                            outlier_reasons[idx].append(f"{col}: {value:.1f} (below {lower_bound:.1f})")
                        else:
                            outlier_reasons[idx].append(f"{col}: {value:.1f} (above {upper_bound:.1f})")
        
        # Domain-specific outlier rules for healthcare data
        if 'Age' in df.columns:
            age_outliers = df[(df['Age'] < 0) | (df['Age'] > 120)].index
            outlier_indices.update(age_outliers)
            for idx in age_outliers:
                if idx not in outlier_reasons:
                    outlier_reasons[idx] = []
                outlier_reasons[idx].append(f"Age: {df.loc[idx, 'Age']} (impossible age)")
        
        if not outlier_indices:
            return pd.DataFrame({'Info': ['No outliers detected using IQR method']})
        
        # Create outlier DataFrame
        outliers = df.loc[list(outlier_indices)].copy()
        outliers['Outlier_Reasons'] = ['; '.join(outlier_reasons.get(idx, [])) for idx in outliers.index]
        
        # Add severity score
        outliers['Outlier_Severity'] = outliers['Outlier_Reasons'].str.count(';') + 1
        
        return outliers.sort_values('Outlier_Severity', ascending=False)
        
    except Exception as e:
        st.error(f"Outlier detection error: {e}")
        return pd.DataFrame({'Error': [str(e)]})

def patient_persona_clustering(df, n_clusters=3):
    """Enhanced patient persona clustering with robust error handling"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Check required columns and create feature matrix
        features = []
        feature_names = []
        
        # Age feature (essential)
        if 'Age' in df.columns:
            age_clean = pd.to_numeric(df['Age'], errors='coerce')
            age_clean = age_clean.fillna(age_clean.median())
            features.append(age_clean.values.reshape(-1, 1))
            feature_names.append('Age')
        
        # MPR feature: use existing column if present, else calculate
        mpr_added = False
        mpr_df = None
        
        if 'MPR' in df.columns:
            mpr_clean = pd.to_numeric(df['MPR'], errors='coerce').fillna(df['MPR'].median())
            features.append(mpr_clean.values.reshape(-1, 1))
            feature_names.append('MPR')
            mpr_added = True
            # Create mpr_df for later use
            mpr_df = df[['Patient_ID', 'MPR']].copy()
        elif all(col in df.columns for col in ['Patient_ID', 'Prescription_Date', 'Days_Supplied']):
            try:
                mpr_df = calculate_mpr(df, show_transparency=False)
                if not mpr_df.empty:
                    df_with_mpr = df.merge(mpr_df[['Patient_ID', 'MPR']], on='Patient_ID', how='left')
                    mpr_clean = df_with_mpr['MPR'].fillna(df_with_mpr['MPR'].median())
                    features.append(mpr_clean.values.reshape(-1, 1))
                    feature_names.append('MPR')
                    mpr_added = True
            except:
                pass  # Skip MPR if calculation fails
        
        # Additional categorical features
        categorical_cols = ['Gender', 'Insurance_Type', 'Region']
        for col in categorical_cols:
            if col in df.columns:
                # One-hot encode categorical variables
                col_encoded = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                if not col_encoded.empty:
                    features.append(col_encoded.values)
                    feature_names.extend([f"{col}_{c}" for c in col_encoded.columns])
        
        # Combine all features
        if not features:
            return pd.DataFrame({'Error': ['No suitable features found for clustering']})
        
        # Concatenate features
        X = np.concatenate(features, axis=1)
        
        # Handle edge cases
        if len(X) < n_clusters:
            n_clusters = min(2, len(X))
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create result DataFrame
        result_df = df[['Patient_ID']].copy()
        if 'Age' in df.columns:
            result_df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        if mpr_added and mpr_df is not None:
            if 'MPR' not in result_df.columns:
                result_df = result_df.merge(mpr_df[['Patient_ID', 'MPR']], on='Patient_ID', how='left')
            else:
                # MPR already exists in result_df, ensure it's numeric
                result_df['MPR'] = pd.to_numeric(result_df['MPR'], errors='coerce')
        
        result_df['Cluster'] = clusters
        
        # Add cluster centers interpretation
        result_df['Cluster_Description'] = result_df['Cluster'].map({
            0: 'Low Risk Profile',
            1: 'Moderate Risk Profile', 
            2: 'High Risk Profile',
            3: 'Complex Care Profile',
            4: 'Emerging Risk Profile'
        })
        
        return result_df
        
    except Exception as e:
        st.error(f"Clustering error: {e}")
        return pd.DataFrame({'Error': [str(e)]})

def detect_outliers(df):
    """Enhanced outlier detection for multiple numeric columns"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        outlier_indices = set()
        outlier_reasons = {}
        
        # Check numeric columns for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add Age if it exists but isn't numeric
        if 'Age' in df.columns and 'Age' not in numeric_cols:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            if not df['Age'].isna().all():
                numeric_cols.append('Age')
        
        if not numeric_cols:
            return pd.DataFrame({'Info': ['No numeric columns found for outlier detection']})
        
        for col in numeric_cols:
            if df[col].dtype in ['int64', 'float64'] and not df[col].isna().all():
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                    outlier_indices.update(col_outliers)
                    
                    # Track reasons
                    for idx in col_outliers:
                        if idx not in outlier_reasons:
                            outlier_reasons[idx] = []
                        
                        value = df.loc[idx, col]
                        if value < lower_bound:
                            outlier_reasons[idx].append(f"{col}: {value:.1f} (below {lower_bound:.1f})")
                        else:
                            outlier_reasons[idx].append(f"{col}: {value:.1f} (above {upper_bound:.1f})")
        
        # Domain-specific outlier rules for healthcare data
        if 'Age' in df.columns:
            age_outliers = df[(df['Age'] < 0) | (df['Age'] > 120)].index
            outlier_indices.update(age_outliers)
            for idx in age_outliers:
                if idx not in outlier_reasons:
                    outlier_reasons[idx] = []
                outlier_reasons[idx].append(f"Age: {df.loc[idx, 'Age']} (impossible age)")
        
        if not outlier_indices:
            return pd.DataFrame({'Info': ['No outliers detected using IQR method']})
        
        # Create outlier DataFrame
        outliers = df.loc[list(outlier_indices)].copy()
        outliers['Outlier_Reasons'] = ['; '.join(outlier_reasons.get(idx, [])) for idx in outliers.index]
        
        # Add severity score
        outliers['Outlier_Severity'] = outliers['Outlier_Reasons'].str.count(';') + 1
        
        return outliers.sort_values('Outlier_Severity', ascending=False)
        
    except Exception as e:
        st.error(f"Outlier detection error: {e}")
        return pd.DataFrame({'Error': [str(e)]})

def adherence_prediction(df):
    """Enhanced adherence risk prediction with robust feature engineering"""
    try:
        if df is None or df.empty:
            return []
        
        # Use existing MPR if present
        if 'MPR' in df.columns:
            mpr_df = df[['Patient_ID', 'MPR']].copy()
        elif all(col in df.columns for col in ['Patient_ID', 'Prescription_Date', 'Days_Supplied']):
            mpr_df = calculate_mpr(df, show_transparency=False)
            if mpr_df.empty:
                return predict_adherence_from_visits(df)
        else:
            # Alternative: Use visit recency as proxy for adherence risk
            return predict_adherence_from_visits(df)
        
        # Prepare features for prediction
        feature_cols = ['Age', 'Gender', 'Region', 'Insurance_Type', 'Disease_Type']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            # Fallback: simple MPR-based risk assessment
            return simple_mpr_risk_assessment(mpr_df)
        
        # Merge features with MPR
        feature_df = df[['Patient_ID'] + available_cols].drop_duplicates(subset=['Patient_ID'])
        merged = mpr_df.merge(feature_df, on='Patient_ID', how='left')
        
        # Create target variable
        merged['Non_Adherent'] = (merged['MPR'] < 80).astype(int)
        
        # Feature engineering
        X_features = []
        feature_names = []
        
        # Age feature
        if 'Age' in merged.columns:
            age_clean = pd.to_numeric(merged['Age'], errors='coerce').fillna(merged['Age'].median())
            X_features.append(age_clean.values.reshape(-1, 1))
            feature_names.append('Age')
        
        # Categorical features
        for col in ['Gender', 'Region', 'Insurance_Type', 'Disease_Type']:
            if col in merged.columns:
                # Create dummy variables
                dummies = pd.get_dummies(merged[col], prefix=col, dummy_na=True)
                X_features.append(dummies.values)
                feature_names.extend([f"{col}_{c}" for c in dummies.columns])
        
        if not X_features:
            return simple_mpr_risk_assessment(mpr_df)
        
        # Combine features
        X = np.concatenate(X_features, axis=1)
        y = merged['Non_Adherent'].values
        
        # Check if we have both classes
        if len(np.unique(y)) < 2:
            # Only one class - assign risk based on MPR distribution
            risk_scores = []
            for _, row in merged.iterrows():
                mpr = row['MPR']
                
                if mpr >= 90:
                    risk = 0.1
                elif mpr >= 80:
                    risk = 0.3
                elif mpr >= 60:
                    risk = 0.6
                else:
                    risk = 0.9
                
                risk_scores.append({
                    'patient_id': row['Patient_ID'],
                    'mpr': mpr,
                    'risk_score': risk,
                    'risk_category': 'High' if risk > 0.7 else 'Medium' if risk > 0.3 else 'Low'
                })
            return risk_scores
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Get prediction probabilities
        risk_probabilities = model.predict_proba(X)[:, 1]
        
        # Create results
        results = []
        for i, (_, row) in enumerate(merged.iterrows()):
            risk_score = risk_probabilities[i]
            results.append({
                'patient_id': row['Patient_ID'],
                'mpr': row['MPR'],
                'risk_score': risk_score,
                'risk_category': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low',
                'current_adherent': row['MPR'] >= 80
            })
        
        return results
        
    except Exception as e:
        st.error(f"Adherence prediction error: {e}")
        return [{'error': str(e)}]

def predict_adherence_from_visits(df):
    """Fallback prediction using visit patterns"""
    if 'Last_Visit_Date' not in df.columns:
        return [{'error': 'Insufficient data for adherence prediction'}]
    
    df_copy = df.copy()
    df_copy['Last_Visit_Date'] = pd.to_datetime(df_copy['Last_Visit_Date'], errors='coerce')
    df_copy['Days_Since_Visit'] = (pd.Timestamp.today() - df_copy['Last_Visit_Date']).dt.days
    
    results = []
    for _, row in df_copy.iterrows():
        days_since = row.get('Days_Since_Visit', 999)
        
        if pd.isna(days_since):
            risk_score = 0.8
        elif days_since > 365:
            risk_score = 0.9
        elif days_since > 180:
            risk_score = 0.7
        elif days_since > 90:
            risk_score = 0.4
        else:
            risk_score = 0.2
        
        results.append({
            'patient_id': row['Patient_ID'],
            'days_since_visit': days_since,
            'risk_score': risk_score,
            'risk_category': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low'
        })
    
    return results

def simple_mpr_risk_assessment(mpr_df):
    """Simple risk assessment based only on MPR"""
    results = []
    for _, row in mpr_df.iterrows():
        mpr = row['MPR']
        
        if mpr >= 90:
            risk_score = 0.1
        elif mpr >= 80:
            risk_score = 0.3
        elif mpr >= 60:
            risk_score = 0.6
        else:
            risk_score = 0.9
        
        results.append({
            'patient_id': row['Patient_ID'],
            'mpr': mpr,
            'risk_score': risk_score,
            'risk_category': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low'
        })
    
    return results

def cohort_trends(df):
    """Enhanced cohort trends analysis with multiple metrics"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Try prescription date first
        date_col = None
        if 'Prescription_Date' in df.columns:
            date_col = 'Prescription_Date'
        elif 'Last_Visit_Date' in df.columns:
            date_col = 'Last_Visit_Date'
        else:
            return pd.DataFrame({'Info': ['No date columns found for cohort analysis']})
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        if df.empty:
            return pd.DataFrame({'Info': ['No valid dates found for cohort analysis']})
        
        # Create cohort periods
        df['YearMonth'] = df[date_col].dt.to_period('M')
        df['Quarter'] = df[date_col].dt.to_period('Q')
        df['Year'] = df[date_col].dt.year
        
        # Analyze by different time periods
        trends = []
        
        # Monthly trends
        monthly = df.groupby('YearMonth').agg({
            'Patient_ID': 'nunique',
            'Age': 'mean' if 'Age' in df.columns else lambda x: None
        }).reset_index()
        monthly['Period_Type'] = 'Monthly'
        monthly['Cohort'] = monthly['YearMonth'].astype(str)
        monthly = monthly.rename(columns={'Patient_ID': 'Patient_Count'})
        
        # If we have prescription data, add MPR trends
        if date_col == 'Prescription_Date' and all(col in df.columns for col in ['Patient_ID', 'Days_Supplied']):
            try:
                mpr_df = calculate_mpr(df, show_transparency=False)
                if not mpr_df.empty:
                    df_with_mpr = df.merge(mpr_df[['Patient_ID', 'MPR']], on='Patient_ID', how='left')
                    mpr_monthly = df_with_mpr.groupby('YearMonth').agg({
                        'MPR': 'mean',
                        'Patient_ID': 'nunique'
                    }).reset_index()
                    mpr_monthly['Cohort'] = mpr_monthly['YearMonth'].astype(str)
                    mpr_monthly = mpr_monthly.rename(columns={'Patient_ID': 'Patient_Count'})
                    
                    # Combine with main trends
                    monthly = monthly.merge(mpr_monthly[['Cohort', 'MPR']], on='Cohort', how='left')
            except:
                pass
        
        trends.append(monthly)
        
        # Quarterly trends
        quarterly = df.groupby('Quarter').agg({
            'Patient_ID': 'nunique',
            'Age': 'mean' if 'Age' in df.columns else lambda x: None
        }).reset_index()
        quarterly['Period_Type'] = 'Quarterly'
        quarterly['Cohort'] = quarterly['Quarter'].astype(str)
        quarterly = quarterly.rename(columns={'Patient_ID': 'Patient_Count'})
        trends.append(quarterly)
        
        # Combine all trends
        final_trends = pd.concat(trends, ignore_index=True)
        
        # Add trend indicators
        for period_type in final_trends['Period_Type'].unique():
            period_data = final_trends[final_trends['Period_Type'] == period_type]
            if len(period_data) > 1:
                # Calculate growth rates
                period_data = period_data.sort_values('Cohort')
                period_data['Patient_Growth'] = period_data['Patient_Count'].pct_change() * 100
                final_trends.loc[final_trends['Period_Type'] == period_type, 'Patient_Growth'] = period_data['Patient_Growth']
        
        return final_trends
        
    except Exception as e:
        st.error(f"Cohort trends error: {e}")
        return pd.DataFrame({'Error': [str(e)]})

def custom_segmentation(df, by='Disease_Type'):
    """Enhanced custom segmentation with comprehensive metrics"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        if by not in df.columns:
            available_cols = [col for col in df.columns if col in ['Disease_Type', 'Insurance_Type', 'Region', 'Gender', 'Comorbidities']]
            if available_cols:
                by = available_cols[0]
            else:
                return pd.DataFrame({'Info': ['No suitable segmentation columns found']})
        
        # Basic segmentation
        seg = df.groupby(by).agg({
            'Patient_ID': 'nunique'
        }).reset_index().rename(columns={'Patient_ID': 'Patient_Count'})
        
        # Add percentage
        total_patients = seg['Patient_Count'].sum()
        seg['Percentage'] = (seg['Patient_Count'] / total_patients * 100).round(1)
        
        # Add additional metrics if available
        additional_metrics = {}
        
        if 'Age' in df.columns:
            age_stats = df.groupby(by)['Age'].agg(['mean', 'median']).round(1)
            seg = seg.merge(age_stats, left_on=by, right_index=True, how='left')
            seg = seg.rename(columns={'mean': 'Avg_Age', 'median': 'Median_Age'})
        
        # Add gender distribution if available
        if 'Gender' in df.columns and by != 'Gender':
            gender_dist = df.groupby([by, 'Gender']).size().unstack(fill_value=0)
            gender_pct = gender_dist.div(gender_dist.sum(axis=1), axis=0) * 100
            
            # Add most common gender for each segment
            most_common_gender = gender_pct.idxmax(axis=1)
            seg = seg.merge(most_common_gender.rename('Dominant_Gender'), left_on=by, right_index=True, how='left')
        
        # Add insurance distribution if available
        if 'Insurance_Type' in df.columns and by != 'Insurance_Type':
            insurance_dist = df.groupby(by)['Insurance_Type'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
            seg = seg.merge(insurance_dist.rename('Common_Insurance'), left_on=by, right_index=True, how='left')
        
        # Sort by patient count
        seg = seg.sort_values('Patient_Count', ascending=False)
        
        # Add rank
        seg['Rank'] = range(1, len(seg) + 1)
        
        return seg
        
    except Exception as e:
        st.error(f"Segmentation error: {e}")
        return pd.DataFrame({'Error': [str(e)]})