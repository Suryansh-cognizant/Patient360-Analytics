# 🏥 Patient360 Analytics Dashboard

> **Comprehensive Healthcare Analytics Platform with Data Quality Transparency**

A powerful, interactive healthcare analytics dashboard built with Streamlit that provides comprehensive patient data analysis, medication adherence tracking, and advanced healthcare insights with complete data quality transparency.

---

## 🚀 Live Application

**Access the deployed app here:**  
👉 [https://patient360-analytics.streamlit.app/](https://patient360-analytics.streamlit.app/)

The application is deployed using **Streamlit Cloud** and the source code is managed via **GitHub**.

---

## ✨ Key Features

- **Automatic Data Cleaning**: Removes duplicates, handles missing values, validates data types
- **Comprehensive Quality Dashboard**: Real-time data quality metrics and reports
- **Demographics Analysis**: Age, gender, region, disease type, and insurance insights
- **Medication Adherence (MPR) Analysis**: Interactive visualizations and clinical insights
- **Care Recency & Gap Analysis**: Identifies patients overdue for care
- **Advanced Analytics**: Clustering, predictive analytics, and correlation analysis
- **Enhanced Visualizations**: Plotly, Altair, Bokeh, Seaborn, Folium integration
- **Export Capabilities**: Download cleaned data and analysis results

---

## 📋 Getting Started

1. **Go to the app**: [https://patient360-analytics.streamlit.app/](https://patient360-analytics.streamlit.app/)
2. **Download the sample data** from the Home section
3. **Upload your own data** or use the sample to explore all features
4. **Navigate using the sidebar** to explore analytics, visualizations, and export options

---

## 📁 Project Structure

```
patient360-analytics/
├── app.py                      # Main Streamlit application
├── analytics_utils.py          # Core analytics functions
├── visualization_utils.py      # Enhanced visualization utilities  
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── sample_data_10k.csv         # Sample healthcare dataset
├── .gitignore                  # Git ignore rules
└── data/                       # Additional data files (optional)
    └── sample_data_10k.csv     # Backup location for sample data
```

---

## 📊 Expected Data Format

Your healthcare data should include these columns (case-sensitive):

| Column Name            | Description                       | Example                 |
|------------------------|-----------------------------------|-------------------------|
| `Patient_ID`           | Unique patient identifier         | PT000001                |
| `Age`                  | Patient age in years              | 45                      |
| `Gender`               | Patient gender                    | Male/Female             |
| `Disease_Type`         | Primary medical condition         | Hypertension            |
| `Comorbidities`        | Additional conditions             | Diabetes; Heart Disease |
| `Insurance_Type`       | Insurance coverage                | Medicare/Commercial     |
| `Region`               | Geographic location               | California/New York     |
| `MPR`                  | Medication Possession Ratio        | 85.5                    |
| `Last_Visit_Date`      | Date of last visit                | 2024-01-15              |
| `Total_Days_Supplied`  | Days of medication supplied       | 90                      |
| `Prescription_Count`   | Number of prescriptions           | 3                       |

**Optional columns**: `First_Name`, `Last_Name`, `Date_of_Birth`, `Total_Visits`, `Most_Common_Visit_Type`

---

## 🛠️ Technical Stack

| Category             | Technologies                                    |
|----------------------|-------------------------------------------------|
| **Frontend**         | Streamlit, HTML, CSS, JavaScript                |
| **Data Processing**  | Pandas, NumPy                                   |
| **Visualizations**   | Plotly, Altair, Bokeh, Seaborn, Folium          |
| **Machine Learning** | Scikit-learn                                    |
| **Statistical Analysis** | NumPy, Pandas, SciPy                        |
| **Deployment**       | Streamlit Cloud, GitHub                         |

---

## 🔒 Security & Privacy

- **No Persistent Storage**: Uploaded data is never permanently stored
- **Session Isolation**: Each user session is completely isolated
- **Synthetic Data**: Sample data is completely synthetic and HIPAA-safe
- **Local Processing**: All data processing happens locally in your browser session
- **No External API Calls**: Healthcare data never leaves your environment

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests if applicable
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request`

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Troubleshooting

**Common Issues:**

- **Import Errors:**  
  Ensure all dependencies are installed:  
  `pip install -r requirements.txt --upgrade`

- **Sample Data Not Found:**  
  Ensure `sample_data_10k.csv` is in the root directory  
  Check file permissions

- **Visualization Issues:**  
  Reinstall visualization libraries:  
  ```
  pip uninstall bokeh altair seaborn folium
  pip install bokeh altair seaborn folium
  ```

- **Memory Issues with Large Datasets:**  
  The app automatically samples large datasets for visualization  
  Consider filtering your data before analysis

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/patient360-analytics/issues)
- **Documentation**: [Wiki](https://github.com/yourusername/patient360-analytics/wiki)
- **Email**: suryansh.prasad@cognizant.com

---

**Built with ❤️ for Healthcare Professionals**

*Empowering data-driven healthcare decisions with transparency and precision.*
