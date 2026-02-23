# Student Performance Prediction Project

A machine learning project to predict student performance index based on various factors like study hours, previous scores, and sleep hours.

**Dataset:** [Kaggle](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)  
**Repository:** [GitHub](https://github.com/SohaibAamir28/Student-Performance-Prediction-Project)

## Features
- **Data Preprocessing**: Handles duplicates and categorical encoding.
- **Model Training**: Uses Multiple Linear Regression to train on historical data.
- **Metrics**: Calculates and displays R², Adjusted R², RMSE, and MAE.
- **Interactive Dashboard**: A Streamlit-based UI for real-time predictions and data visualization.

## Project Structure
- `app.py`: Streamlit dashboard application.
- `src/model.py`: Core logic for data processing and model training.
- `models/`: Contains the serialized trained model (`.pkl`).

## How to Run
1. Navigate to the project folder:
   ```bash
   cd Student Performance Prediction Project
   ```
2. Train the model (if needed):
   ```bash
   python src/model.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
