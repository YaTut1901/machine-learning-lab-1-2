# Machine Learning & Adaptive Intelligence – Lab 1 & 2  

This repository contains solutions to **Laboratory Work 1–2** for the course *Methods of Machine Learning and Adaptive Intelligence*.

The tasks cover the basics of **data preprocessing, regression models, feature selection, and time series forecasting** using Python and popular ML libraries.  

---

## 📂 Repository Structure  

```
├── data/                # Datasets (CSV files or links to external sources)
├── notebooks/           # Jupyter notebooks with step-by-step solutions
│   ├── task1_intro.ipynb
│   ├── task2_preprocessing.ipynb
│   ├── task3_simple_regression.ipynb
│   ├── task4_multiple_regression.ipynb
│   ├── task5_polynomial_regression.ipynb
│   ├── task6_feature_selection.ipynb
│   └── task7_time_series_regression.ipynb
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 📝 Tasks Overview  

### Task 1: Getting Started  
- Create a Jupyter Notebook or Python script.  
- Import libraries: **NumPy, Pandas, Matplotlib, scikit-learn**.  
- Generate synthetic data for linear regression.  
- Visualize data with Matplotlib.  

### Task 2: Data Preprocessing  
- Load a real dataset (e.g., CSV with housing prices or student grades).  
- Explore data with Pandas: missing values, data types, summary statistics.  
- Visualize data with scatter plots to identify linear relationships.  

### Task 3: Simple Linear Regression  
- Implement simple regression using **scikit-learn**.  
- Split dataset into train/test sets.  
- Train the model and evaluate with **Mean Squared Error (MSE)**.  
- Plot regression line vs data points.  

### Task 4: Multiple Linear Regression  
- Load dataset with multiple features.  
- Handle categorical variables & feature scaling.  
- Implement multiple regression with **scikit-learn**.  
- Evaluate with **MSE** and **R² score**.  
- Analyze feature coefficients.  

### Task 5: Polynomial Regression  
- Generate synthetic nonlinear data.  
- Apply polynomial regression with **PolynomialFeatures**.  
- Choose polynomial degree and evaluate performance.  
- Visualize regression curve.  

### Task 6: Feature Selection  
- Apply methods such as **Recursive Feature Elimination (RFE)** or feature importance.  
- Train regression model with selected features.  
- Compare performance with full feature set.  

### Task 7: Time Series Regression  
- Load time series dataset (e.g., stock prices, weather data).  
- Preprocess data for sequential analysis.  
- Apply regression model for forecasting.  
- Evaluate with **MAE** / **RMSE**.  

---

## 📊 Datasets  

Some datasets used:  
- [California Housing Prices](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)  
- [Auto MPG](https://archive.ics.uci.edu/ml/datasets/Auto+MPG)  
- [Fuel Consumption](https://www.kaggle.com/datasets/ahmettyilmazz/fuel-consumption)  
- [Advertising Dataset](https://www.kaggle.com/ashydv/advertising-dataset)  
- [S&P 500 Stock Prices](https://www.kaggle.com/camnugent/sandp500)  
- [Synthetic datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html#synthetic-dataset)  

---

## ⚙️ Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/ml-ai-lab1-2.git
   cd ml-ai-lab1-2
   ```  

2. Create a virtual environment and install dependencies:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```  

3. Run Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```  

---

## 🛠 Requirements  

Main libraries used:  
- Python 3.8+  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  
- Jupyter Notebook  

---

## 📌 Notes  

- Each task is implemented in both **Jupyter Notebooks** (interactive) and **Python scripts** (for automation).  
- Code is structured for clarity and reproducibility.  
- Metrics and visualizations are included for model evaluation.  

---

## 📖 License  

This repository is intended for educational purposes.  
