# Data Analysis & Linear Algebra Toolkit

A practical Python toolkit for matrix operations, data cleaning, and machine learning using NumPy, Pandas, and Matplotlib.

**Perfect for**: Data analysis, machine learning projects, and understanding data science fundamentals.

## 🎯 Features

✅ **Matrix Operations** - Add, subtract, multiply, transpose, determinant, inverse, rank using NumPy  
✅ **Data Processing** - Missing values, outlier detection, groupby analysis with Pandas  
✅ **Linear Regression** - Simple ML model implementation with evaluation metrics  
✅ **Visualization** - Scatter plots, histograms, correlation heatmaps using Matplotlib  
✅ **Easy to Use** - Simple, readable code with practical examples  


## 🚀 Quick Start

### Matrix Operations
```python
from src.data_toolkit import SimpleMatrixOps
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Operations
result = SimpleMatrixOps.add_matrices(A, B)
det = SimpleMatrixOps.determinant(A)
inv = SimpleMatrixOps.inverse(A)
rank = SimpleMatrixOps.rank(A)
```

### Data Processing
```python
from src.data_toolkit import DataProcessor

processor = DataProcessor('data.csv')
processor.fill_missing('price', method='mean')
processor.remove_outliers('price', threshold=3.0)
processor.basic_stats()
processor.group_by_analysis('city', 'price', 'mean')
```

### Linear Regression
```python
from src.data_toolkit import SimpleLinearRegression
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

model = SimpleLinearRegression()
model.fit(X, y)
predictions = model.predict(np.array([3.5]))
r2 = model.r_squared(X, y)
```

---

## 📚 Examples

All examples are in `examples/` folder:
- `example1_matrix_ops.py` - Matrix operations
- `example2_data_cleaning.py` - Data processing
- `example3_linear_regression.py` - ML model

Run any example:
```bash
python examples/example1_matrix_ops.py
```

---

## 📊 Project Components

### 1. SimpleMatrixOps Class
Matrix operations using NumPy
- `add_matrices()` - Add two matrices
- `subtract_matrices()` - Subtract matrices
- `multiply_matrices()` - Matrix multiplication
- `transpose()` - Transpose matrix
- `determinant()` - Calculate determinant
- `inverse()` - Calculate inverse
- `rank()` - Find matrix rank
- `solve_linear_system()` - Solve Ax = b

### 2. DataProcessor Class
Data cleaning and analysis with Pandas
- `show_data()` - Display data
- `missing_values()` - Check missing values
- `fill_missing()` - Impute missing values (mean/median)
- `remove_outliers()` - Remove outliers (Z-score)
- `basic_stats()` - Statistical summary
- `group_by_analysis()` - Groupby aggregation

### 3. SimpleLinearRegression Class
ML model implementation
- `fit(X, y)` - Train model
- `predict(X)` - Make predictions
- `r_squared(X, y)` - Calculate R² metric

### 4. VisualizationHelper Class
Visualization using Matplotlib
- `plot_scatter()` - Scatter plot
- `plot_histogram()` - Histogram
- `plot_correlation_heatmap()` - Correlation heatmap

---

## 💡 Use Cases

**Data Cleaning**: Remove missing values and outliers from datasets  
**Linear Algebra**: Matrix operations for ML algorithms  
**Regression**: Simple prediction model with evaluation  
**Visualization**: Understand data through plots and heatmaps  

---

## 🔧 Technologies Used

- **NumPy** - Matrix operations and numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **SciPy** - Z-score calculation for outlier detection

---

## 📈 Sample Output

```
======================================================================
Data Analysis & Linear Algebra Toolkit
======================================================================

📊 EXAMPLE 1: Basic Matrix Operations (NumPy)
Matrix A:
[[1 2]
 [3 4]]

✓ Addition Result:
[[ 6  8]
 [10 12]]

✓ Determinant: -2.00

📊 EXAMPLE 2: Data Processing
Missing values: price    1
Shape: (6, 4)

✓ Filled missing values in 'price' using mean

📊 EXAMPLE 3: Linear Regression
✓ Model fitted: y = 1.85x + 1.20
R² Score: 0.9532
```





