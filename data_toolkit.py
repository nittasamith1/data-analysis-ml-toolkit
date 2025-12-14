# ============================================================================
# SIMPLE MATRIX OPERATIONS & DATA ANALYSIS PROJECT
# Using: NumPy, Pandas, Matplotlib
# Level: Intermediate (No complex algorithms)
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# ============================================================================
# SECTION 1: BASIC MATRIX OPERATIONS (Using NumPy)
# ============================================================================

class SimpleMatrixOps:
    """Simple matrix operations using NumPy."""
    
    @staticmethod
    def add_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Add two matrices."""
        return A + B
    
    @staticmethod
    def subtract_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Subtract matrix B from A."""
        return A - B
    
    @staticmethod
    def multiply_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiply two matrices (matrix multiplication, not element-wise)."""
        return np.dot(A, B)
    
    @staticmethod
    def transpose(A: np.ndarray) -> np.ndarray:
        """Transpose a matrix."""
        return A.T
    
    @staticmethod
    def determinant(A: np.ndarray) -> float:
        """Calculate determinant of a square matrix."""
        return np.linalg.det(A)
    
    @staticmethod
    def inverse(A: np.ndarray) -> np.ndarray:
        """Calculate inverse of a square matrix."""
        return np.linalg.inv(A)
    
    @staticmethod
    def rank(A: np.ndarray) -> int:
        """Find rank of a matrix."""
        return np.linalg.matrix_rank(A)
    
    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b for x."""
        return np.linalg.solve(A, b)


# ============================================================================
# SECTION 2: DATA PROCESSING WITH PANDAS
# ============================================================================

class DataProcessor:
    """Simple data cleaning and processing with Pandas."""
    
    def __init__(self, csv_file: str = None):
        """Initialize with CSV file or create sample data."""
        if csv_file:
            self.df = pd.read_csv(csv_file)
        else:
            # Create sample housing dataset
            self.df = pd.DataFrame({
                'price': [200000, 300000, 250000, np.nan, 450000, 350000],
                'sqft': [1000, 1500, 1200, 1800, 2000, 1600],
                'rooms': [2, 3, 2, 4, 4, 3],
                'city': ['NYC', 'NYC', 'LA', 'LA', 'NYC', 'SF']
            })
    
    def show_data(self):
        """Display first few rows."""
        print("📊 First few rows:")
        print(self.df.head())
        print(f"\nShape: {self.df.shape}")
    
    def missing_values(self):
        """Check missing values."""
        print("\n📌 Missing values:")
        print(self.df.isnull().sum())
    
    def fill_missing(self, column: str, method: str = 'mean'):
        """Fill missing values using mean or median."""
        if method == 'mean':
            self.df[column].fillna(self.df[column].mean(), inplace=True)
        elif method == 'median':
            self.df[column].fillna(self.df[column].median(), inplace=True)
        print(f"✓ Filled missing values in '{column}' using {method}")
        return self
    
    def remove_outliers(self, column: str, threshold: float = 3.0):
        """Remove outliers using Z-score."""
        from scipy.stats import zscore
        
        initial_len = len(self.df)
        z_scores = np.abs(zscore(self.df[column].dropna()))
        self.df = self.df[(np.abs(zscore(self.df[column])) < threshold) | (self.df[column].isnull())]
        
        removed = initial_len - len(self.df)
        print(f"✓ Removed {removed} outliers from '{column}'")
        return self
    
    def basic_stats(self):
        """Show basic statistics."""
        print("\n📈 Statistical Summary:")
        print(self.df.describe())
    
    def group_by_analysis(self, column: str, agg_column: str, agg_func: str = 'mean'):
        """Group by and aggregate."""
        result = self.df.groupby(column)[agg_column].agg(agg_func)
        print(f"\n🔍 {agg_func.upper()} of '{agg_column}' by '{column}':")
        print(result)
        return result


# ============================================================================
# SECTION 3: LINEAR REGRESSION (Simple Implementation)
# ============================================================================

class SimpleLinearRegression:
    """
    Simple linear regression using NumPy.
    Formula: y = mx + b
    """
    
    def __init__(self):
        self.m = None  # Slope
        self.b = None  # Intercept
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        n = len(X)
        
        # Calculate slope (m) and intercept (b)
        # m = (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)
        # b = (Σy - m*Σx) / n
        
        self.m = (n * np.sum(X * y) - np.sum(X) * np.sum(y)) / (n * np.sum(X**2) - np.sum(X)**2)
        self.b = (np.sum(y) - self.m * np.sum(X)) / n
        
        print(f"✓ Model fitted: y = {self.m:.2f}x + {self.b:.2f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.m * X + self.b
    
    def r_squared(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² (goodness of fit)."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# ============================================================================
# SECTION 4: CORRELATION & HEATMAP VISUALIZATION
# ============================================================================

class VisualizationHelper:
    """Visualization helper using Matplotlib."""
    
    @staticmethod
    def plot_scatter(X: np.ndarray, y: np.ndarray, title: str = "Scatter Plot"):
        """Plot scatter plot."""
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, alpha=0.6, color='blue')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_histogram(data: np.ndarray, title: str = "Histogram", bins: int = 20):
        """Plot histogram."""
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=bins, color='green', alpha=0.7, edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame):
        """Plot correlation heatmap."""
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        return plt


# ============================================================================
# DEMONSTRATION & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLE MATRIX & DATA ANALYSIS PROJECT")
    print("=" * 70)
    
    # ========================================================================
    # EXAMPLE 1: BASIC MATRIX OPERATIONS
    # ========================================================================
    print("\n📊 EXAMPLE 1: Basic Matrix Operations (NumPy)")
    print("-" * 70)
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    
    print("\n✓ Addition (A + B):")
    print(SimpleMatrixOps.add_matrices(A, B))
    
    print("\n✓ Multiplication (A @ B):")
    print(SimpleMatrixOps.multiply_matrices(A, B))
    
    print("\n✓ Transpose (A^T):")
    print(SimpleMatrixOps.transpose(A))
    
    print(f"\n✓ Determinant of A: {SimpleMatrixOps.determinant(A):.2f}")
    print(f"\n✓ Inverse of A:")
    print(SimpleMatrixOps.inverse(A))
    
    # ========================================================================
    # EXAMPLE 2: DATA PROCESSING WITH PANDAS
    # ========================================================================
    print("\n\n📊 EXAMPLE 2: Data Processing with Pandas")
    print("-" * 70)
    
    processor = DataProcessor()
    processor.show_data()
    processor.missing_values()
    
    processor.fill_missing('price', method='mean')
    processor.basic_stats()
    processor.group_by_analysis('city', 'price', 'mean')
    
    # ========================================================================
    # EXAMPLE 3: LINEAR REGRESSION
    # ========================================================================
    print("\n\n📊 EXAMPLE 3: Simple Linear Regression")
    print("-" * 70)
    
    # Create sample data: y = 2x + 3 + noise
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5])
    y = 2 * X + 3 + np.random.normal(0, 1, 5)
    
    print("Data:")
    print(f"X: {X}")
    print(f"y: {y}")
    
    # Fit model
    model = SimpleLinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    print(f"\nPredictions: {y_pred}")
    
    # Model evaluation
    r2 = model.r_squared(X, y)
    print(f"R² Score: {r2:.4f}")
    
    # ========================================================================
    # EXAMPLE 4: SOLVING LINEAR SYSTEMS
    # ========================================================================
    print("\n\n📊 EXAMPLE 4: Solving Linear System (Ax = b)")
    print("-" * 70)
    
    # System: 2x + y = 5
    #         x + 3y = 6
    A = np.array([[2, 1], [1, 3]])
    b = np.array([5, 6])
    
    print("System of equations:")
    print("2x + y = 5")
    print("x + 3y = 6")
    
    x = SimpleMatrixOps.solve_linear_system(A, b)
    print(f"\nSolution: x = {x[0]:.2f}, y = {x[1]:.2f}")
    
    # Verify
    print(f"Verification: A @ x = {np.dot(A, x)}")
    
    # ========================================================================
    # EXAMPLE 5: MATRIX RANK & PROPERTIES
    # ========================================================================
    print("\n\n📊 EXAMPLE 5: Matrix Properties")
    print("-" * 70)
    
    C = np.array([[1, 2, 3], [2, 4, 6], [3, 5, 7]])
    print("Matrix C (some rows are dependent):")
    print(C)
    
    rank = SimpleMatrixOps.rank(C)
    print(f"\nRank: {rank}")
    print(f"Shape: {C.shape}")
    print(f"Full Rank: {rank == min(C.shape)}")
    
    print("\n" + "=" * 70)
    print("All examples completed! ✅")
    print("=" * 70)
