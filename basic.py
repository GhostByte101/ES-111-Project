import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read the dataset
df = pd.read_csv('student_habits_performance.csv')

# 1. Data Visualization and Frequency Distribution
def analyze_age_distribution():
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(df['age'], bins=10, edgecolor='black')
    plt.title('Age Distribution Histogram')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # Add frequency labels
    for i in range(len(n)):
        plt.text(bins[i], n[i], str(int(n[i])), ha='center', va='bottom')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    age_counts = df['age'].value_counts()
    plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%')
    plt.title('Age Distribution Pie Chart')
    
    plt.tight_layout()
    plt.savefig('age_distribution.png')
    plt.close()
    
    # Frequency distribution table
    freq_dist = pd.cut(df['age'], bins=10).value_counts().sort_index()
    print("\nFrequency Distribution Table:")
    print(freq_dist)
    
    return freq_dist

# 2. Calculate mean and variance using frequency distribution
def calculate_stats(freq_dist):
    # Get midpoints of each bin
    midpoints = [(interval.left + interval.right) / 2 for interval in freq_dist.index]
    
    # Calculate weighted mean
    weighted_mean = np.sum(midpoints * freq_dist) / np.sum(freq_dist)
    
    # Calculate weighted variance
    weighted_variance = np.sum(freq_dist * (midpoints - weighted_mean)**2) / np.sum(freq_dist)
    
    # Compare with direct calculations
    direct_mean = df['age'].mean()
    direct_variance = df['age'].var()
    
    print("\nStatistics Comparison:")
    print(f"Weighted Mean: {weighted_mean:.2f}")
    print(f"Direct Mean: {direct_mean:.2f}")
    print(f"Weighted Variance: {weighted_variance:.2f}")
    print(f"Direct Variance: {direct_variance:.2f}")
    
    return weighted_mean, weighted_variance

# 3. Confidence and Tolerance Intervals
def calculate_intervals():
    # Split data into 80% training and 20% validation
    train_size = int(0.8 * len(df))
    train_data = df['age'].iloc[:train_size]
    test_data = df['age'].iloc[train_size:]
    
    # Calculate sample statistics
    n = len(train_data)
    mean = train_data.mean()
    std = train_data.std()
    
    # 95% Confidence Interval for mean
    ci_lower = mean - stats.t.ppf(0.975, n-1) * (std/np.sqrt(n))
    ci_upper = mean + stats.t.ppf(0.975, n-1) * (std/np.sqrt(n))
    
    # 95% Confidence Interval for variance
    var_ci_lower = (n-1) * std**2 / stats.chi2.ppf(0.975, n-1)
    var_ci_upper = (n-1) * std**2 / stats.chi2.ppf(0.025, n-1)
    
    # 95% Tolerance Interval
    k = stats.norm.ppf(0.975) * np.sqrt(1 + 1/n)
    tol_lower = mean - k * std
    tol_upper = mean + k * std
    
    # Validate with test data
    test_mean = test_data.mean()
    test_var = test_data.var()
    
    print("\nConfidence and Tolerance Intervals:")
    print(f"95% CI for Mean: ({ci_lower:.2f}, {ci_upper:.2f})")
    print(f"95% CI for Variance: ({var_ci_lower:.2f}, {var_ci_upper:.2f})")
    print(f"95% Tolerance Interval: ({tol_lower:.2f}, {tol_upper:.2f})")
    print(f"\nTest Data Statistics:")
    print(f"Test Mean: {test_mean:.2f}")
    print(f"Test Variance: {test_var:.2f}")
    
    # Check if test statistics fall within intervals
    print("\nValidation Results:")
    print(f"Test Mean within CI: {ci_lower <= test_mean <= ci_upper}")
    print(f"Test Variance within CI: {var_ci_lower <= test_var <= var_ci_upper}")
    print(f"Test Data within Tolerance: {tol_lower <= test_data.min() and test_data.max() <= tol_upper}")

def manual_mean(data):
    sum_values = 0
    count = 0
    for value in data:
        sum_values += value
        count += 1
    return sum_values / count

def manual_variance(data, mean):
    sum_squared_diff = 0
    count = 0
    for value in data:
        sum_squared_diff += (value - mean) ** 2
        count += 1
    return sum_squared_diff / count

def manual_std_dev(variance):
    # Manual square root using Newton's method
    if variance == 0:
        return 0
    x = variance
    for _ in range(20):  # 20 iterations for good precision
        x = (x + variance/x) / 2
    return x

def manual_t_critical(df, alpha):
    # Manual approximation of t-critical value using Newton's method
    # This is a simplified approximation for demonstration
    if df <= 30:
        return 2.042  # Approximation for 95% confidence, df=30
    else:
        return 1.96   # Approximation for large samples (normal distribution)

def manual_p_value(t_stat, df):
    # Manual approximation of p-value
    # This is a simplified approximation for demonstration
    if t_stat < 0:
        t_stat = -t_stat
    if df <= 30:
        if t_stat > 2.042:
            return 0.05
        elif t_stat > 1.697:
            return 0.10
        else:
            return 0.20
    else:
        if t_stat > 1.96:
            return 0.05
        elif t_stat > 1.645:
            return 0.10
        else:
            return 0.20

def manual_t_statistic(mean, hypothesized_mean, std_dev, n):
    # Calculate t-statistic manually
    standard_error = std_dev / manual_std_dev(n)
    return (mean - hypothesized_mean) / standard_error

def test_sleep_hypothesis():
    # Read the dataset
    df = pd.read_csv('student_habits_performance.csv')
    sleep_hours = df['sleep_hours'].tolist()
    
    # Calculate statistics manually
    mean = manual_mean(sleep_hours)
    variance = manual_variance(sleep_hours, mean)
    std_dev = manual_std_dev(variance)
    n = len(sleep_hours)
    
    # Calculate t-statistic
    t_stat = manual_t_statistic(mean, 7, std_dev, n)
    
    # Print results
    print("\nSleep Hours Analysis (Manual Calculations):")
    print(f"Number of students: {n}")
    print(f"Mean sleep hours: {mean:.2f}")
    print(f"Variance: {variance:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")
    print(f"t-statistic: {t_stat:.2f}")
    
    # Compare with hypothesis (7 hours)
    print("\nComparison with Hypothesis (7 hours):")
    print(f"Difference from hypothesized mean: {mean - 7:.2f} hours")
    
    # Calculate percentage of students below/above 7 hours
    below_7 = sum(1 for hours in sleep_hours if hours < 7)
    above_7 = sum(1 for hours in sleep_hours if hours > 7)
    exactly_7 = sum(1 for hours in sleep_hours if hours == 7)
    
    print("\nDistribution relative to 7 hours:")
    print(f"Students sleeping less than 7 hours: {(below_7/n)*100:.1f}%")
    print(f"Students sleeping exactly 7 hours: {(exactly_7/n)*100:.1f}%")
    print(f"Students sleeping more than 7 hours: {(above_7/n)*100:.1f}%")
    
    # Hypothesis Test Conclusion
    print("\nHypothesis Test:")
    print("H0: mean equals to 7 (Null hypothesis: Mean sleep hours is 7)")
    print("H1: mean does not equal to 7 (Alternative hypothesis: Mean sleep hours is not 7)")
    
    # For large sample size (n > 30), critical value is approximately ±1.96 for α = 0.05
    critical_value = 1.96
    
    if abs(t_stat) > critical_value:
        print("\nDecision: Reject the null hypothesis")
        print("Conclusion: There is sufficient evidence to conclude that the mean sleep hours")
        print(f"is different from 7 hours (|t-stat| = {abs(t_stat):.2f} > {critical_value}).")
    else:
        print("\nDecision: Fail to reject the null hypothesis")
        print("Conclusion: There is not sufficient evidence to conclude that the mean sleep hours")
        print(f"is different from 7 hours (|t-stat| = {abs(t_stat):.2f} ≤ {critical_value}).")

def create_visualizations():
    # Set up the figure with subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Gender Distribution (Pie Chart)
    plt.subplot(2, 3, 1)
    gender_counts = df['gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'pink'])
    plt.title('Gender Distribution')
    
    # 2. Study Hours Distribution (Histogram)
    plt.subplot(2, 3, 2)
    plt.hist(df['study_hours_per_day'], bins=15, edgecolor='black', color='green', alpha=0.7)
    plt.title('Study Hours per Day Distribution')
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    
    # 3. Diet Quality Distribution (Pie Chart)
    plt.subplot(2, 3, 3)
    diet_counts = df['diet_quality'].value_counts()
    plt.pie(diet_counts, labels=diet_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'yellow', 'orange'])
    plt.title('Diet Quality Distribution')
    
    # 4. Sleep Hours Distribution (Histogram)
    plt.subplot(2, 3, 4)
    plt.hist(df['sleep_hours'], bins=10, edgecolor='black', color='purple', alpha=0.7)
    plt.title('Sleep Hours Distribution')
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    
    # 5. Parental Education Level (Pie Chart)
    plt.subplot(2, 3, 5)
    education_counts = df['parental_education_level'].value_counts()
    plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%')
    plt.title('Parental Education Level Distribution')
    
    # 6. Exam Score Distribution (Histogram)
    plt.subplot(2, 3, 6)
    plt.hist(df['exam_score'], bins=15, edgecolor='black', color='red', alpha=0.7)
    plt.title('Exam Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()
    
    # Create additional detailed visualizations
    # 7. Social Media vs Netflix Hours (Scatter Plot)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['social_media_hours'], df['netflix_hours'], alpha=0.5)
    plt.title('Social Media Hours vs Netflix Hours')
    plt.xlabel('Social Media Hours')
    plt.ylabel('Netflix Hours')
    plt.grid(True)
    plt.show()
    
    # 8. Exercise Frequency Distribution (Bar Chart)
    plt.figure(figsize=(10, 6))
    exercise_counts = df['exercise_frequency'].value_counts().sort_index()
    exercise_counts.plot(kind='bar', color='blue', alpha=0.7)
    plt.title('Exercise Frequency Distribution')
    plt.xlabel('Frequency (times per week)')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=0)
    plt.show()

    # 9. Display DataFrame head in a popup window
    plt.figure(figsize=(12, 4))
    plt.axis('off')
    table = plt.table(cellText=df.head().values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('First 5 Rows of the Dataset')
    plt.show()

# Execute the analysis
print("Starting Analysis...")
freq_dist = analyze_age_distribution()
weighted_mean, weighted_variance = calculate_stats(freq_dist)
calculate_intervals()

# Execute the hypothesis test
print("Analyzing sleep hours data...")
test_sleep_hypothesis()

# Execute the visualization function
print("Creating visualizations...")
create_visualizations()
