import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def plot_class_distribution(df, target_column, output_path, summary_csv_path=None):
    """
    Plots and saves the class distribution of a binary target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): Name of the binary target column (e.g., 'request_help').
        output_path (str): Path to save the plot image (e.g., 'dvc_plots/class_distribution.png').
        summary_csv_path (str, optional): If provided, saves the class distribution summary to this CSV file.

    Returns:
        pd.DataFrame: A summary table with class counts and percentages.
    """
    # Count class occurrences
    class_counts = df[target_column].value_counts()
    class_percentages = df[target_column].value_counts(normalize=True) * 100

    # Print class distribution
    print(f"Class distribution for '{target_column}':")
    for label, count in class_counts.items():
        print(f"Class {label}: {count} samples ({class_percentages[label]:.2f}%)")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Plot the distribution
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="pastel")
    plt.title(f"Distribution of '{target_column}'")
    plt.xlabel(f"{target_column} (0 = No, 1 = Yes)")
    plt.ylabel("Number of samples")
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values,
        'Percentage': class_percentages.values
    })

    # Optionally save summary as CSV
    if summary_csv_path:
        os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False)

    return summary_df

def analyze_feature_importance_shap(df, target_column, drop_columns=None, output_dir="shap_analysis", output_prefix="shap", test_size=0.3, random_state=42):
    """
    Performs feature importance analysis using SHAP values and saves the results as images and CSV.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): Name of the target column (e.g., 'request_help').
        drop_columns (list, optional): List of column names to exclude from the analysis.
        output_dir (str): Directory to save output files.
        output_prefix (str): Prefix for output filenames.
        test_size (float): Proportion of data to use for testing (0.0 to 1.0).
        random_state (int): Random state for reproducibility.

    Returns:
        dict: Dictionary with SHAP values and feature importance summary.
    """
    # Filter rows where exercise_is_evaluation == 0 if column exists
    if 'exercise_is_evaluation' in df.columns:
        df = df.dropna(subset=['exercise_is_evaluation'])
        df = df[df['exercise_is_evaluation'] == 0]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define default columns to drop if none provided
    if drop_columns is None:
        drop_columns = ['group_id', 'date_time', 'student_mother_tongue', 'finished_exercise', 
                       'exercise_is_evaluation', 'student_motivation', 'valid_solution', 
                       'exercise_valid_solution', 'exercise_level', 'grade', 'tree_grade']

    # Prepare feature set and target
    columns_to_drop = [col for col in drop_columns if col in df.columns]
    columns_to_drop.append(target_column)  # Add target to columns to drop

    X = df.drop(columns=columns_to_drop)
    y = df[target_column]

    print(f"Features used for SHAP analysis: {len(X.columns)}")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train XGBoost model
    print("Training XGBoost model for SHAP analysis...")
    model = xgb.XGBClassifier(eval_metric='aucpr')
    model.fit(X_train, y_train)

    # Calculate SHAP values
    print("Calculating SHAP values...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Generate and save SHAP plots
    print("Generating SHAP visualizations...")

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=X.shape[1], show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=X.shape[1], show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Waterfall plot for first instance
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], max_display=X.shape[1], show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}_waterfall.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create feature importance DataFrame based on mean absolute SHAP values
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(shap_values.values).mean(0)
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    # Save feature importance to CSV
    feature_importance.to_csv(f"data/{output_prefix}.csv", index=False)

    print(f"SHAP analysis completed. Files saved to data/")
    print(f"Top 10 most important features:")
    print(feature_importance.head(10))

    return {
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'model': model
    }

def compute_time_steps_by_group_and_day(df, group_col='group_id', datetime_col='date_time',
                                        output_path='data/time_steps_analysis.csv',
                                        date_format='%Y-%m-%d', sep=','):
    """
    Computes the number of time_steps per group_id and day and saves it to a CSV.

    time_steps = number of rows sharing the same group_id and the same date_time formatted as yyyy-mm-dd.
    request_help_count = number of rows with request_help == 1 for the same group_id and date.

    Args:
        df (pd.DataFrame): Input DataFrame.
        group_col (str): Column name containing the group_id.
        datetime_col (str): Column name containing the timestamp (date_time).
        output_path (str): Output CSV path (default 'data/time_steps_analysis.csv').
        date_format (str): Date format for daily aggregation.
        sep (str): CSV separator (default ',').

    Returns:
        pd.DataFrame: DataFrame with columns ['group_id', 'date', 'time_steps', 'request_help_count'].
    """
    # Basic column validation
    if group_col not in df.columns or datetime_col not in df.columns:
        print(f"Warning: columnas '{group_col}' y/o '{datetime_col}' no encontradas en el DataFrame.")
        return pd.DataFrame(columns=['group_id', 'date', 'time_steps', 'request_help_count'])

    # Keep only needed columns and build date column
    if 'request_help' in df.columns:
        tmp = df[[group_col, datetime_col, 'request_help']].copy()
    else:
        tmp = df[[group_col, datetime_col]].copy()
        tmp['request_help'] = 0  # safe default if column is missing

    tmp[datetime_col] = pd.to_datetime(tmp[datetime_col], errors='coerce')
    tmp = tmp.dropna(subset=[datetime_col])

    # Extract date using the required format
    tmp['date'] = tmp[datetime_col].dt.strftime(date_format)

    # Ensure request_help is numeric (0/1) before aggregation
    tmp['request_help_num'] = pd.to_numeric(tmp['request_help'], errors='coerce').fillna(0).astype(int)

    # Group and aggregate: time_steps (size) and request_help_count (sum of request_help==1)
    summary = (
        tmp.groupby([group_col, 'date'])
           .agg(time_steps=('request_help_num', 'size'),
                request_help_count=('request_help_num', 'sum'))
           .reset_index()
    )

    # Save to CSV using ',' as separator
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path, index=False, sep=sep)

    print(f"Time steps analysis guardado en: {output_path} (filas: {len(summary)})")
    return summary

def compute_time_steps_means(df, group_col='group_id', datetime_col='date_time',
                             output_path='data/time_steps_mean_analysis.csv',
                             date_format='%Y-%m-%d', sep=','):
    """
    Computes and saves two global means across time series (grouped by group_id and date):
      - mean_help_requests_per_time_series: average number of help requests per time series
      - mean_time_steps_per_time_series: average number of interactions (time steps) per time series
    """
    # Validate required columns
    if group_col not in df.columns or datetime_col not in df.columns:
        print(f"Warning: columnas '{group_col}' y/o '{datetime_col}' no encontradas en el DataFrame.")
        # Save empty structure to keep pipeline consistent
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(columns=['mean_help_requests_per_time_series', 'mean_time_steps_per_time_series']) \
          .to_csv(output_path, index=False, sep=sep)
        return pd.DataFrame(columns=['mean_help_requests_per_time_series', 'mean_time_steps_per_time_series'])

    # Prepare minimal working frame with request_help
    if 'request_help' in df.columns:
        tmp = df[[group_col, datetime_col, 'request_help']].copy()
    else:
        tmp = df[[group_col, datetime_col]].copy()
        tmp['request_help'] = 0  # safe default if missing

    # Parse datetime and derive date
    tmp[datetime_col] = pd.to_datetime(tmp[datetime_col], errors='coerce')
    tmp = tmp.dropna(subset=[datetime_col])
    tmp['date'] = tmp[datetime_col].dt.strftime(date_format)

    # Ensure request_help is numeric (0/1)
    tmp['request_help_num'] = pd.to_numeric(tmp['request_help'], errors='coerce').fillna(0).astype(int)

    # Aggregate per time series (group_id + date)
    per_series = (
        tmp.groupby([group_col, 'date'])
           .agg(time_steps=('request_help_num', 'size'),
                help_requests=('request_help_num', 'sum'))
           .reset_index()
    )

    # Compute means across series
    means_df = pd.DataFrame({
        'mean_help_requests_per_time_series': [per_series['help_requests'].mean() if not per_series.empty else 0.0],
        'mean_time_steps_per_time_series': [per_series['time_steps'].mean() if not per_series.empty else 0.0]
    })

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    means_df.to_csv(output_path, index=False, sep=sep)

    print(f"Time steps mean analysis saved to: {output_path}")
    return means_df



if __name__ == "__main__":

    # Check command-line arguments
    if len(sys.argv) < 5:
        print("Usage: python data_analysis.py <params_file> <input_csv_file> <output_class_image> <output_class_csv> [output_time_steps_csv] [output_time_steps_mean_csv]")
        sys.exit(1)

    # Loading the parameters (strip to remove accidental leading spaces)
    params_file = sys.argv[1].strip()
    input_csv_file = sys.argv[2].strip()

    # Required output arguments
    output_class_distribution_image = sys.argv[3].strip()
    output_class_distribution_summary = sys.argv[4].strip()

    # Load dataset
    df = pd.read_csv(input_csv_file)

    # Filter students by age ≤ 15 with valid numeric values
    df = df.dropna(subset=['student_age'])
    df = df.dropna(subset=['exercise_is_evaluation'])
    df = df[pd.to_numeric(df['student_age'], errors='coerce').notna()]
    df['student_age'] = df['student_age'].astype(float)
    df = df[df['student_age'] <= 15]
    df = df[df['exercise_is_evaluation'] == 0]
    df = df.reset_index(drop=True)

    print(f"Total rows after the filter: {len(df)}")

    # Compute time steps per group_id and day
    output_time_steps_csv = sys.argv[5].strip() if len(sys.argv) > 5 else "data/time_steps_analysis.csv"
    compute_time_steps_by_group_and_day(df, output_path=output_time_steps_csv)

    # Compute global means across time series
    output_time_steps_mean_csv = sys.argv[6].strip() if len(sys.argv) > 6 else "data/time_steps_mean_analysis.csv"
    compute_time_steps_means(df, output_path=output_time_steps_mean_csv)

    # Analyze class distribution
    summary = plot_class_distribution(
        df=df,
        target_column='request_help',
        output_path=output_class_distribution_image,
        summary_csv_path=output_class_distribution_summary
    )

    # Print the summary table
    print("\nSummary table:")
    print(summary)

    # Perform SHAP analysis if requested
    print("\nPerforming SHAP feature importance analysis...")

    # Perform SHAP analysis
    shap_results = analyze_feature_importance_shap(
        df=df,
        target_column='request_help',
        output_dir="dvc_plots",
        output_prefix="shap_feature_importance"
    )

    print("SHAP analysis completed successfully.")