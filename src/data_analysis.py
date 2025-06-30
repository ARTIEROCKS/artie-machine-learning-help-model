import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

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


if __name__ == "__main__":

    # Loading the parameters
    params_file = sys.argv[1]
    input_csv_file = sys.argv[2]
    output_class_distribution_image = sys.argv[3]
    output_class_distribution_summary = sys.argv[4]

    # Load dataset
    df = pd.read_csv(input_csv_file)

    # Filter students by age ≤ 15 with valid numeric values
    df = df.dropna(subset=['student_age'])
    df = df[pd.to_numeric(df['student_age'], errors='coerce').notna()]
    df['student_age'] = df['student_age'].astype(float)
    df = df[df['student_age'] <= 15]
    df = df.reset_index(drop=True)

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