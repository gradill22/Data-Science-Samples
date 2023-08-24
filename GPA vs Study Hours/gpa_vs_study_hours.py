import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_bell_curve(column):
    # Calculate mean and standard deviation for data
    mean_data1 = np.mean(column)
    std_data1 = np.std(column)

    # Generate x-values for the bell curve
    x = np.linspace(min(column), max(column), 100)

    # Calculate y-values for the bell curve using the Gaussian formula
    y = (1 / (std_data1 * np.sqrt(2 * np.pi))) * np.exp(-(x - mean_data1)**2 / (2 * std_data1**2))

    return x, y


def histogram(column, label: str, bins: int = 20, bell_curve: bool = True) -> None:
    num_students = len(column)

    # Plot histogram of data column with respective parameters
    plt.hist(column, bins=bins, density=bell_curve, label=label, rwidth=0.8)
    plt.title(f"Histogram of {num_students} Students' {label}")
    plt.xlabel(label)

    # If required, change the y-axis label, plot the bell curve, and display the legend
    # Otherwise, set the y-axis label to the standard label
    if bell_curve:
        plt.ylabel('% of Students')
        x1, y1 = get_bell_curve(column)
        plt.plot(x1, y1, color='orange', linewidth=2, label='Bell Curve')
        plt.legend()
    else:
        plt.ylabel('# of Students')

    # Display and close histogram
    plt.show()
    plt.close()


if __name__ == '__main__':
    data_csv = pd.read_csv('gpa_study_hours.csv')
    # data_csv = data_csv.sort_values(by=['gpa'], ignore_index=True)  optional sorting line

    # Defining data columns and their respective labels
    data1, data2 = data_csv['study_hours'], data_csv['gpa']
    x_label1, x_label2 = 'Study Hours', 'GPA'

    # Display histograms of datasets
    histogram(data1, x_label1)
    histogram(data2, x_label2)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Create box and whisker plot for the first dataset
    ax1.boxplot(data1, patch_artist=True)
    ax1.set_title(f'Study Hours of {len(data1)} Students')
    ax1.set_ylabel(x_label1)

    # Create box and whisker plot for the second dataset
    ax2.boxplot(data2, patch_artist=True)
    ax2.set_title(f'GPA of {len(data2)} Students')
    ax2.set_ylabel(x_label2)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()
    plt.close()
