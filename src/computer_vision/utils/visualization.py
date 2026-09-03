import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def general_performance(dataframe, title : str, dest : Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.set_theme(style="whitegrid")

    # Plot 1: Overall Average Performance
    metrics_to_mean = ['Precision', 'Recall', 'F1 Score', 'Weighted Recall', "Latency Recall"]
    means = dataframe[metrics_to_mean].mean()

    sns.barplot(x=means.index, y=means.values, ax=axes[0, 0], palette="viridis", hue=means.index)
    axes[0, 0].set_title(f"Average Performance Metrics ({title})", fontsize=18)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Score", fontsize=15)
    axes[0, 0].tick_params(axis='both', labelsize=13)
    for i, v in enumerate(means.values):
        if pd.notna(v):
            axes[0, 0].text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=14, fontweight='bold')

    # Plot 2: Recall by Zone 
    zone_metrics = ['Red Recall', 'Orange Recall', 'Green Recall']
    zone_means = dataframe[zone_metrics].mean()
    zone_colors = ['#e74c3c', '#e67e22', '#2ecc71'] # Red, Orange, Green

    sns.barplot(x=zone_metrics, y=zone_means.values, ax=axes[0, 1], palette=zone_colors, hue=zone_metrics)
    axes[0, 1].set_title("Average Recall by Physical Zone", fontsize=18)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Recall Rate", fontsize=15)
    axes[0, 1].tick_params(axis='both', labelsize=13)
    for i, v in enumerate(zone_means.values):
        if pd.notna(v):
            axes[0, 1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=14, fontweight='bold')

    # Plot 3: Distribution of F1 Scores with Kernel Density Estimate to smooth the data
    sns.histplot(dataframe['F1 Score'].dropna(), bins=10, kde=True, ax=axes[1, 0], color="royalblue")
    axes[1, 0].set_title("Distribution of F1 Scores Across Videos", fontsize=18)
    axes[1, 0].set_xlabel("F1 Score", fontsize=15)
    axes[1, 0].set_ylabel("Number of Videos", fontsize=15)
    axes[1, 0].tick_params(axis='both', labelsize=13)

    # Plot 4: Miss Probability vs False Alarm Rate
    sns.scatterplot(data=dataframe, x='P_false_alarm', y='P_miss', ax=axes[1, 1], s=100, color="crimson", alpha=0.7)
    axes[1, 1].set_title("Trade-off: Miss Probability vs. False Alarm Probability", fontsize=18)
    axes[1, 1].set_xlabel("Probability of False Alarm (P_fa)", fontsize=15)
    axes[1, 1].set_ylabel("Probability of Miss (P_miss)", fontsize=15)
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_xlim(-0.05, 1.05)
    axes[1, 1].tick_params(axis='both', labelsize=13)
    # Add an "ideal" marker at 0,0
    axes[1, 1].plot(0, 0, marker='*', color='gold', markersize=15, label="Ideal Performance")
    axes[1, 1].legend(fontsize=13)

    plt.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()

def plot_delay(dataframe, dest : Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.set_theme(style="whitegrid")

    bins = 20
    sns.histplot(dataframe['frame'], bins=bins, kde=False, ax=axes[0], color="royalblue", edgecolor="black")
    axes[0].set_title("Delay Distribution (In Frames)", fontsize=18)
    axes[0].set_xlabel("Delay (Frames)", fontsize=15)
    axes[0].set_ylabel("Frequency (Number of Videos)", fontsize=15)
    axes[0].tick_params(axis='both', labelsize=13)

    # Adicionando uma anotação para destacar o pico de 100 frames
    peak_frames = len(dataframe[dataframe['frame'] == 100])
    axes[0].annotate(f'Peak in the threshold\nof the window ({peak_frames} videos)',
                    xy=(100 - 100/(bins*2), peak_frames),
                    xytext=(60, peak_frames-15),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                    fontsize=14,
                    backgroundcolor='white')


    # --- Gráfico 2: Atraso em Segundos ---
    sns.histplot(dataframe['second'], bins=20, kde=False, ax=axes[1], color="crimson", edgecolor="black")
    axes[1].set_title("Delay Distribution (In Seconds)", fontsize=18)
    axes[1].set_xlabel("Delay (Seconds)", fontsize=15)
    axes[1].set_ylabel("Frequency (Number of Videos)", fontsize=15)
    axes[1].tick_params(axis='both', labelsize=13)

    plt.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()

def plot_distortions(dataframe, plot_order, metric : str, dest:Path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    sns.set_theme(style="whitegrid")

    sns.barplot(
        data=dataframe,
        x='Distortion_Type',
        y=metric,
        order=plot_order,
        ax=axes[0],
        palette="tab10",
        hue='Distortion_Type',
        errorbar=None # Removes confidence intervals for a cleaner look if desired
    )
    axes[0].set_title(f"Average {metric}: Original vs. Distortions", fontsize=20)
    axes[0].set_ylabel(metric, fontsize=15)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=30, labelsize=17)
    axes[0].tick_params(axis='y', labelsize=13)

    sns.boxplot(
        data=dataframe,
        x='Distortion_Type',
        y=metric,
        order=plot_order,
        ax=axes[1],
        palette="tab10",
        hue='Distortion_Type'
    )
    axes[1].set_title(f"Consistency of {metric} Across Distortions", fontsize=20)
    axes[1].set_ylabel(metric, fontsize=15)
    axes[1].set_xlabel("Distortion Type", fontsize=15)
    axes[1].tick_params(axis='x', rotation=30, labelsize=17)
    axes[1].tick_params(axis='y', labelsize=13)

    plt.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()

def plot_occurence(dataframe, dest:Path):
    col_detections = 'Predicted Interval'
    col_total = 'Groud Truth Interval'

    total_detected = dataframe[col_detections].sum()
    total_ground_truth = dataframe[col_total].sum()

    success_rate = (total_detected / total_ground_truth) * 100 if total_ground_truth > 0 else 0

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    categories = ['Total Ground Truth Intervals\n(People Actually Present)',
                'Detected Intervals\n(At Least 1 Detection)']
    values = [total_ground_truth, total_detected]
    colors = ['#34495e', '#2ecc71']  # Dark blue-gray for total, bright green for success

    ax = sns.barplot(x=categories, y=values, palette=colors, hue=categories, legend=False)

    plt.title("Detection Coverage", fontsize=20, pad=15)
    plt.ylabel("Number of Intervals", fontsize=15)
    plt.ylim(0, max(values) * 1.2)
    ax.tick_params(axis='both', labelsize=13)

    for i, v in enumerate(values):
        ax.text(i, v + (max(values) * 0.02), f"{int(v)}", ha='center', fontsize=17, fontweight='bold')

    ax.text(1, total_detected / 2, f"{success_rate:.1f}% Coverage",
            ha='center', color='white', fontsize=15, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()

def extract_video_info(filename, distortions):
    clean_name = str(filename)
    if clean_name.endswith('.mp4'):
        clean_name = clean_name[:-4]
        
    for dist in distortions:
        suffix = f"_{dist}"
        if clean_name.endswith(suffix):
            base_name = clean_name[:-len(suffix)]
            return pd.Series([base_name, dist])

    # If no suffix matches, it must be the original video
    return pd.Series([clean_name, "original"])

def generate_metrics_report(dataframe_ped, df_delay, dest: Path):
    """Generates a .txt file with the average values of all key metrics."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Weighted Recall', 
               'Latency Recall', 'P_miss', 'P_false_alarm', 'NDCR',
               'Red Recall', 'Orange Recall', 'Green Recall']
               
    means = dataframe_ped[metrics].mean()
    
    with open(dest, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("OVERALL AVERAGE METRICS (Pedestrians Present)\n")
        f.write("==================================================\n")
        
        for metric in metrics:
            if metric in means and pd.notna(means[metric]):
                f.write(f"{metric:<20}: {means[metric]:.4f}\n")
            else:
                f.write(f"{metric:<20}: N/A\n")
                
        f.write("\n==================================================\n")
        f.write("DELAY METRICS\n")
        f.write("==================================================\n")
        if not df_delay.empty:
            f.write(f"{'Average Delay (Frames)':<20}: {df_delay['frame'].mean():.2f}\n")
            f.write(f"{'Average Delay (Sec)':<20}: {df_delay['second'].mean():.4f}\n")
        else:
            f.write("No delay data available.\n")

systems = ['blaxtair', 'brigade', 'efa', 'stonkam_camera', 'stonkam_screen']

for system in systems:
    csvpath = Path(f'graphs/{system}_complete_test.csv')
    delaypath = Path(f'graphs/{system}_complete_test_delay.csv')
    dest = Path("graphs") / f"{system}"
    print(dest)

    distortions = [
        'gaussian_noise', 'gaussian_noise_conv', 'gaussian_blur',
        'smoke', 'salt_and_pepper', 'rain', 'dirt', 'fog', 'light'
    ]

    try:
        df = pd.read_csv(csvpath)
        df_delay = pd.read_csv(delaypath)
    except FileNotFoundError as e:
        print(f"Skipping {system}: {e}")
        continue

    df[['Base_Name', 'Distortion_Type']] = df['Name'].apply(extract_video_info, distortions = distortions)
    df_delay[['Base_Name', 'Distortion_Type']] = df_delay['name'].apply(extract_video_info, distortions = distortions)

    df_pedestrians = df[df['Has_Pedestrians'] == True]

    generate_metrics_report(df, df_delay, dest / 'average_metrics_summary.txt')

    general_performance(df_pedestrians, 'Pedestrian', dest / 'pedestrian.jpg')

    codes = ['_00_', '_02_', '_03_', '_04_']
    for code in codes:

        regex_pattern = rf"{code}\d+(\.mp4)?$"

        df_code = df[df['Base_Name'].str.contains(regex_pattern, regex=True)]
        
        general_performance(df_code, code, dest / f'code {code}.jpg')

    plot_delay(df_delay, dest / "delay.jpg")

    

    plot_order = ['original'] + distortions

    for metric in ['F1 Score', 'Latency Recall', 'P_miss', 'P_false_alarm', 'Weighted Recall']:
        plot_distortions(df, plot_order, metric, dest / f"distortion_{metric}.jpg")


    plot_occurence(df, dest / "Total_Occurence.jpg")

    df_original = df[df['Distortion_Type'] == 'original']

    plot_occurence(df_original, dest / "Original_Occurence.jpg")

    general_performance(df_original, 'Original', dest / 'original_performance.jpg')

    df_delay_original = df_delay[df_delay['Distortion_Type'] == 'original'] 

    generate_metrics_report(df_original, df_delay_original, dest / 'average_original_metrics_summary.txt')