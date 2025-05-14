import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Read the data
df = pd.read_csv('data.csv')

df = df.dropna(subset=['Sentiment'])
df = df[df['Sentiment'] != '']
# drop request for feature row 
df = df[df['Codes'] != 'Request for feature']

# Filter rows by sentiment
positive = df[df['Sentiment'] == 'Positive']
negative = df[df['Sentiment'] == 'Negative']
neutral = df[df['Sentiment'] == 'Neutral']

# Get all columns except 'Sentiment' for plotting
columns_to_plot = df.columns[1:-1]  # Exclude 'Codes' and 'Sentiment'

# Convert all data to numeric, coercing errors to NaN
for col in columns_to_plot:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    positive[col] = pd.to_numeric(positive[col], errors='coerce')
    negative[col] = pd.to_numeric(negative[col], errors='coerce')
    neutral[col] = pd.to_numeric(neutral[col], errors='coerce')

# Calculate sums, replacing NaN with 0
positive_sums = positive[columns_to_plot].fillna(0).sum().tolist()
negative_sums = negative[columns_to_plot].fillna(0).sum().tolist()
neutral_sums = neutral[columns_to_plot].fillna(0).sum().tolist()

# Set up the plot - increase figure height and reduce width ratio
fig, ax = plt.subplots(figsize=(15, 12))  # Increased height from 8 to 12

# Set the x positions for the bars
x = np.arange(len(columns_to_plot))
width = 0.5  # Reduced width from 0.75 to 0.5 to make bars skinnier

# Create stacked bars
bottom_pos = [0] * len(columns_to_plot)  # Starting position for positive bars (bottom)
pos_bars = ax.bar(x, positive_sums, width, label='Positive', color='green', bottom=bottom_pos)

# Add neutral on top of positive
bottom_neu = positive_sums.copy()  # Starting position for neutral bars
neu_bars = ax.bar(x, neutral_sums, width, label='Neutral', color='black', bottom=bottom_neu)

# Add negative on top of neutral
bottom_neg = [p + n for p, n in zip(positive_sums, neutral_sums)]  # Starting position for negative bars
neg_bars = ax.bar(x, negative_sums, width, label='Negative', color='darkred', bottom=bottom_neg)
# Get the unique codes from the dataset
codes = df['Codes'].unique()
code_to_index = {code: i+1 for i, code in enumerate(codes)}

# Add code numbers to each segment in stacked bars
def add_code_labels_stacked(bars, sentiment, bottom_positions):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:  # Only add label if bar has height
            sentiment_df = df[df['Sentiment'] == sentiment]
            
            # Get all codes for this sentiment
            sentiment_codes = []
            for code in sentiment_df['Codes'].unique():
                code_value = sentiment_df[sentiment_df['Codes'] == code][columns_to_plot[i]].sum()
                if pd.notna(code_value) and code_value > 0:
                    sentiment_codes.append((code, code_value, code_to_index[code]))
            
            # Sort by code index
            sentiment_codes.sort(key=lambda x: x[2])
            
            # Track the y position
            current_y = bottom_positions[i]
            # Add each code and draw dividing line
            for j, (code, code_value, code_index) in enumerate(sentiment_codes):
                if code_value > 0:
                    # Add the code index at the appropriate height in the bar
                    # Better centering by calculating exact center of segment
                    center_x = bar.get_x() + bar.get_width()/2
                    center_y = current_y + code_value/2
                    
                    ax.text(center_x, center_y, 
                            str(code_index), 
                            ha='center', 
                            va='center', 
                            color='white', 
                            fontweight='bold', 
                            fontsize=9,
                            bbox=dict(boxstyle="circle,pad=0.3", fc="none", ec="none"))  # Add padding around numbers
                    
                    # Draw horizontal dividing line between codes (except after the last one)
                    if j < len(sentiment_codes) - 1:
                        next_y = current_y + code_value
                        ax.plot(
                            [bar.get_x(), bar.get_x() + bar.get_width()],
                            [next_y, next_y],
                            color='white', linestyle='-', linewidth=1.2, alpha=1.0  # Much thicker and fully opaque lines
                        )
                    
                    # Update y position for next code
                    current_y += code_value

# Add dividing lines between bars
for i in range(len(x) - 1):
    ax.axvline(x=i + 0.5, color='black', linestyle='-', alpha=0.35, linewidth=1.0) # Thicker dividing lines between bars
add_code_labels_stacked(pos_bars, 'Positive', bottom_pos)
add_code_labels_stacked(neu_bars, 'Neutral', bottom_neu)
add_code_labels_stacked(neg_bars, 'Negative', bottom_neg)

# Add dividing lines between bars
for i in range(len(x) - 1):
    ax.axvline(x=i + 0.5, color='black', linestyle='-', alpha=0.25, linewidth=0.7)

# Add horizontal grid lines to help read values
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

# Create a more horizontal code legend in a separate figure
legend_fig = plt.figure(figsize=(12, 6))  # Wider, shorter figure
legend_ax = legend_fig.add_subplot(111)
legend_ax.axis('off')

# Create a neat, horizontal legend for codes with numbers
sorted_codes = sorted([(code_to_index[code], code) for code in codes])
legend_handles = []
legend_labels = []

# Format the code legend entries
for i, (idx, code) in enumerate(sorted_codes):
    legend_handles.append(Patch(facecolor='lightgray', edgecolor='black'))
    legend_labels.append(f"{idx}: {code}")

# Calculate number of columns based on number of codes
ncol = min(4, len(sorted_codes) // 6 + 1)  # Maximum 4 columns, minimum 1
ncol = max(ncol, 2)  # At least 2 columns for better horizontal layout

# Add the code legend as a separate figure element with multiple columns
legend_ax.legend(handles=legend_handles, labels=legend_labels, 
                loc='center', 
                title='Code Index Legend',
                ncol=ncol,  # Multiple columns for horizontal layout
                fontsize='small',
                frameon=True,
                fancybox=True,
                shadow=True)

# Add sentiment legend to main plot with new Rectangle objects
sentiment_handles = [
    plt.Rectangle((0,0), 1, 1, color='green', label='Positive'),
    plt.Rectangle((0,0), 1, 1, color='black', label='Neutral'),
    plt.Rectangle((0,0), 1, 1, color='darkred', label='Negative')
]
sentiment_legend = ax.legend(handles=sentiment_handles, 
                            labels=['Positive', 'Neutral', 'Negative'],
                            loc='upper left', bbox_to_anchor=(1.05, 0.5), 
                            title='Sentiment')
ax.add_artist(sentiment_legend)

# Add code reference legend with new Line2D object
code_ref_handle = [plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor='black', markersize=6)]  # Slightly larger marker
code_ref_legend = ax.legend(handles=code_ref_handle, 
                           labels=['Numbers represent code types\n(see Code Legend)'], 
                           loc='upper left', bbox_to_anchor=(1.05, 0.8))
ax.add_artist(code_ref_legend)

# Add labels, title
ax.set_xlabel('Places', fontsize=11)
ax.set_ylabel('People mentioned', fontsize=11)
ax.set_title('Stacked Sentiment Distribution Across Places with Code Segmentation', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(columns_to_plot, rotation=45, ha='right', fontsize=10)  # Slightly larger x tick labels

# Adjust layout with more space for the legends
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Make room for the legend

# Show the plots in separate windows
legend_fig.tight_layout()
legend_fig.savefig('code_legend.png', bbox_inches='tight')

# Focus on the main figure and show it
plt.figure(fig.number) 
plt.show()