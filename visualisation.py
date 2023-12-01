import matplotlib.pyplot as plt

mAP_values = [0.67, 0.68, 0.74, 0.65]  # Replace with your recorded mAP values
training_times = [82, 79, 130, 83]  # Replace with your recorded training times
model_sizes = [92, 81.7, 82.9, 16.5]  # Replace with your recorded model sizes
model_names = ['Mod. GoogleNet', 'ResNet34', 'ViT_s', 'EfficientNet']
colors = ['red', 'blue', 'green', 'orange']

# Define a scaling factor for model sizes to adjust point sizes in the plot
scaling_factor = 10  # Adjust this factor for better visualization

# Create a scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(training_times, mAP_values, s=[size * scaling_factor for size in model_sizes], c=colors, alpha=0.7)

# Set labels and title
plt.xlabel('Training Time (in minutes)')
plt.xticks(range(70, 150, 15))
plt.ylabel('mAP')
plt.title('mAP vs Training Time (size scaled by Model Size)')

# Show grid and display plot
plt.grid(True)

# Add labels for each data point
for i, txt in enumerate(model_names):
    plt.annotate(txt, (training_times[i], mAP_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')

# Add color bar legend
plt.legend(handles=scatter.legend_elements()[0], labels=model_names)

plt.tight_layout()
plt.show()

### CONFUSION MATRIX
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Sample data (replace this with your data)
# data = [[0.710, 0.720, 0.619],
#         [0.674, 0, 0],
#         [0.661, 0.638, 0.640],
#         [0.430, 0, 0]]
#
# # Create a heatmap using imshow with origin at top left
# plt.figure(figsize=(8, 6))
# heatmap = plt.imshow(data, cmap='Greens', interpolation='nearest', origin='upper')
#
# # Add text annotations for each cell
# for i in range(len(data)):
#     for j in range(len(data[0])):
#         plt.text(j, i, f'{data[i][j]:.2f}', ha='center', va='center', color='black')
#
# # Add color bar
# plt.colorbar(heatmap)
#
# # Add labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Heatmap with Values Displayed (Origin: Top Left)')
#
# # Show the plot
# plt.show()
