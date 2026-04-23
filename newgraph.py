import matplotlib.pyplot as plt

# Data for the table
top_features = [10, 50, 100, 200, 300, 400, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 1507328]
accuracy_1shot = [33.3, 33.3, 36.2, 46.73, 57.3, 61.1, 64.5, 76.07, 82.53, 83.63, 83.60, 82.23, 84.2, 84.1, 84.0]
accuracy_2shot = [45.3, 82.8, 86.73, 86.10, 87.23, 88.47, 87.7, 90.47, 90.20, 91.20, 91.09, 91.43, 90.6, 91.2, 90.6]
accuracy_5shot = [33.3, 57.5, 71.7, 81.4, 95.4, 94.67, 94.67, 94.30, 97.1, 97.17, 98.5, 98.93, 98.53, 98.1, 98.2]

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting the data with different styles
plt.plot(top_features, accuracy_1shot, label='3-way 1-shot', marker='o', linestyle='-', color='r')
plt.plot(top_features, accuracy_2shot, label='3-way 2-shot', marker='o', linestyle='-', color='g')
plt.plot(top_features, accuracy_5shot, label='3-way 5-shot', marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Number of Selected Features (Top Features)', fontsize=20)
plt.ylabel('Accuracy (%)', fontsize=20)

# Add a grid, legend, and set the x-axis to logarithmic scale
plt.grid(True)
plt.legend(fontsize=16)
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.xticks(top_features, rotation=45)

# Ensure tight layout
plt.tight_layout()

# Save the plot as a high-quality PDF
plt.savefig('accuracy_vs_features.jpg', format='jpg', dpi=300)

# Show the plot
plt.show()
