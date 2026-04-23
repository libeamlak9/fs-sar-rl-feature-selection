import matplotlib.pyplot as plt

# Data for the table
top_features = [10, 50, 100, 200, 300, 400, 500, 5000, 10000, 50000, 100000, 500000, 1000000]
accuracy_1shot = [33.3, 33.3, 36.2, 46.73, 57.3, 61.1, 64.5, 76.07, 82.53, 83.63, 83.60, 82.23, 84.2]
accuracy_2shot = [45.3, 82.8, 86.73, 86.10, 87.23, 88.47, 87.7, 90.47, 90.20, 91.20, 91.09, 91.43, 90.6]
accuracy_5shot = [33.3, 57.5, 71.7, 81.4, 95.4, 94.67, 94.67, 94.30, 97.1, 97.17, 98.5, 98.93, 98.53]

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting the data
plt.plot(top_features, accuracy_1shot, label='3-way 1-shot', marker='o', linestyle='-', color='r')
plt.plot(top_features, accuracy_2shot, label='3-way 2-shot', marker='o', linestyle='-', color='g')
plt.plot(top_features, accuracy_5shot, label='3-way 5-shot', marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Number of Selected Features (Top Features)')
plt.ylabel('Accuracy (%)')

# Add a grid, legend, and show the plot
plt.grid(True)
plt.legend()
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.xticks(top_features, rotation=45)
plt.tight_layout()

# Save the plot as an image file
plt.savefig('accuracy_vs_features.png')

# Show the plot
plt.show()


