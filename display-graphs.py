import matplotlib.pyplot as plt

# Data
algorithms = ['Optimized Apriori', 'iApriori', 'AprioriTID', 'Original Apriori']
times = [0.4274, 0.6191, 1.1829, 1.0699]

# Create bar chart
plt.figure(figsize=(8, 5))
plt.bar(algorithms, times, color=['green', 'blue', 'orange', 'red'])

# Labels and title
plt.xlabel('Algorithm')
plt.ylabel('Execution Time (seconds)')
plt.title('Comparison of Apriori Algorithm Execution Times')
plt.ylim(0, 1.3)  # Set y-axis limit for better visualization

# Display values on bars
for i, time in enumerate(times):
    plt.text(i, time + 0.02, f'{time:.4f}', ha='center', fontsize=12)

# Show the plot
plt.show()
