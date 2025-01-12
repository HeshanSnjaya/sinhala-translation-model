import matplotlib.pyplot as plt

# Data for model iterations and corresponding BLEU scores
models = ['sinhala_final_translation_model 1', 'sinhala_final_translation_model 2', 'sinhala_final_translation_model 3', 'sinhala_corrected_translation_model 4']
bleu_scores = [0.4563, 0.4687, 0.4772, 0.4834]

# Create a figure and axis
plt.figure(figsize=(8, 6))

# Plot the data
plt.plot(models, bleu_scores, marker='o', linestyle='-', color='b', label='BLEU Score')

# Add labels and title
plt.xlabel('Model Iteration')
plt.ylabel('BLEU Score')
plt.title('BLEU Scores Across Model Iterations')

# Display grid for better readability
plt.grid(True)

# Show the plot
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()
