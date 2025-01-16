import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style to white background and deep color palette
sns.set(style="white", palette="deep")

# List of CSV files (replace with your actual file paths)
csv_files = [
    "Train loss_default.csv",
    "Train loss.csv",
    "Test rel. L^1 error_default.csv",
    "Test rel. L^1 error.csv",
]

# Labels for the loss functions (replace with your actual labels)
labels = [
    "Train loss default hyper-params.",
    "Train loss best hyper-params.",
    "Test loss default hyper-params.",
    "Test loss best hyper-params",
]

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Read and plot each CSV file
for i, (file, label) in enumerate(zip(csv_files, labels)):
    # Read the CSV file
    df = pd.read_csv(file)

    s = 5
    df = df.iloc[::s, :]

    # Plot the loss function
    sns.lineplot(data=df, x=df.columns[1], y=df.columns[2], label=label)

# Add labels and title
plt.xlabel("Epoch/Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss Functions Comparison")
plt.grid()
plt.legend()

# Show the plot
plt.show()
# plt.savefig("loss_functions.png", dpi=300, bbox_inches="tight")
