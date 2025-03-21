import matplotlib.pyplot as plt
import seaborn as sns
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


#########################################
# Overlapped Histograms Plot routines
#########################################
@jaxtyped(typechecker=beartype)
def plot_overlapped_histograms(
    errors: list[Float[Tensor, "n_samples"]],
    str_norm: str,
    labels: list[str] = None,
):

    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid", palette="deep")
    plt.figure(figsize=(10, 6), layout="constraint")

    if labels is None:
        labels = [f"Error {i+1}" for i in range(len(errors))]

    for error, label in zip(errors, labels):
        sns.histplot(
            error.to("cpu").numpy(),
            bins=100,
            kde=True,
            edgecolor="black",
            label=label,
            alpha=0.6,
        )

    plt.xlabel("Relative Error", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title(
        f"Histogram of the Relative Error in Norm {str_norm}", fontsize=14, pad=20
    )
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.show()

    # Resets the style to default
    plt.style.use("default")
