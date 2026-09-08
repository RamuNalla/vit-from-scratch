import json
import os
import matplotlib.pyplot as plt

def plot_training_curves(json_path: str = "docs/assets/training_history.json"):
    if not os.path.exists(json_path):
        print(f"Error: Training history file not found at {json_path}. Run training first!")
        return

    with open(json_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Set up styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, marker='s', markersize=4)
    ax1.set_title("Cross-Entropy Loss (From Scratch ViT)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc (%)", color="#1f77b4", linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, history["val_acc"], label="Val Acc (%)", color="#ff7f0e", linewidth=2, marker='s', markersize=4)
    ax2.set_title("Classification Accuracy (%)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epochs", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    output_path = "docs/assets/training_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Performance curves generated and saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_training_curves()