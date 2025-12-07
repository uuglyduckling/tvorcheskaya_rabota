from pathlib import Path
import matplotlib.pyplot as plt


def plot_metrics(history):
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    #Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.plot(epochs, history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(["Train", "Validation"])
    plt.grid(True)

    loss_path = output_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()

    #Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"])
    plt.plot(epochs, history["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend(["Train", "Validation"])
    plt.grid(True)

    acc_path = output_dir / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {output_dir.resolve()}")
