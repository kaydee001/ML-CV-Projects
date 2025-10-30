import matplotlib.pyplot as plt

def plot_loss_history(loss_history, val_acc_history, save_path="loss_curve.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # plotting training loss -> decresaes overtime
    ax1.plot(loss_history, marker='o', color="blue")
    ax1.set_title("training loss over epochs")    
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True)

    # plotting validation accuracy -> increases overtime
    ax2.plot(val_acc_history, marker='o', color="green")
    ax2.set_title("validation accuracy over epochs")    
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"loss plot saved to {save_path}")