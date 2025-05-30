# Plot Loss
    plt.figure()
    for name, hist in results.items():
        plt.plot(hist['train_loss'], label=name)
    plt.title('Training Loss Comparison (No Augmentation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Accuracy
    plt.figure()
    for name, hist in results.items():
        plt.plot(hist['train_acc'], label=name)
    plt.title('Training Accuracy Comparison (No Augmentation)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()