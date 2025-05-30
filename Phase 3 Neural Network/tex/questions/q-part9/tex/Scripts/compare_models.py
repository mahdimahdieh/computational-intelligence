if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data_no_augmentation()

    configs = [
        {'name': 'GD-Sigmoid', 'activations': ['sigmoid', 'sigmoid'], 'opt': {'lr': 0.01, 'momentum': 0}},
        {'name': 'Momentum-ReLU', 'activations': ['relu', 'sigmoid'], 'opt': {'lr': 0.01, 'momentum': 0.9}},
    ]

    trained_models = {}
    results = {}

    for cfg in configs:
        print(f"\n== Training {cfg['name']} Without Augmentation ==")
        model = NeuralNetwork(layer_dims=[3072, 64, 1], activations=cfg['activations'], optimizer_cfg=cfg['opt'])
        train_loader = SimpleLoader(x_train, y_train)
        model.train(train_loader, epochs=10, print_freq=2)
        results[cfg['name']] = model.history
        trained_models[cfg['name']] = model

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

    # Evaluation
    print("\n-- Test Set Evaluation --")
    for name, model in trained_models.items():
        yh, _ = model.forward(x_test)
        yp = (yh >= 0.5).astype(int).flatten()
        print(f"\n{name}:\n", confusion_matrix(y_test, yp))
        print(classification_report(y_test, yp, digits=4))