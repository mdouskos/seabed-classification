import torch
import numpy as np


def create_model(D_in, H1, H2, num_classes, gpu=False):
    model = torch.nn.Sequential(
        # torch.nn.BatchNorm1d(D_in),
        torch.nn.Linear(D_in, H1),
        torch.nn.ReLU(),
        # torch.nn.BatchNorm1d(H1),
        torch.nn.Linear(H1, H2),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(H2, num_classes),
    )

    if gpu:
        model = model.cuda()

    return model


def model_eval(model, val_loader, gt_available=True, gpu=False):
    y_pred = np.empty((0,), dtype=np.uint8)
    accuracy = 0
    if gt_available:
        correct = 0
        total = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            samples = data[0]
            samples = samples.type(torch.FloatTensor)
            if gpu:
                samples = samples.cuda()

            if gt_available:
                labels = data[1]
                labels = labels.type(torch.LongTensor)
                if gpu:
                    labels = labels.cuda()

            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = np.append(y_pred, predicted.detach().cpu().numpy())

            if gt_available:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    if gt_available:
        accuracy = correct / total

    return y_pred, accuracy


def train_network(model, train_set, val_set, epochs=10, batch_size=2048, gpu=False):

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            samples, labels = data
            samples = samples.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)
            if gpu:
                samples = samples.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(samples)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print("[{}, {}] loss: {}".format(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        y_pred, acc = model_eval(model, val_loader, gpu=gpu)
        print("Accuracy at epoch {}: {}".format(epoch + 1, 100 * acc))

    return model, y_pred
