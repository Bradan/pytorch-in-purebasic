import torch
import torch.utils.data
import torchvision

import model


def train(model, dataset):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=8)
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for epoch in range(3):
        loss_sum = 0.0
        count = 0
        model.train()

        for inputs, outputs in data_loader:
            batch_size = inputs.size(0)

            outputs = torch.nn.functional.one_hot(outputs, 10).to(dtype=torch.float)
            result = model(inputs)

            loss = loss_fn(result, outputs)
            loss_sum += float(loss)
            count += batch_size

            loss.backward()
            optim.step()

            if count >= 10000:
                break

        print(f"Epoch {epoch} finished with loss {loss_sum / count}")
        model.eval()
        torch.jit.save(torch.jit.script(model), f"epoch_{epoch}.pt")


def validate(model, dataset):
    model.eval()
    with torch.no_grad():
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=8,
                                                  shuffle=True,
                                                  num_workers=8)
        loss_fn = torch.nn.MSELoss()

        loss_sum = 0.0
        count = 0

        for inputs, outputs in data_loader:
            batch_size = inputs.size(0)

            outputs = torch.nn.functional.one_hot(outputs, 10).to(dtype=torch.float)
            result = model(inputs)

            loss = loss_fn(result, outputs)
            loss_sum += float(loss)
            count += batch_size

        print(f"Validation loss: {loss_sum / count}")


if __name__ == '__main__':
    model = model.Model()

    mnist_data_train = torchvision.datasets.MNIST("./data",
                                                  train=True,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)
    mnist_data_test = torchvision.datasets.MNIST("./data",
                                                 train=False,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)

    train(model, mnist_data_train)
    validate(model, mnist_data_test)
