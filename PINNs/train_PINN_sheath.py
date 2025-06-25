import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import json

training_data_file = "training_data.txt"

test_data_file = "testing_data.txt"

loss_output = "loss_vlasov.npy"

def read_data_from_txt(fname):
    data = []
    with open(fname, "r") as file:
        for line in file:
            data.append(json.loads(line))

    input_tensor = torch.zeros((len(data), 3))
    output_tensor = torch.zeros((len(data), 1))
    jac_tensor = torch.zeros((len(data), 3))

    for i in range(len(data)):
        datapoint = data[i]
        inputs = datapoint["input"]
        j = datapoint["output"]["j"]
        jac = datapoint["jacobian"]["j"]
        input_tensor[i, :] = torch.tensor([inputs["n"], inputs["T"], inputs["Vp"]])
        output_tensor[i, 0] = j
        jac_tensor[i, :] = torch.tensor([jac["n"], jac["T"], jac["Vp"]])

    return input_tensor, output_tensor, jac_tensor


input_tensor, output_tensor, jac_tensor = read_data_from_txt(training_data_file)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return -torch.exp(self.net(torch.log(x)))

    def save_weights_np(self):
        w1 = self.net[0].weight.detach().numpy()
        b1 = self.net[0].bias.detach().numpy()
        w2 = self.net[2].weight.detach().numpy()
        b2 = self.net[2].bias.detach().numpy()
        w3 = self.net[4].weight.detach().numpy()
        b3 = self.net[4].bias.detach().numpy()
        w4 = self.net[6].weight.detach().numpy()
        b4 = self.net[6].bias.detach().numpy()
        np.savez("model_weights.npz", w1, b1, w2, b2, w3, b3, w4, b4)

#neural network functions
def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
        grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def compute_loss(model: nn.Module, x: torch.Tensor):
    #evaluating u(x), f(x), du/dx at training points
    x.requires_grad_(True)
    u = model(x)

    #data residual
    log_residual = torch.log(torch.abs(u)) - torch.log(torch.abs(output_tensor))
    interior_loss = torch.mean(log_residual ** 2)

    # jacobian residual
    #jac_residual = torch.log(1 + torch.abs(du)) - torch.log(1 + torch.abs(jac_tensor))
    #jac_loss = torch.mean(jac_residual ** 2)
    #calculating boundary loss for problem


    return interior_loss



loss_over_time = []

def train_PINN(x_train):
    # Training the PINN
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Iterate training over epochs
    for epoch in range(100000):
        loss = compute_loss(model, x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_over_time.append(loss_value)
        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss_value:.6f}")

        if loss_value < 0.01:
            "Loss <0.01, terminating early"
            break


    return model

model = train_PINN(input_tensor)

test_input, test_output, _ = read_data_from_txt(test_data_file)

print("Testing model!")

for i in range(test_input.shape[0]):
    input = test_input[i]
    print(f"input = {input}")
    print(f"output = {model(input).detach().numpy()}, target = {test_output[i].numpy()}")


model.save_weights_np()



# load the neural network
np_weights = np.load("model_weights.npz")
linear1 = np_weights["arr_0"]
bias1 = np_weights["arr_1"]
linear2 = np_weights["arr_2"]
bias2 = np_weights["arr_3"]
linear3 = np_weights["arr_4"]
bias3 = np_weights["arr_5"]
linear4 = np_weights["arr_6"]
bias4 = np_weights["arr_7"]

pinn_inputs = test_input[0].numpy()

x1 = linear1 @ np.log(pinn_inputs)
x2 = linear2 @ np.tanh(x1 + bias1)
x3 = linear3 @ np.tanh(x2 + bias2)
x4 = linear4 @ np.tanh(x3 + bias3)

print(-np.exp(x4 + bias4))

np.save(loss_output,np.array(loss_over_time))

plt.semilogy(loss_over_time)
plt.xlabel("epoch")
plt.ylabel("loss")

plt.show()
