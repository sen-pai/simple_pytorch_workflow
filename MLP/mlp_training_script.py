from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import DummyDataset
from mlp_model import SimpleMLP

dataset = DummyDataset()

dataloader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

mse_loss = nn.MSELoss()

def train_model(model, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for input, out in tqdm(dataloader):
            input = input.to(device)
            out = out.to(device)

            # forward
            optimizer.zero_grad()

            pred_out = model(input)

            loss = mse_loss(out, pred_out)

            loss.backward()

            optimizer.step()

        print("loss: ", loss)
        print("out: ", out[0], pred_out[0])


        # deep copy the model
        # if epoch % args.save_freq == 0:
        #     print("saving model")
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     weight_name = "model_weights_" + str(epoch) + ".pt"
        #     torch.save(best_model_wts, weight_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
train_model(model, optimizer, num_epochs=300)
