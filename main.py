# %%
import torch
import torch.nn as nn
import sklearn.datasets
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import plotly.express as px
from dataclasses import dataclass
from models import EncNet

X = sklearn.datasets.make_swiss_roll(n_samples=int(1e4))[0]

# %%


def embedding_activation(encoding: torch.Tensor) -> torch.Tensor:
    return 1 - torch.exp(-(encoding**2))


def custom_loss(
    target: torch.Tensor,
    model: EncNet,
    lagrangian: float,
) -> torch.Tensor:

    # Form encoding:
    encoding = model.encode(target)
    # Add noise:
    noise = 1 - 2 * torch.rand(target.shape[0], target.shape[1])
    # Remove noise from channel zero:
    noise[:, 0] = 0
    # Decoded encoding with additive noise is the model output:
    output = model(encoding + noise)

    # Collect accuracy:
    accuracy_loss = F.mse_loss(output, target)

    # Collect encoding dimension loss:
    encoding_loss = lagrangian * torch.mean(
        torch.abs((torch.arange(target.shape[1]) + 1) * embedding_activation(encoding)),
    )

    # Complete loss:
    complete_loss = accuracy_loss + encoding_loss

    return complete_loss


def embedding_loss(embedding: torch.Tensor) -> torch.Tensor:
    emb_loss = torch.mean(
        torch.abs(
            (torch.arange(embedding.shape[1]) + 1) * embedding_activation(embedding)
        ),
    )
    return emb_loss


# %%
# HARDCODES:
# Optimization hardcodes:
STARTING_LEARNING_RATE = float(1e-2)
LEARNING_RATE_DECAY = 0.995
MIN_LEARNING_RATE = float(1e-5)

WEIGHT_DECAY = float(1e-8)
BATCH_SIZE = 100
N_EPOCHS = int(1e5)


# Model hardcodes:
INPUT_SIZE = 3
N_LAYERS = 3
N_NEURONS = 50

# Defining model:
model = EncNet(INPUT_SIZE, N_LAYERS, N_NEURONS, dropout_rate=0.1)

# Defining optimizer:
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1,
    weight_decay=WEIGHT_DECAY,
)

# Defining learning rate scheduler:
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: max(
        STARTING_LEARNING_RATE * LEARNING_RATE_DECAY**epoch, MIN_LEARNING_RATE
    ),
)

# %% Define lagrangian increase:
@dataclass
class Lagrangian:
    lagrangian: float = float(1e-8)
    target_accuracy: float = float(1e-2)

    def __init__(self):
        self.history = []

    def assess(
        self,
        accuracy_loss: float,
        embedding_loss: float,
        lagrangian: float,
    ) -> None:

        # Type conversions:
        accuracy_loss = float(accuracy_loss)
        accuracy_loss = max(accuracy_loss, self.target_accuracy)
        embedding_loss = float(embedding_loss)

        # Total current loss:
        current_loss = accuracy_loss + lagrangian * embedding_loss
        self.history.append(current_loss)
        
        # Recommended lagrangian:
        r_lagrangian = (current_loss - accuracy_loss)/self.target_accuracy
        r_lagrangian = max(r_lagrangian, float(1e-8))

        return r_lagrangian


# %% Optimization loop:

Xt = torch.normal(mean=torch.zeros(int(1e4), 3), std=torch.ones(3))
Xv = torch.normal(mean=torch.zeros(int(1e4), 3), std=torch.ones(3))
# Xt = torch.Tensor(X)
for epoch in range(N_EPOCHS):

    # Initialize epoch:
    perm = torch.randperm(len(Xt))
    tot_loss = 0

    # Adapt learning rate:
    scheduler.step()

    for b in range(0, len(Xt), BATCH_SIZE):
        # Collect batch:
        batch = Xt[perm[b : b + BATCH_SIZE], :]

        # Create encoding:
        encoding = model.encode(batch)
        emb_loss = embedding_loss(encoding)

        # Create decoding:
        noise = 1 - 2 * torch.rand(encoding.shape[0], encoding.shape[1])
        noise[:, 0] = 0
        output = model(encoding + noise)
        acc_loss = F.mse_loss(output, batch)

        # Collected loss:
        loss = 0.0001 * emb_loss + acc_loss

        # Optimization:
        optimizer.zero_grad()
        # loss = custom_loss(batch, model, float(1e-2))
        loss.backward()
        optimizer.step()

        # Gather loss to display progress:
        tot_loss += loss.detach()

    if epoch % 10 == 0:
        print(tot_loss)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:E}")


# %%
Xr = model.encode(Xt).detach().numpy()
# Xr = model(Xt).detach().numpy()

fig = px.scatter_3d(x=Xr[:, 0], y=Xr[:, 1], z=Xr[:, 2])
fig.update_traces(marker_size=1)
fig.show()
