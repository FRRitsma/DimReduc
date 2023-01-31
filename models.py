import torch
import torch.nn as nn
import torch.nn.functional as F


class EncNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        layers: int,
        hidden_size: int,
        dropout_rate: float = 0.05,
    ) -> None:

        super(EncNet, self).__init__()

        # TODO: batch normalization layer?

        # Storing layers in a list:
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Adding layers:
        for i in range(layers):
            in_count = hidden_size
            out_count = hidden_size
            # First layer:
            if i == 0:
                in_count = input_size
            # Final layer:
            if i + 1 == layers:
                out_count = input_size
            self.encoder.append(nn.Linear(in_count, out_count))
            self.decoder.append(nn.Linear(in_count, out_count))

        self.dropout = nn.Dropout(p=dropout_rate)

    def encode(self, x) -> torch.Tensor:
        # First layer (non linear/no dropout):
        x = torch.sigmoid(self.encoder[0](x))

        # Middle layers (non linear/dropout):
        for layer in self.encoder[1:-1]:
            x = self.dropout(x)
            x = torch.sigmoid(layer(x))

        # Final layer (linear/no dropout):
        x = self.encoder[-1](x)

        return x

    def decode(self, x) -> torch.Tensor:
        # First layer (non linear/no dropout):
        x = torch.sigmoid(self.decoder[0](x))

        # Middle layers (non linear/dropout):
        for layer in self.decoder[1:-1]:
            x = self.dropout(x)
            x = torch.sigmoid(layer(x))

        # Final layer (linear/no dropout):
        x = self.decoder[-1](x)

        return x

    def forward(self, x) -> torch.Tensor:

        x = self.encode(x)
        x = self.decode(x)

        return x
