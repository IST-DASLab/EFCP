import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.layer = torch.nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        return self.layer(x.float())
