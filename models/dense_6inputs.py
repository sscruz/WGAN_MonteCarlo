import torch 
from torch import nn 

class NNgenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear( 10, 40),
            nn.Sigmoid(),
            nn.Linear( 40, 20),
            nn.Sigmoid(),
            nn.Linear( 20, 6),
        )
        self.output=nn.ReLU(True)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class NNcritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear( 6, 40),
            nn.ReLU(True),
            nn.Linear(40,20),
            nn.ReLU(True),
            nn.Linear(20,1),
        )
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

            
class Dense6inputs():

    def __init__(self):
        channels=1
        self.G=NNgenerator()
        self.D=NNcritic()
        self.latent_space_generator=lambda batch_size : torch.randn(batch_size, 10)
