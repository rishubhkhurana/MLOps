from torch import nn
import timm

class CIFARNet(nn.Module):
    def __init__(
        self,
        modelname: str = 'resnet18',
        pretrained: bool = True,
        output_size: int = 10,
    ):
        super().__init__()
        self.model = timm.create_model(modelname, pretrained = pretrained, num_classes=output_size)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    _ = CIFARNet()
