from typing import List
import torch.nn as nn
import torch

from src.noises.base import do_not_add_noise


class Add(nn.Module):
    """A simple addition layer for residual connections."""

    def __init__(self) -> None:
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors."""
        return x + y


class BasicBlock(nn.Module):
    """A basic residual block.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride for the first convolution.
        first (bool): Whether this is the first block in the sequence.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        first: bool = False,  # Here it is not used
    ) -> None:
        super().__init__()
        if stride != 1:
            conv = nn.Conv2d(inplanes, planes, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(planes)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = None
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.add = Add()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu2(out)
        return out


class Output(nn.Module):
    """An output head combining AdaptiveAvgPool2d and Linear layer.

    Args:
        inplanes (int): Number of input channels.
        output_size (int): Number of output classes.
    """

    def __init__(self, inplanes: int, output_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(inplanes, output_size))
        do_not_add_noise(self.layers[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


class Blocks(nn.Module):
    """A sequence of residual blocks.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        blocks (int): Number of residual blocks.
        block (BasicBlock): Residual block type.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        block: BasicBlock = BasicBlock,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(blocks):
            self.layers.append(
                block(
                    inplanes=inplanes if i == 0 else planes * block.expansion,
                    planes=planes,
                    stride=stride if i == 0 else 1,
                    first=i == 0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


class _ResNet(nn.Module):
    """A ResNet model.

    Args:
        layers (List[int]): Number of residual blocks in each layer.
        planes (List[int]): Number of output channels in each layer.
        strides (List[int]): Stride for the first convolution in each layer.
        output_size (int): Number of output classes.
        block (BasicBlock): Residual block type.
    """

    def __init__(
        self,
        layers: List[int] = [2, 2, 2, 2],
        planes: List[int] = [32, 64, 128, 256],
        strides: List[int] = [1, 2, 2, 2],
        output_size: int = 10,
        block: BasicBlock = BasicBlock,
    ) -> None:
        super().__init__()
        self.output_size = output_size

        assert len(planes) == len(layers)
        assert len(strides) == len(layers)
        assert len(planes) == len(strides)

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(3, planes[0], kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes[0]),
                nn.ReLU(inplace=True),
            ]
        )

        for i in range(len(layers)):
            self.layers.append(
                Blocks(
                    inplanes=planes[0] if i == 0 else planes[i -
                                                             1] * block.expansion,
                    planes=planes[i],
                    blocks=layers[i],
                    stride=strides[i],
                    block=block,
                )
            )

        self.layers.append(Output(planes[-1] * block.expansion, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet(_ResNet):
    """A ResNet model."""

    def __init__(
        self,
        layers: List[int] = [2, 2, 2, 2],
        planes: List[int] = [32, 64, 128, 256],
        strides: List[int] = [1, 2, 2, 2],
        output_size: int = 10,
    ) -> None:
        super().__init__(
            layers=layers,
            planes=planes,
            strides=strides,
            output_size=output_size,
            block=BasicBlock,
        )
