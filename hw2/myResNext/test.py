import torch

from myResNext.resnext import resnext50


def simple_test():
    print('Start simple test')
    num_classes = 20
    batch = 4
    tensor = torch.rand(batch, 3, 224, 224)
    model = resnext50(num_classes=num_classes)
    output = model(tensor)
    assert output.size() == (batch, num_classes)
    print('Simple test success')


simple_test()
