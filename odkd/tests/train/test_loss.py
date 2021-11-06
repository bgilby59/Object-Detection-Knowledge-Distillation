from odkd.train.loss import (
    MultiBoxLoss,
    NetwithLoss,
    NetwithDistillatedLoss
)


def test_multiboxloss(config, predictions, targets):
    loss = MultiBoxLoss(config)
    x = loss(predictions, targets)
    print(x.item())
    assert len(x.shape) == 0


def test_netwithloss(config, ssdlite, dataloader):
    loss = NetwithLoss(config, ssdlite)
    for images, targets in dataloader:
        l = loss.forward(images, targets)
        print(l.item())


def test_netwithdistillatedloss(config, ssdlite, dataloader):
    loss = NetwithDistillatedLoss(config, ssdlite, ssdlite)
    for images, targets in dataloader:
        l = loss.forward(images, targets)
        print(l.item())
