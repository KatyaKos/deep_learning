import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from vae.vae import VAE, loss_function
from vae.trainer import Trainer


def get_config():
    parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_dcgan.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=32,
                        help='size of images to generate')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=100)
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def main():
    config = get_config()
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.Scale(config.image_size), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root=config.data_root, download=True,
                                     transform=transform, train=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True)
    dataset = datasets.CIFAR10(root=config.data_root, download=True,
                                    transform=transform, train=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                              num_workers=0, pin_memory=True)

    net = VAE(config.image_size)

    trainer = Trainer(model=net,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         optimizer=Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                         loss_function=loss_function,
                         device='cpu')

    for epoch in range(0, config.epochs):
        trainer.train(epoch, config.log_metrics_every)
        trainer.test(epoch, config.log_metrics_every)


if __name__ == '__main__':
    main()
