"""
MIT License

Copyright (c) 2024 Carlos Tojal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import yaml
import sys
import os
sys.path.append(".")
from models.ndnetpp.ndnetpp_cls import NDNetppClassifier
from datasets.ModelNet import ModelNet
from libs.pyprogress.ProgressBar import ProgressBar

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ndnetpp_cls.yaml", 
                        help="Path to the YAML network configuration file")
    parser.add_argument("--data_path", type=str,
                        help="Path to the ModelNet40 dataset", required=True)
    args = parser.parse_args()

    # check the configuration file for existance and parse
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"The configuration file \"{args.config}\" does not seem to exist.")
    # open the file descriptor
    try:
        f = open(args.config)
    except Exception as e:
        raise RuntimeError(f"Error opening configuration file: {repr(e)}")
    # parse the configuration file
    try:
        config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading the network configuration file: {repr(e)}")
    # close the file descriptor
    try:
        f.close()
    except Exception as e:
        raise RuntimeError(f"Error closing configuration file: {repr(e)}")

    print(config)

    # detect the device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # build the model
    model = NDNetppClassifier(config)
    model = model.to(device)
    print(model)

    # load the dataset
    train_dataset = ModelNet(root_dir=args.data_path, stage="train")
    test_dataset = ModelNet(root_dir=args.data_path, stage="test")

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # create the criterion (cross-entropy loss due to being a classifier)
    criterion = nn.CrossEntropyLoss()

    # create the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=float(config['learning_rate']), momentum=float(config['momentum']),
                            weight_decay=float(config['weight_decay']))

    # iterate the epochs
    for epoch in range(int(config['num_epochs'])):

        # iterate the training set
        train_bar = ProgressBar(len(train_loader))

        for i, sample in train_loader:
            pcd, cls = sample.to(device)

            # zero the gradients
            optim.zero_grad()

            # create the target tensor
            target: torch.Tensor = nn.functional.one_hot(cls, num_classes=40).float()

            # forward pass
            pred = model(pcd)

            # compute the loss
            loss = criterion(pred, target)

            # backward
            loss.backward()
            optim.step()

            train_bar.update(i, extra=f"loss={loss.item()}")

    # TODO: save pth files with the weights

    # TODO: testing samples

    # exit with code 0
    exit(0)

