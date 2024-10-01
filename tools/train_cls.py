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
from argparse import ArgumentParser
import yaml
import sys
import os
sys.path.append(".")
from models.ndnetpp.ndnetpp_cls import NDNetppClassifier

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

    # dummy forward
    """
    pcd = torch.rand(1, 2000, 3, device=device)
    print(pcd.shape)
    out = model(pcd)
    """
    

    



    # TODO: build the model
    # TODO: load the dataset and create a dataloader
    # TODO: create the optimizer and criterion
    # TODO: training loop
    # TODO: save pth files with the weights

    raise NotImplementedError("Classifier training loop not implemented")

    # exit with code 0
    exit(0)

