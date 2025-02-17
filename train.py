import argparse

import torch
import torch.optim as optim

from model import MyLinear
from trainer import Trainer

from utils import load_data

def argument_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True, help="model_file_name")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1, help="cuda_gpu_id")
    p.add_argument("--lr", type=float, default=1e-3, help="learning_rate")
    p.add_argument("--n_epochs", type=int, default=2000, help="number_of_epochs")
    p.add_argument("--interval", type=int, default=100, help="number_of_print_interval")

    config = p.parse_args()

    return config

def main(config):
    # Train 진행할 Device 선택 - cpu or gpu
    device = torch.device('cpu') if config.gpu_id == -1 else torch.device(f"cuda:{config.gpu_id}")
    print(f"Device : {device}")

    # Load Data
    x, y = load_data(is_full=False)

    print(f"Train Data : {x.shape}")
    print(f"Target Data : {y.shape}")

    input_size = x.size(-1)
    output_size = y.size(-1)

    # Define Model
    model = MyLinear(input_size, output_size)

    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    print(f"Model : {model}")
    print(f"Optimizer : {optimizer}")

    trainer = Trainer(model, optimizer)

    trainer.train(x, y, config)

    # Save best model weights
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)



if __name__ == "__main__":
    config = argument_parser()
    main(config)
