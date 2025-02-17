import pandas as pd

import torch

from model import MyLinear

from utils import load_data

def load(model_fn, device):
    d = torch.load(model_fn, map_location=device, weights_only=False)

    return d['model'], d['config']

def test(model, x, y, to_be_shown=True):
    with torch.no_grad():
        y_hat = model(x)

        if to_be_shown:
            import seaborn as sns
            import matplotlib.pyplot as plt

            y = y.to('cpu')
            y_hat = y_hat.to('cpu')

            df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(), columns=["y", "y_hat"])
            sns.pairplot(df, height=5)
            plt.show()

def main():
    model_fn = "./model/model.pth"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    x, y = load_data(is_full=False)
    x, y = x.to(device), y.to(device)

    input_size = x.size(-1)
    output_size = y.size(-1)

    model_dict, config = load(model_fn, device)

    model = MyLinear(input_size, output_size).to(device)

    model.load_state_dict(model_dict)

    test(model, x, y, to_be_shown=True)

if __name__ == "__main__":
    main()