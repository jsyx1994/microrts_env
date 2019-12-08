from algo.model import ActorCritic
import torch
import os
from algo.eval import evaluate_game


def load_model(path, height, width) -> ActorCritic:
    model = ActorCritic(height, width)
    model.load_state_dict(torch.load(path))
    return model


def main():
    model = load_model(os.path.join('../models', '100k.pth'), 6, 6)
    evaluate_game('Eval-v0', model)


if __name__ == '__main__':
    main()