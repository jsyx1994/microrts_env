from torch.utils.tensorboard import SummaryWriter

from rts_wrapper.envs.utils import encoded_utt_dict
from algo.model import ActorCritic
from torch import optim
import torch
from sl.sl_data_processor import get_data


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    storage = get_data()
    ac = ActorCritic(6, 6)
    writer = SummaryWriter()


    # input()
    ac.to(device)

    iteration = 1000000
    batch_size = 128
    criteria = torch.nn.NLLLoss()
    optimizer = optim.Adam(ac.parameters(), lr=10e-6)

    for i in range(iteration):

        loss = 0
        sample_dict = storage.sample(batch_size)
        for key in sample_dict:
            if sample_dict[key]:
                spatial_features, unit_features, actions = sample_dict[key]

                spatial_features = torch.from_numpy(spatial_features).float().to(device)
                unit_features = torch.from_numpy(unit_features).float().to(device)
                encoded_utt = torch.from_numpy(encoded_utt_dict[key]).unsqueeze(0).float().repeat(unit_features.size(0), 1).to(device)
                # cat utt and the individual feature together
                unit_features = torch.cat([unit_features, encoded_utt], dim=1)
                actions = torch.from_numpy(actions).long().to(device)
                # print(states.device, units.device)
                probs = ac.actor_forward(key, spatial_features, unit_features)
                # print(probs.device)
                # input()
                # _actions = torch.zeros_like(prob)
                # for i in range(len(actions)):
                #     _actions[i][actions[i]] = 1

                log_probs = torch.log(probs)
                loss += criteria(log_probs, actions)
        if i % 100 == 0:
            writer.add_scalar("all losses", loss, i)
            print("iter{}, loss:{}".format(i, loss))

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), .1)
        optimizer.step()
        # print(prob[i])

    torch.save(ac.state_dict(), '../models/100k.pth')


if __name__ == '__main__':
    main()