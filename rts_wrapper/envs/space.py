from gym.spaces import Dict



class DictSpace(Dict):
    def sample(self):
        return super().sample()