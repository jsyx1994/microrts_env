from gym.spaces import Dict

class BaseActions:
    TYPE_NONE = 0





class DictSpace(Dict):
    def sample(self):
        return super().sample()