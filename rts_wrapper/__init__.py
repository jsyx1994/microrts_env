from gym.envs.registration import register
import os
from .datatypes import Config

base_dir_path = os.path.dirname(os.path.realpath(__file__))
# register will call this:
"""A specification for a particular instance of the environment. Used
to register the parameters for official evaluations.
Args:
    id (str): The official environment ID
    entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
    reward_threshold (Optional[int]): The reward threshold before the task is considered solved
    kwargs (dict): The kwargs to pass to the environment class
    nondeterministic (bool): Whether this environment is non-deterministic even after seeding
    tags (dict[str:any]): A set of arbitrary key-value tags on this environment, including simple property=True tags
    max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of
Attributes:
    id (str): The official environment ID
"""

register(
    id='Microrts-v0',
    entry_point='rts_wrapper.envs:MicroRts',
    kwargs={'config': Config(
        ai1_type='myAI',
        ai2_type='passive',
        map_path='maps/16x16/basesWorkers16x16.xml',
        render=True,
        auto_port=True,
        microrts_path='~/microrts_env',

    )}
)