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
        ai1_type='socketAI',
        ai2_type='Passive',
        map_path='maps/16x16/basesWorkers16x16.xml',
        render=1,
        height=16,
        width=16,
        max_cycles=2000,
        max_episodes=2,
        # auto_port=True,
    )}
)

register(
    id='CurriculumBaseWorker-v0',
    entry_point='rts_wrapper.envs:MicroRts',
    kwargs={'config': Config(
        ai1_type='socketAI',
        ai2_type='Passive',
        map_path='maps/6x6/baseTwoWorkersMaxResources6x6.xml',
        height=6,
        width=6,
        render=1,
        max_cycles=3000,
        max_episodes=10000,

    )}
)

register(
    id='TwoWorkersAndBaseWithResources-v0',
    entry_point='rts_wrapper.envs:MicroRts',
    kwargs={'config': Config(
        ai1_type='socketAI',
        ai2_type='Passive',
        map_path='maps/4x4/baseTwoWorkers4x4.xml',
        height=4,
        width=4,
        render=1,
        max_cycles=300,
        max_episodes=10000,

    )}
)

register(
    id='OneWorkerAndBaseWithResources-v0',
    entry_point='rts_wrapper.envs:MicroRts',
    kwargs={'config': Config(
        ai1_type='socketAI',
        ai2_type='Passive',
        map_path='maps/6x6/baseWorkerResources6x6.xml',
        height=6,
        width=6,
        render=1,
        max_cycles=300,
        max_episodes=10000,

    )}
)
# self-play
register(
    id="SelfPlayOneWorkerAndBaseWithResources-v0",
    entry_point='rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type="socketAI",
        ai2_type="socketAI",

        map_path="maps/6x6/baseWorkerResources6x6.xml",
        height=6,
        width=6,
        render=1,
        max_cycles=3000,
        max_episodes=10000,

        )}
)

# eval
register(
    id='Eval-v0',
    entry_point='rts_wrapper.envs:MicroRts',
    kwargs={'config': Config(
        period=1,
        ai1_type='socketAI',
        ai2_type='Random',
        map_path='maps/6x6/baseWorkerResources6x6.xml',
        height=6,
        width=6,
        render=1,
        max_cycles=3000,
        max_episodes=1000,

    )}
)

# test
register(
    id="LargeMapTest-v0",
    entry_point='rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type="socketAI",
        ai2_type="socketAI",

        map_path="maps/16x16/basesWorkers16x16.xml",
        height=16,
        width=16,
        render=1,
        max_cycles=5000,
        max_episodes=10000,
        )}
)