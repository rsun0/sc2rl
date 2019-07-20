from pysc2.env import sc2_env
from pysc2.lib import features, protocol
from pysc2.actions import FUNCTIONS
import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)



env = sc2_env.SC2Env(
    map_name="DefeatRoaches",
    agent_interface_format=features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=84,minimap=84),
        use_feature_units=True
    ),
    step_mul=8,
    visualize=False,
    game_steps_per_episode=None
)


for i in range(100):
    base_action = np.random.randint(len(FUNCTIONS))
    print("\n\n", FUNCTIONS[base_action], "\n\n")
    
