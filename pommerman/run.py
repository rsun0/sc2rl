import sys
sys.path.insert(0, "../interface/")

from abstract_core import Experiment, RunSettings
from custom_env import PommermanEnvironment
from simple_agent import SimpleAgent
from noop_agent import NoopAgent
from mcts_agent import MCTSAgent, MCTSWrapperAgent

if __name__ == '__main__':
    env = PommermanEnvironment(render=True, num_agents=2, game_state_file='start.json')

    run_settings = RunSettings(
        num_episodes=3,
        num_epochs=1,
        batch_size=1,
        train_every=1024,
        save_every=10240,
        graph_every=0,
        averaging_window=100
    )

    agent1 = SimpleAgent()
    # agent2 = MCTSWrapperAgent(MCTSAgent())
    agent2 = SimpleAgent()

    experiment = Experiment([agent1, agent2], env, run_settings)
    experiment.train()