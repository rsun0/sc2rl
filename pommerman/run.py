import sys
sys.path.insert(0, "../interface/")

from abstract_core import Experiment, RunSettings
from custom_env import PommermanEnvironment
from simple_agent import SimpleAgent

if __name__ == '__main__':
    env = PommermanEnvironment(render=True)

    run_settings = RunSettings(
        num_episodes=1,
        num_epochs=1,
        batch_size=1,
        train_every=1024,
        save_every=10240,
        graph_every=0,
        averaging_window=100
    )

    agent = SimpleAgent()

    experiment = Experiment([agent], env, run_settings)
    experiment.train()