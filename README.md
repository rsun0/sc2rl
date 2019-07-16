# Reinforcement Learning in StarCraft II
By Ray Sun, Michael McGuire, and David Long

This research project applies reinforcement learning to the real-time strategy game StarCraft II through the PySC2 environment.
We are currently focusing on micro and the minigames included in PySC2, specifically DefeatRoaches, in which the agent must control
a group of marines to kill a group of enemy roaches. We are currently experimenting with
[PPO](https://openai.com/blog/openai-baselines-ppo/) and [graph convolutions](https://arxiv.org/abs/1609.02907), and plan on applying
[continuous control](https://arxiv.org/abs/1509.02971v5) soon.

## Directories
* `old` - The first version of our experiment code, using TensorFlow and modified state and action spaces
* `PPO` - The second version of our experiment code, running PPO with LSTMs. Switched from TensorFlow to PyTorch.
* `interface` - A general training framework that allows models and environments to be changed easily, allowing experiments to iterate faster.
* `experiments` - The third and current version of our experiment code, using the framework in `interface`. Current experiment
uses PPO with graph convolutional layers in the network.

## Acknowledgements
### Libraries
* [PySC2](https://github.com/deepmind/pysc2), DeepMind's StarCraft II Learning Environment
* [PyTorch](https://github.com/pytorch/pytorch)
* [TensorFlow](https://github.com/tensorflow/tensorflow)

This work utilizes resources supported by the National Science Foundation’s Major Research Instrumentation program,
grant #1725729, as well as the University of Illinois at Urbana-Champaign.
