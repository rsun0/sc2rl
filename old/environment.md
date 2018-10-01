# Modified environment
To get our experimentation started, we will use a simplified environment for testing reinforcement learning algorithms.

## Mini-game
We will use the DefeatRoaches mini-game.

## State space
We will use a modified state space based on the following inputs from the original environment:
* Screen features
  * player_relative
  * selected
  * hit_points
  * unit_density
* Structured
  * army count

We will use player_relative to split hit_points and unit_density into two layers each, one for the marines and one for the roaches.  
We will use selected to give the agent the position of the current marine being ordered and its location.
Thus, our agent will receive the following features:
* Spatial features
  * Position of current marine
  * Hit points of all marines
  * Hit points of all roaches
  * Unit density of all marines
  * Unit density of all roaches
* Scalar features
  * Hit points of current marine
  * Army count

Note that our script may use additional input from the original environment to execute the actions detailed below.

## Action space
For each marine, our agent will choose one of 3 actions:
* Retreat (move away from the center of mass of enemies)
* Attack closest (attack the roach closest to this marine)
* Attack weakest (attack the roach with the lowest health)
The agent will order one marine at a time, cycling through the army of marines. There will be an adjustable number of frames skipped between each order.

## Environment Handler 
Every frame, the agent will pass its observations to the environment handler. The environment handler will behave differently depending on the number of frames skipped between actions. The environment handler will have two important variables:
  * frame_skip         # The number of frames between every action
  * frame_no           # This variable always has the following value: frame_no = total_frames % frame_skip

The environment handler will have a step function. This function will have three primary functionalities.
  * if frame_no is 0, the environment handler will select a unit and decide on an action. S, A, R, S', D is returned.
  * if frame_no is 1, the selected unit will be assigned an action by our policy and send the action to the environment.
  * if frame_no is any other value, step performs no actions on the environment.

At every iteration of step, frame_no will be incremented, and then set equal to frame_no % frame_skip. All interaction between the agent and the environment will pass through this environment handler. This step function handles the multi-frame action sequences needed in the pysc2 environment. 

## Main Class Structure:
1. Imports
2. Constants
	-input space: W x H x D
	-auxiliary_input_size
	-action space
	-player
	-game constants
3. Player Class
4. main():
	Agent = Agent()
	Agent.train()
	Agent.test()


Class Agent():
	action_handler
	replay_mem = deque(limit=MAX_REPLAY_SIZE)
	games[]

	def train():
		action_handler.training = True
		for game in num_games:
			Game = run_game()
			replay_mem.append(Game.mem)
			gradient_step()

	def run_game():
		action_handler.new_game()
		done = False
		while !done:
			s, a, r, s’, done = action_handler.get_env()
			memory.append( (s, a, r, s, done) )
			action_handler.exec(jobs)
