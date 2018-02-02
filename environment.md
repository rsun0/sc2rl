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
