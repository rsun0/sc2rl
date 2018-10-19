from pysc2.lib import actions
from enum import Enum
import numpy as np
from scipy import ndimage

class Action(Enum):
    """
    NO_OP = 1
    SELECT = 2
    RETREAT = 3
    ATTACK = 4
    """
    NO_OP = 1
    SELECT = 2
    
    LEFT = 3
    UP_LEFT = 4
    UP = 5
    UP_RIGHT = 6
    RIGHT = 7
    DOWN_RIGHT = 8
    DOWN = 9
    DOWN_LEFT = 10
    ATTACK_NEAREST = 11
    ATTACK_WEAKEST = 12
    
class Actuator:
    def __init__(self):
        self.units_selected = False
        self.move_multiplier = 5
        self.location = None
        

    def reset(self):
        self.units_selected = False
        self.location = None
        
    def action_space(self):
        return 10

    def compute_action(self, action, custom_obs):
        '''
        Computes the raw action corresponding to the chosen abstract action

        :param action: The chosen abstract action (NO_OP, SELECT, RETREAT, or ATTACK)
        :param custom_obs: Custom observations as given by modified state space
        :returns: The raw action to return to environment
        '''
        if action == Action.NO_OP:
            return actions.FUNCTIONS.no_op()

        selected = custom_obs[0]
        friendly_unit_density = custom_obs[2]
        enemy_unit_hitpoints = custom_obs[3]
        enemy_unit_density = custom_obs[4]
        
        if np.all(selected == 0):
            self.units_selected = False

        if not self.units_selected:
            assert action == Action.SELECT, 'Actuator cannot order units without selection (unit may have died)'

            self.units_selected = True
            return actions.FUNCTIONS.select_army('select')
        
        else:
            location = [0, 0]
            try:
                location = np.array(np.where(selected==np.min(selected[np.nonzero(selected)])))[:,0]
            except:
                print("Location invalid")
                return actions.FUNCTIONS.no_op()
            """
            if action == Action.RETREAT:
                return self._compute_retreat(friendly_unit_density, enemy_unit_density)
            elif action == Action.ATTACK:
                return self._compute_attack(enemy_unit_density)
            assert False, 'Actuator cannot select with preexisting selection'
            """
            if action == Action.LEFT:
                return self._compute_move(location,
                                            -1,
                                            0)
            elif action == Action.UP_LEFT:
                return self._compute_move(location,
                                            -1,
                                            1)
            elif action == Action.UP:
                return self._compute_move(location,
                                            0,
                                            1)
            elif action == Action.UP_RIGHT:
                return self._compute_move(location,
                                            1,
                                            1)
            elif action == Action.RIGHT:
                return self._compute_move(location,
                                            1,
                                            0)
            elif action == Action.DOWN_RIGHT:
                return self._compute_move(location,
                                            1,
                                            -1)
            elif action == Action.DOWN:
                return self._compute_move(location,
                                            0,
                                            -1)
            elif action == Action.DOWN_LEFT:
                return self._compute_move(location,
                                            -1,
                                            -1)
            elif action == Action.ATTACK_NEAREST:
                return self._compute_attack_nearest(location,
                                            enemy_unit_density)
            elif action == Action.ATTACK_WEAKEST:
                return self._compute_attack_nearest(location,
                                            enemy_unit_hitpoints)
                                            
    @staticmethod
    def _compute_retreat(friendly_unit_density, enemy_unit_density):
        friendly_com = np.flip(np.array(ndimage.measurements.center_of_mass(friendly_unit_density)), 0)
        enemy_com = np.flip(np.array(ndimage.measurements.center_of_mass(enemy_unit_density)), 0)
        direction_vector = -(enemy_com - friendly_com)

        max_movement_x = Actuator._compute_movement_multiplier(direction_vector[0], friendly_com[0], friendly_unit_density.shape[1])
        max_movement_y = Actuator._compute_movement_multiplier(direction_vector[1], friendly_com[1], friendly_unit_density.shape[0])
        max_movement = min(max_movement_x, max_movement_y)
        
        retreat_target = np.round(max_movement * direction_vector + friendly_com)
        return actions.FUNCTIONS.Move_screen('now', retreat_target)

    @staticmethod
    def _compute_movement_multiplier(direction, position, screen_size):
        if direction < 0:
            return position / -direction
        elif direction == 0:
            return np.infty
        else:
            return (screen_size - 1 - position) / direction

    @staticmethod
    def _compute_attack(enemy_unit_density):
        enemy_com = np.flip(np.array(ndimage.measurements.center_of_mass(enemy_unit_density)), 0)
        return actions.FUNCTIONS.Attack_screen('now', enemy_com)
        
    @staticmethod
    def _compute_attack_nearest(location, enemy_unit_density):
        enemy_indices = np.array(enemy_unit_density.nonzero())
        target_index = np.argmin( la.norm( location - enemy_indices, 2, axis=0))
        target_loc = enemy_indices[target_index]
        return action.FUNCTIONS.Attack_screen('now', target_loc)
        
    @staticmethod
    def _compute_attack_weakest(location, enemy_unit_hitpoints):
        target_loc = np.array(np.where(enemy_unit_hitpoints==np.min(enemy_unit_hitpoints[np.nonzero(enemy_unit_hitpoints)])))[:,0]
        return actions.FUNCTIONS.Attack_screen('now', target_loc)
        
        
    @staticmethod
    def _compute_move(location, dx, dy):
        
        direction_vector = self.move_multiplier * np.array([dy, dx])
        
        move_target = _screen_normalize(location + direction_vector)
        return actions.FUNCTIONS.Move_screen('now', move_target)
        
    @staticmethod
    def _screen_normalize(coords):
        for i in range(len(coords)):
            if coords[i] < 0:
                coords[i] = 0
            if coords[i] > screen_size - 1:
                coords[i] = screen_size - 1;
        
