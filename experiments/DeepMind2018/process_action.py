from pysc2.lib.actions import FUNCTIONS, TYPES


"""
    Takes in agent_action, array of length three.
    agent_action = [base_action, args, spatial_args]

    base_action: int, refers to index in FUNCTIONS
    args: int array of length len(TYPES).
    spatial_args: array of int arrays of length 2
"""

def action_to_pysc2(agent_action):

    [base_action, args, spatial_args] = agent_action

    base_action_func = FUNCTIONS._func_list[base_action]
    arg_types = base_action_func.args
    arg_ids = [arg_types[i].id for i in range(len(arg_types))]

    arg_inputs = []
    spatial_arg_names = []
    for i in range(len(arg_ids)):
        id = arg_ids[i]
        if is_spatial_arg(id):
            spatial_arg_inputs.append(spatial_args[id])
        elif TYPES[id].value is not None:
            arg_inputs.append(TYPES[id].values(args[id]))
        else:
            arg_inputs.append(args[id])

    return base_action_func(*arg_inputs, *spatial_arg_inputs)

def is_spatial_arg(id):
    return id < 3
