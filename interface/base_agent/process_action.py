from pysc2.lib.actions import FUNCTIONS, TYPES, FunctionCall


"""
    Takes in agent_action, array of length three.
    agent_action = [base_action, args, spatial_args]

    base_action: int, refers to index in FUNCTIONS
    args: int array of length len(TYPES).
    spatial_args: array of int arrays of length 2
"""

def action_to_pysc2(agent_action):

    [[base_action, args, spatial_args]] = agent_action

    base_action_func = FUNCTIONS._func_list[base_action]
    arg_types = base_action_func.args
    arg_ids = [arg_types[i].id for i in range(len(arg_types))]

    arg_inputs = []
    spatial_arg_inputs = []
    for i in range(len(arg_ids)):
        id = arg_ids[i]
        if is_spatial_arg(id):
            spatial_arg_inputs.append(2 * spatial_args[id])
        elif TYPES[id].values is not None:
            arg_inputs.append([TYPES[id].values(args[id-3])])
        else:
            arg_inputs.append([args[id-3]])

    function = FunctionCall(base_action_func.id, arg_inputs + spatial_arg_inputs)
    return function

def is_spatial_arg(id):
    return id < 3
