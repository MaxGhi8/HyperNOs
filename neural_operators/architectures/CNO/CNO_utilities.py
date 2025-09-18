"""
In this file there are some utilities functions that are used in the main file.
"""


def count_params_cno(config, accurate=True):
    """
    Function to approximate the number of parameters in the CNO model.
    """
    latent = 64
    P_Q = (
        config["kernel_size"] ** config["problem_dim"]
        * latent
        * (
            config["in_dim"]
            + config["out_dim"]
            + (3 / 2) * config["channel_multiplier"]
        )
    )
    pow4 = 4 ** (config["N_layers"] - 1)
    sq = (pow4 - 1) / (4 - 1)
    hidden = (
        config["kernel_size"] ** config["problem_dim"]
        * config["channel_multiplier"] ** 2
        * (
            pow4 * 2 * config["N_res_neck"]
            + 2 * config["N_res"] * (1 / 4 + sq)
            + (31 / 6) * pow4
            - 11 / 12
        )
    )
    if accurate:
        return hidden + P_Q
    else:
        return hidden


def compute_channel_multiplier(total_param, config):

    pow4 = 4 ** (config["N_layers"] - 1)
    sq = (pow4 - 1) / (4 - 1)
    channel_multiplier = (
        total_param
        / (
            config["kernel_size"] ** config["problem_dim"]
            * (
                pow4 * 2 * config["N_res_neck"]
                + 2 * config["N_res"] * (1 / 4 + sq)
                + (31 / 6) * pow4
                - 11 / 12
            )
        )
    ) ** (1 / 2)

    return int(channel_multiplier)
