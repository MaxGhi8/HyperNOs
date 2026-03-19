"""
In this file there are some utilities functions that are used in the main file.
"""


def count_params_fno(config, accurate=True):
    """
    function to approximate the number of parameters for the FNO model and classical architecture
    """
    latent = 128
    P_Q = (
        config["in_dim"] + 2 * config["width"] + config["out_dim"] + 2
    ) * latent + config["width"] * config["out_dim"]

    hidden = (
        config["n_layers"]
        * (config["width"] ** 2)
        * config["modes"] ** config["problem_dim"]
        * 2 ** config["problem_dim"]
    )

    if accurate:
        return (
            hidden + P_Q + config["n_layers"] * (config["width"] ** 2 + config["width"])
        )
    else:
        return hidden


def compute_modes(total_param, maximum, config):
    modes = min(
        max(
            int(
                (
                    total_param
                    / (
                        2 ** config["problem_dim"]
                        * config["n_layers"]
                        * config["width"] ** 2
                    )
                )
                ** (1 / config["problem_dim"])
            ),
            1,
        ),
        maximum,
    )

    return modes
