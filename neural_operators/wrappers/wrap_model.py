from .AirfoilWrapper import AirfoilWrapper
from .BAMPNO_Continuation_Wrapper import BAMPNO_Continuation_Wrapper
from .CrossTrussWrapper import CrossTrussWrapper
from .DeepONetWrapper import DeepONetWrapper
from .StiffnessMatrixWrapper import StiffnessMatrixWrapper


def wrap_model(model, which_example, grid_size=None):

    # Wrap the model based on which_example
    match which_example:
        case "airfoil":
            return AirfoilWrapper(model)
        case "crosstruss":
            return CrossTrussWrapper(model)
        case "stiffness_matrix":
            return StiffnessMatrixWrapper(model)
        case "bampno_continuation":
            return BAMPNO_Continuation_Wrapper(model)
        case "darcy_don":
            if grid_size is None:
                raise ValueError("grid_size must be provided for darcy_don wrapper")
            return DeepONetWrapper(model, grid_size)
        case _:
            print(
                f"No wrapper defined for {which_example}, returning the original model"
            )

    return model


def wrap_model_builder(original_builder, which_example, grid_size=None):

    def wrapped_builder(config):
        model = original_builder(config)

        # Get grid_size from config if available, otherwise use the parameter
        _grid_size = (
            config.get("size_grid", grid_size) if hasattr(config, "get") else grid_size
        )

        return wrap_model(model, which_example, _grid_size)

    return wrapped_builder
