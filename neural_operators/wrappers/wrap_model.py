
from .AirfoilWrapper import AirfoilWrapper
from .BAMPNO_Continuation_Wrapper import BAMPNO_Continuation_Wrapper
from .CrossTrussWrapper import CrossTrussWrapper
from .DeepONetWrapper import DeepONetWrapper
from .StiffnessMatrixWrapper import StiffnessMatrixWrapper
from .PermuteWrapper import PermuteWrapper
from .DeepXDEDeepONetWrapper import DeepXDEDeepONetWrapper
from .DeepXDEMIONetWrapper import DeepXDEMIONetWrapper
from .RNOWrapper import RNOWrapper
from .OTNOWrapper import OTNOWrapper


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
        case str(x) if x.endswith("_don"):
            if grid_size is None:
                raise ValueError(
                    f"grid_size must be provided for {which_example} wrapper"
                )
            return DeepONetWrapper(model, grid_size)
        case str(x) if x.endswith("_neural_operator"):
            return PermuteWrapper(model)
        case str(x) if x.endswith("_mionet_deepxde"):
            return DeepXDEMIONetWrapper(model)
        case str(x) if x.endswith("_deepxde"):
            return DeepXDEDeepONetWrapper(model)
        case str(x) if x.startswith("OTNO"):
            return OTNOWrapper(model)
        case str(x) if x.startswith("RNO"):
            return RNOWrapper(model)
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
