from .AirfoilWrapper import AirfoilWrapper
from .BAMPNO_Continuation_Wrapper import BAMPNO_Continuation_Wrapper
from .CrossTrussWrapper import CrossTrussWrapper
from .StiffnessMatrixWrapper import StiffnessMatrixWrapper
from .PermuteWrapper import PermuteWrapper
from .DeepXDEDeepONetWrapper import DeepXDEDeepONetWrapper


def wrap_model(model, which_example):

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
        case str(x) if x.endswith("_neural_operator"):
            return PermuteWrapper(model)
        case str(x) if x.endswith("_deepxde") or x.endswith("_pod_deepxde"):
            return DeepXDEDeepONetWrapper(model)
        case _:
            print(f"No wrapper defined for {which_example}, returning the original model")

    return model


def wrap_model_builder(original_builder, which_example):

    def wrapped_builder(config):
        model = original_builder(config)
        return wrap_model(model, which_example)

    return wrapped_builder
