from .AirfoilWrapper import AirfoilWrapper
from .CrossTrussWrapper import CrossTrussWrapper
from .StiffnessMatrixWrapper import StiffnessMatrixWrapper


def wrap_model(model, which_example):

    # Wrap the model based on which_example
    match which_example:
        case "airfoil":
            return AirfoilWrapper(model)
        case "crosstruss":
            return CrossTrussWrapper(model)
        case "stiffness_matrix":
            return StiffnessMatrixWrapper(model)
        case _:
            return model

    return model


def wrap_model_builder(original_builder, which_example):

    def wrapped_builder(config):
        model = original_builder(config)
        return wrap_model(model, which_example)

    return wrapped_builder
