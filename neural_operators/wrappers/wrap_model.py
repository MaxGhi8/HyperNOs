from wrappers.AirfoilWrapper import AirfoilWrapper
from wrappers.CrossTrussWrapper import CrossTrussWrapper


def wrap_model_builder(original_builder, which_example):
    def wrapped_builder(config):
        # Get the base model using the original builder
        model = original_builder(config)

        # Wrap the model based on which_example
        match which_example:
            case "airfoil":
                return AirfoilWrapper(model)
            case "crosstruss":
                return CrossTrussWrapper(model)
            case _:
                return model

    return wrapped_builder
