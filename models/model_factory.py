from models.gemma import GemmaModel

from models.aya import AyaModel


def load_model(config):

    model_name = config["model"]["name"]

    if "gemma" in model_name.lower():

        return GemmaModel(model_name)

    elif "aya" in model_name.lower():

        return AyaModel(model_name)

    else:

        raise ValueError(f"Unsupported model {model_name}")
