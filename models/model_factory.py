from models.gemma import GemmaModel


def load_model(config):

    model_name = config["model"]["name"]

    if "gemma" in model_name.lower():
        return GemmaModel(model_name)

    raise ValueError(f"Unsupported model : {model_name}")
