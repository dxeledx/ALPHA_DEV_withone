from models import TCFormer


def get_model_cls(model_name: str):
    if model_name != "TCFormer":
        raise KeyError(f"仅支持 TCFormer，收到: {model_name}")
    return TCFormer
