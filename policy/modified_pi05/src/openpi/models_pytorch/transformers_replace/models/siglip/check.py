import inspect

import transformers
from transformers.models.paligemma.modeling_paligemma import PaliGemmaModel


def check_whether_transformers_replace_is_installed_correctly():
    if transformers.__version__ != "4.53.2":
        return False
    params = inspect.signature(PaliGemmaModel.get_image_features).parameters
    return "num_frames" in params