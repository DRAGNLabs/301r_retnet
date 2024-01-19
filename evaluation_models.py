from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from hugging_face_model import RetNetModel
from transformers import PreTrainedModel

class EvalRetNetModel(LM):
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_env
    
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError
    
    def generate_until(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError