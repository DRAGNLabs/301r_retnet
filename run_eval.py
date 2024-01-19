import lm_eval
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from hugging_face_model import RetNetModel, TransformerModel
from evaluation_models import EvalRetNetModel
from torchscale.architecture.config import RetNetConfig, DecoderConfig

AutoConfig.register('retnet', RetNetConfig)
AutoModel.register(RetNetConfig, RetNetModel)
AutoModelForCausalLM.register(RetNetConfig, RetNetModel)

model = AutoModelForCausalLM.from_pretrained('retnet')
eval_model = EvalRetNetModel(model)

lm_eval.tasks.initialize_tasks()

results = lm_eval.simple_evaluate(
    model=eval_model,
    tasks=['hellaswag'],
    num_fewshot=0,
    device='cpu',
)

breakpoint()