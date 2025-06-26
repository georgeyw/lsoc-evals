import os
import lm_eval

from huggingface_hub import login
from lm_eval import evaluator

from s3 import push_pickle_to_s3
from spec import MODELS, TASKS


DEVICE = 'cuda'

os.environ['HF_TOKEN'] = os.getenv('HF_API_KEY')
login(token=os.getenv('HF_API_KEY'))


# Run evaluations
for model in MODELS:
    model = lm_eval.models.huggingface.HFLM(pretrained=model)
    results = evaluator.simple_evaluate(
        model=model,
        tasks=TASKS,
        batch_size="auto",
        log_samples=True,
        write_out=True,
        device=DEVICE,
        limit=100,
    )
    object_name = f'{model}_evals_test.pkl'
    push_pickle_to_s3(data=results, object_name=object_name)


