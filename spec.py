GPU_ID = None

CHAT_TEMPLATE = {
    '01-ai/Yi-6B': False,
    '01-ai/Yi-6B-Chat': True,
    'aisingapore/SEA-LION-v1-7B': False,  # triton_pre_mlir problem
    'aisingapore/SEA-LION-v1-7B-IT': True,  # triton_pre_mlir problem
    'allenai/OLMo-7B-0424': False,  # missing OLMoTokenizer problem
    'allenai/OLMo-7B-hf': False,  # missing OLMoTokenizer problem
    'allenai/OLMo-7B-Instruct': True,  # missing OLmoTokenizer problem
    'bigscience/T0pp': False,  # 11B
    'BioMistral/BioMistral-7B': False,
    'databricks/dolly-v2-12b': False,
    'databricks/dolly-v2-7b': False,
    'databricks/dolly-v2-3b': False,
    'deepseek-ai/deepseek-coder-6.7b-instruct': True,
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': True,
    'EleutherAI/gpt-j-6b': False,
    'EleutherAI/pythia-12b': False,
    'EleutherAI/pythia-14m': False,
    'EleutherAI/pythia-160m': False,
    'EleutherAI/pythia-1b': False,
    'EleutherAI/pythia-2.8b': False,
    'EleutherAI/pythia-31m': False,
    'EleutherAI/pythia-410m': False,
    'EleutherAI/pythia-6.9b': False,
    'EleutherAI/pythia-70m': False,
    'epfl-llm/meditron-7b': False,
    'google/gemma-2-2b': False,
    'google/gemma-2-9b': False,
    'google/gemma-2-9b-it': True,
    'google/gemma-2b': False,
    'google/gemma-2b-it': True,
    'google/gemma-7b': False,
    'google/gemma-7b-it': True,
    'ibm-granite/granite-3.1-2b-base': False,
    'ibm-granite/granite-3.1-2b-instruct': True,
    'ibm-granite/granite-3.1-8b-base': False,
    'ibm-granite/granite-3.1-8b-instruct': True,
    'lmsys/vicuna-13b-v1.3': False,
    'lmsys/vicuna-7b-v1.3': False,
    'maritaca-ai/sabia-7b': False,
    'meta-llama/Llama-2-13b-hf': False,
    'meta-llama/Llama-2-7b-hf': False,
    'meta-llama/Llama-3.2-1B': False,
    'meta-llama/Llama-3.2-1B-Instruct': True,
    'meta-llama/Llama-3.2-3B': False,
    'meta-llama/Llama-3.2-3B-Instruct': True,
    'meta-llama/Meta-Llama-3-8B': False,
    'microsoft/phi-2': False,  # 3B
    'microsoft/Phi-3-medium-4k-instruct': True,  # 14B
    'microsoft/Phi-3-small-8k-instruct': True,  # 7B
    'microsoft/Phi-3.5-mini-instruct': True,  # 4B  # Errored out due to some DynamicCache issue? No attribute "get_max_length", did you mean "get_seq_length"?
    'mistralai/Mistral-7B-Instruct-v0.3': True,
    'mistralai/Mistral-7B-v0.1': False,
    'mistralai/Mistral-Nemo-Base-2407': False,  # 12B
    'mosaicml/mpt-7b': False,  # triton_pre_mlir problem
    'Qwen/Qwen-7B': False,
    'Qwen/Qwen1.5-14B': False,
    'Qwen/Qwen1.5-14B-Chat': True,
    'Qwen/Qwen1.5-7B': False,
    'Qwen/Qwen1.5-7B-Chat': True,
    'Qwen/Qwen2.5-7B-Instruct': True,
    'sail/Sailor-7B': False,
    'sail/Sailor-7B-Chat': True,
    'scb10x/llama-3-typhoon-v1.5-8b': False,
    'scb10x/llama-3-typhoon-v1.5-8b-instruct': True,
    'scb10x/typhoon-7b': False,
    'stabilityai/stablelm-base-alpha-3b': False,
    'stabilityai/stablelm-base-alpha-7b': False,
    'tiiuae/falcon-7b': False,
    'tiiuae/falcon-7b-instruct': True,  # some problem with assert len(continuation_enc) > 0
}


MODELS = {
    'A100_1': [
        'EleutherAI/pythia-410m',
        'EleutherAI/pythia-31m',
    ], # 1B
    ###### H100 threshold ######
    'H100_1': [
        '01-ai/Yi-6B',
        'scb10x/llama-3-typhoon-v1.5-8b',
    ], # 14B
    'H100_2': [
        '01-ai/Yi-6B-Chat',
        'scb10x/llama-3-typhoon-v1.5-8b-instruct',
    ], # 14B
    'H100_3': [
        # 'aisingapore/SEA-LION-v1-7B',
        'scb10x/typhoon-7b',
    ], # 14B
    'H100_4': [
        # 'aisingapore/SEA-LION-v1-7B-IT',
        'stabilityai/stablelm-base-alpha-7b'
    ], # 14B
    'H100_5': [
        # 'allenai/OLMo-7B-0424',
        'tiiuae/falcon-7b',
    ], # 14B
    'H100_6': [
        # 'allenai/OLMo-7B-hf',
        'tiiuae/falcon-7b-instruct'
    ], # 14B
    'H100_7': [
        # 'allenai/OLMo-7B-Instruct',
        'google/gemma-2b',
        'EleutherAI/pythia-70m',
    ], # 7B
    'H100_8': [
        'BioMistral/BioMistral-7B',
    ], # 9B
    'H100_9': [
        'databricks/dolly-v2-7b',
        'google/gemma-2b-it',
    ], # 9B
    'H100_10': [
        'deepseek-ai/deepseek-coder-6.7b-instruct',
        'ibm-granite/granite-3.1-2b-instruct',
    ], # 9B
    'H100_11': [
        'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'meta-llama/Llama-3.2-1B',
    ], # 9B
    'H100_12': [
        'EleutherAI/gpt-j-6b',
        'EleutherAI/pythia-2.8b',
    ], # 9B
    'H100_13': [
        'EleutherAI/pythia-6.9b',
        'google/gemma-2-2b',
    ], # 9B
    'H100_14': [
        'epfl-llm/meditron-7b',
        'ibm-granite/granite-3.1-2b-base',
    ], # 9B
    'H100_15': [
        'google/gemma-2-9b',
        'EleutherAI/pythia-1b',
    ], # 10B
    'H100_16': [
        'google/gemma-2-9b-it',
        'EleutherAI/pythia-14m',
    ], # 9B
    'H100_17': [
        'google/gemma-7b',
        'meta-llama/Llama-3.2-1B-Instruct'
    ], # 8B
    'H100_18': [
        'google/gemma-7b-it',  # Failed to apply chat template. removing the system role in chat history.
        'meta-llama/Llama-3.2-3B'
    ], # 10B
    'H100_19': [
        'ibm-granite/granite-3.1-8b-base',
        'databricks/dolly-v2-3b'
    ], # 10B
    'H100_20': [
        'ibm-granite/granite-3.1-8b-instruct',
        'EleutherAI/pythia-160m',
    ], # 8B
    ###### H200 threshold ######
    'H200_1': [
        'bigscience/T0pp',
        'meta-llama/Llama-3.2-3B-Instruct',
        'Qwen/Qwen1.5-7B-Chat',
    ], # 21B
    'H200_2': [
        'databricks/dolly-v2-12b',
        'microsoft/phi-2',
        'Qwen/Qwen2.5-7B-Instruct',
    ], # 22B
    'H200_3': [
        'EleutherAI/pythia-12b',
        'microsoft/Phi-3.5-mini-instruct'
    ], # 16B
    'H200_4': [
        'lmsys/vicuna-13b-v1.3',
        'mistralai/Mistral-7B-v0.1',
    ], # 20B
    'H200_5': [
        'lmsys/vicuna-7b-v1.3',
        'maritaca-ai/sabia-7b',
        'sail/Sailor-7B'
    ], # 21B
    'H200_6': [
        'meta-llama/Llama-2-13b-hf',
        'mosaicml/mpt-7b'
    ], # 20B
    'H200_7': [
        'meta-llama/Llama-2-7b-hf',
        'meta-llama/Meta-Llama-3-8B'
    ], # 15B
    'H200_8': [
        'microsoft/Phi-3-medium-4k-instruct',
        'sail/Sailor-7B-Chat'
    ], # 21B
    'H200_9': [
        'microsoft/Phi-3-small-8k-instruct',
        'mistralai/Mistral-7B-Instruct-v0.3'
    ], # 14B
    'H200_10': [
        'mistralai/Mistral-Nemo-Base-2407',
        'stabilityai/stablelm-base-alpha-3b'
    ], # 15B
    'H200_11': [
        'Qwen/Qwen1.5-14B',
        'Qwen/Qwen-7B',
    ], # 21B
    'H200_12': [
        'Qwen/Qwen1.5-14B-Chat',
        'Qwen/Qwen1.5-7B'
    ], # 21B
}


model_check = []
for gpu_id in MODELS:
    for model in MODELS[gpu_id]:
        model_check.append(model)

# assert len(model_check) == len(CHAT_TEMPLATE)
# assert len(set(model_check)) == len(model_check)
# for model in model_check:
#     assert model in CHAT_TEMPLATE
# for model in CHAT_TEMPLATE:
#     assert model in model_check


EXCLUDED_TASKS = [
    # 'humaneval',
    # 'bigbench_generate_until',
    # 'bigbench_multiple_choice_a',
    # 'bigbench_multiple_choice_b',
]

TASKS_1 = [
    'mmlu',
    'gsm8k',
    'lambada_openai',
    'hellaswag',
    'anli',
    'mathqa',
    'truthfulqa',
    'arc_easy',
    'winogrande',
    'squadv2',
    'glue',
    'openbookqa',
    'piqa',
    'gpqa',
    'drop',
    'logiqa2',
    'toxigen',
    'social_iqa',
]

TASKS_2 = [
    'bigbench_multiple_choice_a',
    'bigbench_multiple_choice_b',
]




################################
### test setup with limit=10 ###
################################
# MODELS = [
#     "EleutherAI/pythia-14m",
#     # "mistralai/Mistral-7B-v0.1",
# ]
# TASKS = [
#     'gpqa'
# ]