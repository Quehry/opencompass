from mmengine.config import read_base

with read_base():
    # LLM compression datasets
    from .datasets.hl_bench.hl_bench_gen import hl_bench_datasets

    # Model configs
    from .models.qwen.hf_qwen1_5_7b import models as qwen1_5_7b
    from .models.qwen.hf_qwen1_5_7b_chat import models as qwen1_5_7b_chat
    from .models.qwen.hf_qwen2_72b_instruct import models as qwen2_72b_instruct
    from .models.qwen.hf_qwen1_5_14b import models as qwen1_5_14b
    from .models.hf_llama.hf_llama2_7b import models as llama2_7b
    from .models.hf_llama.hf_llama2_13b import models as llama2_13b
    from .models.hf_internlm.hf_internlm2_5_20b_chat import models as internlm2_5_20b_chat
    from .models.gemma.hf_gemma2_27b_it import models as gemma2_27b_it
    from .models.chatglm.hf_glm4_9b_chat import models as glm4_9b_chat
    from .models.phi.hf_phi_3_5_moe_instruct import models as phi_3_5_moe_instructs
    from .models.yi.hf_yi_1_5_34b_chat_16k import models as yi_1_5_34b_chat_16k
    from .models.mistral.hf_mistral_7b_instruct_v0_2 import models as mistral_7b_instruct_v0_2


from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.summarizers import LLMCompressionSummarizer


# -------------Inference Stage ----------------------------------------
datasets = [*hl_bench_datasets]
workdir = 'outputs/hl_bench'

models = [
    # *qwen1_5_7b,
    # *qwen2_72b_instruct,
    # *internlm2_5_20b_chat,
    # *phi_3_5_moe_instructs,
    # *yi_1_5_34b_chat_16k,
    # *gemma2_27b_it,
    *mistral_7b_instruct_v0_2,
    # *glm4_9b_chat
    # *qwen1_5_14b,
    # *llama2_7b,
    # *llama2_13b,
]

# Set custom batch_size and num_gpus for faster loss calculation
# Smaller batch_size should give more precise results, at the cost of worse performance
model_cfg = dict(
    batch_size=1,
    max_out_len=32768,
    run_cfg=dict(num_gpus=4, num_procs=1),
    generation_kwargs=dict(temperature=0.8, do_sample=True)
)

for mdl in models:
    mdl.update(model_cfg)


infer = dict(
    # The OpenCompass implementation of BPC currently only supports NaivePartitioner, as the sliding window approach requires the dataset to be loaded sequentially. Using other partitioner types may produce incorrect results.
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
        max_num_workers=256,  # Maximum concurrent evaluation task count
    ),
)


# -------------Evaluation Stage ----------------------------------------
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask),
        max_num_workers=256,
    )
)


# -------------Summarization Stage ----------------------------------------
# summarizer = dict(type=LLMCompressionSummarizer)
