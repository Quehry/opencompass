from mmengine.config import read_base

with read_base():
    from .datasets.hellobench.hellobench_gen import hellobench_datasets

    # Model configs
    from .models.yi.hf_yi_1_5_34b_chat import models as yi_1_5_34b_chat

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

datasets = [*hellobench_datasets]
workdir = 'outputs/hellobench'

models = [*yi_1_5_34b_chat]

model_cfg = dict(batch_size=1,
                 max_out_len=16384,
                 run_cfg=dict(num_gpus=4, num_procs=1),
                 generation_kwargs=dict(temperature=0.8, do_sample=True))

for mdl in models:
    mdl.update(model_cfg)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
        max_num_workers=256,
    ),
)

eval = dict(partitioner=dict(type=NaivePartitioner),
            runner=dict(
                type=LocalRunner,
                task=dict(type=OpenICLEvalTask),
                max_num_workers=256,
            ))
