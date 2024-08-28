from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='phi-3_5-moe-instruct-hf',
        path='microsoft/Phi-3.5-MoE-instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
