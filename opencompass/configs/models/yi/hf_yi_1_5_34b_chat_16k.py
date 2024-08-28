from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='yi-1.5-34b-chat-16k-hf',
        path='01-ai/Yi-1.5-34B-Chat-16K',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
