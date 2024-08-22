from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HLDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, HLEvaluator

hl_bench_reader_cfg = dict(input_columns=['chat_prompt'], output_column=None)

hl_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{chat_prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32768))

hl_bench_eval_cfg = dict(evaluator=dict(type=HLEvaluator))

hl_bench_datasets = []

category_list = ['open_ended_qa', 'summarization', 'chat', 'text_completion', 'heuristic_text_generation']

for category in category_list:
    hl_bench_datasets.append(
        dict(
            abbr='hl_bench',
            type=HLDataset,
            path=f'./data/HL-Bench/{category}.jsonl',
            reader_cfg=hl_bench_reader_cfg,
            infer_cfg=hl_bench_infer_cfg,
            eval_cfg=hl_bench_eval_cfg
            )
    )
