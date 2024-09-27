from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HelloBenchDataset, HelloBenchEvaluator

hellobench_reader_cfg = dict(input_columns=['chat_prompt'], output_column=None)

hellobench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{chat_prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16384))

hellobench_eval_cfg = dict(evaluator=dict(type=HelloBenchEvaluator))

hellobench_datasets = []

category_list = ['open_ended_qa', 'summarization', 'chat', 'text_completion', 'heuristic_text_generation']

for category in category_list:
    hellobench_datasets.append(
        dict(
            abbr=f'hellobench_{category}',
            type=HelloBenchDataset,
            path=f'./data/HelloBench/{category}.jsonl',
            reader_cfg=hellobench_reader_cfg,
            infer_cfg=hellobench_infer_cfg,
            eval_cfg=hellobench_eval_cfg
            )
    )
