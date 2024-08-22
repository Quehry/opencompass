from datasets import load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HLDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = load_dataset('json', data_files={'test': path}, split='test')
        return dataset


class HLEvaluator(BaseEvaluator):

    def score(self):
        return {'score': 0}
