import json

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class EmentaProblem(text_problems.Text2TextProblem):
    '''
    Produce an Ementa (summary) based on article
    '''

    @property
    def aprox_vocab_size(self):
        return 2**13  # ~8k

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us
        return False

    @property
    def dataset_splits(self):
        '''Splits of data to produce and number of output shards for each'''
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        del data_dir
        del tmp_dir
        del dataset_split

        src_file = 'sample-500.json'
        lines = []
        with open(src_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            obj = json.loads(line, encoding='utf8')
            limite = 500
            if len(obj['ementa']) < 500:
                limite = len(obj['ementa'])
            summary = obj['ementa'].lower()[0:limite]

            limite = 500
            if len(obj['relatorio']) < 500:
                limite = len(obj['relatorio'])
            story = obj['relatorio'].lower()[0:limite]
            if summary and story:
                yield {
                    "inputs": story,
                    "targets": summary,
                }
