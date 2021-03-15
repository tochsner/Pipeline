from typing import Generator

import numpy as np

from pipeline.pipeline_step import PipelineStep, FinalPipelineStep
from pipeline.transformer import FunctionTransformer


class BatchGenerator(PipelineStep):

    def get_next(self, previous: Generator, batch_size=32, **arguments) -> Generator:
        batches = [[n] for n in next(previous)]

        for elem in range(batch_size - 1):
            next_inputs = next(previous)

            for i, next_input in enumerate(next_inputs):
                batches[i].append(next_input)

        yield [np.array(batch) for batch in batches]


class OneHotEncoder(FunctionTransformer):

    def transform(self, class_nr: int = 0, num_classes: int = 1, **arguments):
        one_hot = np.zeros(num_classes)
        one_hot[class_nr] = 1
        return one_hot


class KerasTrainingGenerator(FinalPipelineStep):

    def get_next(self, previous: Generator, batch_size=32, input_indices=None, output_indices=None, **arguments)\
            -> Generator:
        if input_indices is None: input_indices = [0]
        if output_indices is None: output_indices = [1]

        input_data, output_data = [[] for _ in input_indices], [[] for _ in output_indices]

        for _ in range(batch_size):
            all_data = next(previous)

            assert len(all_data) == (len(input_indices) + len(output_indices)),\
                'Number of provided input and output indices does not match with the number of incoming streams.'

            for i, index in enumerate(input_indices):
                input_data[i].append(all_data[index])

            for i, index in enumerate(output_indices):
                output_data[i].append(all_data[index])

        input_data = [np.array(d) for d in input_data]
        output_data = [np.array(d) for d in output_data]

        yield input_data, output_data


class KerasTestGenerator(FinalPipelineStep):

    def get_next(self, previous: Generator, input_indices=None, output_indices=None, **arguments)\
            -> Generator:
        if input_indices is None: input_indices = [0]
        if output_indices is None: output_indices = [1]

        input_data, output_data = [], []

        all_data = next(previous)

        assert len(all_data) == (len(input_indices) + len(output_indices)),\
            'Number of provided input and output indices does not match with the number of incoming streams.'

        for i, index in enumerate(input_indices):
            input_data.append(all_data[index])

        for i, index in enumerate(output_indices):
            output_data.append(all_data[index])

        input_data = [d for d in input_data]
        output_data = [d for d in output_data]

        yield input_data, output_data
