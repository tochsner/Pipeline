import itertools
from collections import Iterable
from typing import Generator, Callable

import numpy as np

from pipeline.pipeline_step import PipelineStep, FinalPipelineStep


class FunctionTransformer(PipelineStep):

    def __init__(self, function: Callable = None, **arguments):
        super().__init__(**arguments)
        self.function = function

    def get_next(self, previous: Generator, **arguments) -> Generator:
        inputs = next(previous)
        yield [self.transform(i, **arguments) for i in inputs]

    def transform(self, input, **arguments):
        return self.function(input, **arguments)


class StreamsToList(PipelineStep):

    def get_next(self, previous: Generator, **arguments) -> Generator:
        yield [next(previous)]


class ListToStreams(PipelineStep):

    def get_next(self, previous: Generator, **arguments) -> Generator:
        yield list(itertools.chain.from_iterable(next(previous)))


class StreamsToTuple(FinalPipelineStep):

    def get_next(self, previous: Generator, **arguments) -> Generator:
        yield tuple(next(previous))


class StackIncomingStreams(PipelineStep):

    def __init__(self, **arguments):
        super().__init__(**arguments)

    def get_next(self, previous: Generator, **arguments) -> Generator:
        yield [np.stack(next(previous), axis=-1)]


class DictToValue(FunctionTransformer):

    def transform(self, dict, key='', **arguments):
        return dict[key]


class ToNumpyArray(FunctionTransformer):

    def transform(self, input, **arguments):
        if isinstance(input, Iterable): return np.array(input)

        return np.array([input])
