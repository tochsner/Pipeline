from typing import Generator

from pipeline.pipeline_step import FirstPipelineStep
from pipeline.transformer import FunctionTransformer


class IntegerStream(FirstPipelineStep):

    def __init__(self, nr_outgoing_streams=1, **arguments):
        super().__init__(**arguments)

        self.nr_outgoing_streams = nr_outgoing_streams
        self.next_number = 0

    def get_next(self, previous: Generator, **arguments) -> Generator:
        self.next_number += 1
        yield [self.next_number] * self.nr_outgoing_streams


class Adder(FunctionTransformer):

    def transform(self, number, increment=0, **arguments):
        return number + increment
