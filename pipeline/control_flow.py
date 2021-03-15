from copy import deepcopy
from typing import Generator, List, Union

from pipeline.pipeline_step import PipelineStep
from pipeline.pipeline_step_view import PipelineStepView


class Identity(PipelineStep):
    """A pipeline step which simply outputs the incoming streams."""

    def get_next(self, previous: Generator, increment_by=1, **arguments) -> Generator:
        yield from previous


class Block(PipelineStep):
    """A pipeline step which allows to group multiple consecutive pipeline steps."""

    def get_next(self, previous: Generator, **arguments) -> Generator:
        raise NotImplemented()

    def __init__(self, steps: List[PipelineStep], **arguments):
        super().__init__(**arguments)
        self.steps = steps

    def connect_to(self, previous: Union[PipelineStepView, List[PipelineStepView]] = None,
                   previous_indices: Union[int, List[int], List[List[int]]] = None) \
            -> PipelineStepView:
        current_view = self.steps[0].connect_to(previous, previous_indices)
        for step in self.steps[1:]:
            current_view = step.connect_to(current_view)
        return current_view


class DuplicateStream(PipelineStep):
    """A pipeline step which duplicates every incoming stream into 'nr_duplications' outgoing streams."""

    def __init__(self, nr_duplications=2, **arguments):
        super().__init__(**arguments)
        self.nr_duplications = nr_duplications

    def get_next(self, previous: Generator, **arguments) -> Generator:
        inputs = next(previous)
        yield [deepcopy(i) for _ in range(self.nr_duplications) for i in inputs]


class Duplicator(PipelineStep):
    """A pipeline step which duplicates every incoming element 'nr_duplications' times."""

    def __init__(self, nr_duplications=2, **arguments):
        super().__init__(**arguments)
        self.nr_duplications = nr_duplications

    def get_next(self, previous: Generator, **arguments) -> Generator:
        inputs = next(previous)
        for i in range(self.nr_duplications):
            yield deepcopy(inputs)


class RoundRobinMerger(PipelineStep):
    """
    A pipeline step which merges the incoming elements of 'merged_steps' steps into one outgoing step with
    ('merged_steps' x nr. incoming streams) outgoing streams.
    """

    def __init__(self, merged_steps=2, **arguments):
        super().__init__(**arguments)
        self.merged_steps = merged_steps

    def get_next(self, previous: Generator, **arguments) -> Generator:
        output = []
        while len(output) < self.merged_steps:
            output += next(previous)
        yield output
