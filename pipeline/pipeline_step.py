from abc import ABC, abstractmethod
from typing import List, Generator, Union

from pipeline.exceptions import IteratedThroughAll
from pipeline.pipeline_step_view import PipelineStepView


class PipelineStep(ABC):
    """
    Corresponds to a node in a Pipeline-Graph.

    This is the abstract base class. All concrete implementations have to provide the 'get_next' method.

    A PipelineStep is never used directly to access to the data stream, but a PipelineStepView is used as a wrapper.
    The PipelineStepView manages how different PipelineSteps are connected (one PipelineStep instance can be wrapped
    by multiple PipelineStepView's and can thus be part of multiple graphs). This also allows the use different views
    of the same underlying PipelineStep's with different parameters passed to the 'get_next' methods.
    """

    def __init__(self, **arguments):
        """The 'arguments' passed are meant to correspond to all views of this PipelineStep."""
        self.arguments = arguments

    @abstractmethod
    def get_next(self, previous: Generator, **arguments) -> Generator:
        """
        Has to be provided by the concrete implementation.

        'previous' yields a list of every currently incoming element per input stream. This is dependent on the specific
        pipeline graph as wired by the PipelineView wrappers. Thus, 'get_next' has to work for an arbitrary number of
        incoming streams.

        'get_next' does not necessarily have to run infinitely, it is simply called again after its termination (this
        is important to consider when 'get_next' itself holds some state).
        """
        pass

    def connect_to(self, previous: Union[PipelineStepView, List[PipelineStepView]] = None,
                   previous_indices: Union[int, List[int], List[List[int]]] = None) \
            -> PipelineStepView:
        """
        Returns a view which connects this pipeline step to one or several previous pipeline streams.

        The input streams can come from several other PipelineStepViews. As every PipelineStepView itself can have
        multiple output streams, one can specify the specific stream indices for every incoming PipelineStepView.

        'previous' can be either None (for the first step in a pipeline), a PipelineStepView instance if all input
        streams come from a single view, or a list of PipelineStepView instances.

        If 'previous_indices' is None, all input streams from all input views are inputted.

        If 'previous' is None, 'previous_indices' has to be None as well.
        If 'previous' if a single instance, 'previous_indices' can be Null, a single int or a list of ints.
        If 'previous' is a list of views, 'previous_indices' can be Null, a single int, a list of ints or a list of lists
        of int. In the second or third case, from all input views the given indices (or the given index) are taken.
        """

        if previous is None:
            assert previous_indices is None, 'Declaration of an input stream index for a non-existing input step.'

            previous = []
            previous_indices = []

        elif isinstance(previous, PipelineStepView):
            previous = [previous]

            if previous_indices is None:
                previous_indices = [None]
            elif isinstance(previous_indices, int):
                previous_indices = [[previous_indices]]
            elif all([isinstance(i, int) for i in previous_indices]):
                previous_indices = [previous_indices]
            else:
                assert False, 'Mismatch of the input steps and input indices.'

        elif isinstance(previous, list):
            if previous_indices is None:
                previous_indices = [None for _ in previous]
            elif isinstance(previous_indices, int):
                previous_indices = [[previous_indices] for _ in previous]
            elif len(previous_indices) == len(previous) and all([isinstance(i, int) for i in previous_indices]):
                previous_indices = [[index] for index in previous_indices]
            elif len(previous_indices) == len(previous):
                # everything okay
                pass
            else:
                assert False, 'Mismatch of the input steps and input indices.'

        if any([isinstance(view.step, FinalPipelineStep) for view in previous]):
            raise AssertionError(f'{self} binds to a FinalPipelineStep which has to be at the end of a pipeline.')

        return PipelineStepView(self, previous, previous_indices)

    def __call__(self, previous: Union[PipelineStepView, List[PipelineStepView]] = None,
                   previous_index: Union[int, List[int], List[List[int]]] = None):
        return self.connect_to(previous, previous_index)


class FirstPipelineStep(PipelineStep):
    """
    The base class for the first node in a pipeline graph (usually a data generator of some sort).

    To support features such as caching, a FirstPipelineStep must be able to stream all data and then call
    'finished_iteration'.
    """

    def __init__(self, **arguments):
        super().__init__(**arguments)

    @abstractmethod
    def get_next(self, previous: Generator, **arguments) -> Generator:
        pass

    def finished_iteration(self):
        raise IteratedThroughAll()


class FinalPipelineStep(PipelineStep, ABC):
    """
    The base class for the last node in a pipeline graph.
    """

