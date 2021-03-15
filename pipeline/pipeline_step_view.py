from __future__ import annotations

import logging
import pickle
from os.path import isfile
from queue import Queue
from random import shuffle
from typing import List, Generator

from pipeline.exceptions import IteratedThroughAll


class PipelineStepView:
    """
    A PipelineStepView acts as a wrapper for an underlying PipelineStep. A view manages the incoming input streams
    and the arguments passed to the 'get_next' methods of this and the preceding PipelineStep instances.

    A PipelineStepView can have multiple input and multiple output streams. New data from the incoming streams is always
    requested

    A PipelineStepView can be cached in order to speed up the process.
    """

    def __init__(self, step: PipelineStep, previous: List[PipelineStepView], previous_indices: List[List[int]],
                 is_cached=False, cache=None, next_cache_index=0, **arguments):
        combined_arguments = {**step.arguments, **arguments}

        self.step = step
        self.arguments = combined_arguments
        self.previous = previous
        self.previous_indices = previous_indices

        self.incoming_generators = [p.get_generator(i) for p, i in zip(self.previous, self.previous_indices)]
        self.outgoing_generator = self.step.get_next(self._collect_incoming_data(), **self.arguments)

        self.outgoing_data_queues = []

        self.is_cached = is_cached
        self.cache = [] if cache is None else cache
        self.next_cache_index = next_cache_index
        if self.is_cached: self.outgoing_generator = self._cache_generator()

    def get_generator(self, indices: List[int] = None) -> Generator:
        """
        Yields the output streams of this PipelineStepView, specified by 'indices'. New data is requested from the
        incoming streams only once an index is requested for the second time since the last retrieval.
        """
        while True:
            try:

                if indices is None:     # load all outgoing data
                    outgoing_data = next(self.outgoing_generator)
                    self._extend_queues_if_necessary(len(outgoing_data))

                    if all([queue.empty() for queue in self.outgoing_data_queues]):
                        yield outgoing_data

                    else:
                        for i, data in enumerate(outgoing_data): self.outgoing_data_queues[i].put(data)
                        yield [queue.get() for queue in self.outgoing_data_queues]

                else:
                    self._extend_queues_if_necessary(max(indices) + 1)

                    if any([queue.empty() for i, queue in enumerate(self.outgoing_data_queues) if i in indices]):
                        outgoing_data = next(self.outgoing_generator)
                        self._extend_queues_if_necessary(len(outgoing_data))
                        for i, data in enumerate(outgoing_data): self.outgoing_data_queues[i].put(data)

                    yield [queue.get() for i, queue in enumerate(self.outgoing_data_queues) if i in indices]

            except StopIteration:
                # once the get_next method of the PipelineStep corresponding to this instance has finished,
                # create a new generator yielding from it
                self.outgoing_generator = self.step.get_next(self._collect_incoming_data(), **self.arguments)

    def _extend_queues_if_necessary(self, num_queues):
        if len(self.outgoing_data_queues) < num_queues:
            self.outgoing_data_queues += [Queue() for _ in range(num_queues - len(self.outgoing_data_queues))]

    def _collect_incoming_data(self) -> Generator:
        """
        Yields a list containing the incoming element for every input stream.
        """
        while True:
            incoming_data = []

            for generator in self.incoming_generators:
                incoming_data += next(generator)

            yield incoming_data

    def get_view(self, **view_arguments) -> PipelineStepView:
        """
        Returns a new view with the same wiring as this PipelineStepView instance. The view arguments are also
        identical expect the ones provided in 'view_arguments'.
        """
        return self._change_view_references(dict(), **view_arguments)

    def _change_view_references(self, references: dict,  **view_arguments) -> PipelineStepView:
        """
        Recursively clones the PipelineStepView graph and returns the new view corresponding to this instance. The newly
        created views are wired identically to the original ones and contain the same view arguments expect the
        ones provided in 'view_arguments'.
        """
        for p in self.previous:
            p._change_view_references(references, **view_arguments)

        previous = [references[p] for p in self.previous]
        combined_arguments = {**self.arguments, **view_arguments}

        if self not in references:
            references[self] = PipelineStepView(self.step, previous, self.previous_indices,
                                                self.is_cached, self.cache, self.next_cache_index, **combined_arguments)

        return references[self]

    def generate_all_data(self) -> List:
        """
        Generates all data until the data source is exhausted. Returns a list of all data which would have been
        outputted.
        """
        view = self.get_view(all_then_stop=True)

        data = []

        generator = view.get_generator()
        try:
            while True:
                a = next(generator)
                data += [a]
        except IteratedThroughAll:
            pass

        return data

    def cache_or_load(self, filepath: str):
        """
        Loads data from 'filepath'. If 'filepath' does not exist, the data is first cached using 'generate_all_data'.

        After loading the data, this PipelineStepView acts identically to before. The outgoing data however is not
        produced on demand by the entire preceding pipeline, but directly obtained from the cache.

        If, on a cached view, the argument 'shuffle' is provided (using the view arguments) and set to true, then a
        cached PipelineStepView simply returns random elements in its cache. Otherwise, it simply loops through all
        elements in the cache.

        If, on a cached view, the argument 'all_then_stop' is provided (using the view arguments) and set to true,
        then the view iterates through the cache once and then raises a IteratedThroughAll exception.
        """
        if not isfile(filepath): self._cache_to_file(filepath)
        self._load_from_cache(filepath)
        logging.info(f'Loaded {len(self.cache)} elements in cache.')

    def _cache_to_file(self, filepath: str):
        self.cache = self.generate_all_data()

        with open(filepath, 'wb') as file:
            pickle.dump(self.cache, file)

    def _load_from_cache(self, filepath: str):
        self.cache = []

        with open(filepath, 'rb') as file:
            self.cache = pickle.load(file)

        self.is_cached = True
        self.next_cache_index = 0

        self.outgoing_generator = self._cache_generator()

    def _cache_generator(self):
        while True:
            if self.next_cache_index == 0 and 'shuffle' in self.arguments and self.arguments['shuffle']:
                shuffle(self.cache)

            yield self.cache[self.next_cache_index]
            self.next_cache_index = (self.next_cache_index + 1) % len(self.cache)

            if 'all_then_stop' in self.arguments and self.arguments['all_then_stop'] and self.next_cache_index == 0:
                raise IteratedThroughAll()
