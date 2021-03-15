import logging
from enum import Enum
from typing import Generator, Callable

import numpy as np

from pipeline.pipeline_step import PipelineStep


class DebugStep(PipelineStep):

    def get_next(self, previous: Generator, logging_function: Callable = None, **arguments) -> Generator:
        element = next(previous)
        if logging_function: logging_function(element)
        yield element


class PreviewType(Enum):
    MINIMAL = 0,
    EXTENDED = 1


class PreviewIdentity(PipelineStep):

    def __init__(self, preview_type=PreviewType.MINIMAL, description='', **arguments):
        super().__init__(**arguments)
        self.preview_step = True
        self.preview_type = preview_type
        self.description = description

    def get_next(self, previous: Generator, **arguments) -> Generator:
        if self.preview_step:
            logging.info(10*'-')
            logging.info(f'Preview of {self.description}:')

            preview = next(previous)

            if self.preview_type == PreviewType.MINIMAL: logging.info(self._get_minimal_preview(preview))

            self.preview_step = False
            logging.info(10 * '-')

            yield preview

        else:
            yield from previous

    def _get_minimal_preview(self, element):
        if isinstance(element, tuple) or isinstance(element, list):
            return [self._get_minimal_preview(e) for e in element]
        elif isinstance(element, np.ndarray):
            return element.shape
        else:
            return str(element)
