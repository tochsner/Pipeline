from unittest import TestCase

from pipeline.transformer import ToNumpyArray
from tests.helper import IntegerStream, Adder
from pipeline.ML_steps import KerasTrainingGenerator, KerasTestGenerator

import numpy as np


class TestKerasSteps(TestCase):

    def test_training_generator_shape(self):
        stream = IntegerStream(nr_outgoing_streams=5)()
        stream = ToNumpyArray()(stream)

        input_1 = Adder(increment=1)(stream, 0)
        input_2 = Adder(increment=2)(stream, 1)
        output_1 = Adder(increment=3)(stream, 2)
        output_2 = Adder(increment=4)(stream, 3)
        output_3 = Adder(increment=5)(stream, 4)

        train_step = KerasTrainingGenerator(batch_size=3, input_indices=[0, 1], output_indices=[2, 3, 4]) \
            ([input_1, input_2, output_1, output_2, output_3])

        generator = train_step.get_generator()

        output = next(generator)
        self.assertTrue(isinstance(output, tuple))
        self.assertTrue(len(output) == 2)
        self.assertTrue(isinstance(output[0], list))
        self.assertTrue(isinstance(output[1], list))
        self.assertTrue(len(output[0]) == 2)
        self.assertTrue(len(output[1]) == 3)
        self.assertTrue(isinstance(output[0][0], np.ndarray))
        self.assertTrue(isinstance(output[0][1], np.ndarray))
        self.assertTrue(isinstance(output[1][0], np.ndarray))
        self.assertTrue(isinstance(output[1][1], np.ndarray))
        self.assertTrue(isinstance(output[1][2], np.ndarray))
        self.assertEquals(output[0][0].shape, (3, 1))
        self.assertEquals(output[0][1].shape, (3, 1))
        self.assertEquals(output[1][0].shape, (3, 1))
        self.assertEquals(output[1][1].shape, (3, 1))
        self.assertEquals(output[1][2].shape, (3, 1))

    def test_training_generator_data(self):
        stream = IntegerStream(nr_outgoing_streams=5)()
        stream = ToNumpyArray()(stream)

        input_1 = Adder(increment=1)(stream, 0)
        input_2 = Adder(increment=2)(stream, 1)
        output_1 = Adder(increment=3)(stream, 2)
        output_2 = Adder(increment=4)(stream, 3)
        output_3 = Adder(increment=5)(stream, 4)

        train_step = KerasTrainingGenerator(batch_size=3, input_indices=[0, 1], output_indices=[2, 3, 4]) \
            ([input_1, input_2, output_1, output_2, output_3])

        generator = train_step.get_generator()

        output = next(generator)
        actual_output = (
            [
                np.array([2, 3, 4]),
                np.array([3, 4, 5])
            ],
            [
                np.array([4, 5, 6]),
                np.array([5, 6, 7]),
                np.array([6, 7, 8]),
            ]
        )

        self.assertTrue(isinstance(output, tuple))
        self.assertTrue(len(output) == 2)
        self.assertTrue(isinstance(output[0], list))
        self.assertTrue(isinstance(output[1], list))
        self.assertTrue(len(output[0]) == 2)
        self.assertTrue(len(output[1]) == 3)
        self.assertTrue(isinstance(output[0][0], np.ndarray))
        self.assertTrue(isinstance(output[0][1], np.ndarray))
        self.assertTrue(isinstance(output[1][0], np.ndarray))
        self.assertTrue(isinstance(output[1][1], np.ndarray))
        self.assertTrue(isinstance(output[1][2], np.ndarray))

        output = next(generator)
        actual_output = (
            [
                np.array([3, 4, 5]),
                np.array([4, 5, 6])
            ],
            [
                np.array([5, 6, 7]),
                np.array([6, 7, 8]),
                np.array([7, 8, 9]),
            ]
        )

        self.assertTrue(isinstance(output, tuple))
        self.assertTrue(len(output) == 2)
        self.assertTrue(isinstance(output[0], list))
        self.assertTrue(isinstance(output[1], list))
        self.assertTrue(len(output[0]) == 2)
        self.assertTrue(len(output[1]) == 3)
        self.assertTrue(isinstance(output[0][0], np.ndarray))
        self.assertTrue(isinstance(output[0][1], np.ndarray))
        self.assertTrue(isinstance(output[1][0], np.ndarray))
        self.assertTrue(isinstance(output[1][1], np.ndarray))
        self.assertTrue(isinstance(output[1][2], np.ndarray))

    def test_test_generator_shape(self):
        stream = IntegerStream(nr_outgoing_streams=5)()
        stream = ToNumpyArray()(stream)

        input_1 = Adder(increment=1)(stream, 0)
        input_2 = Adder(increment=2)(stream, 1)
        output_1 = Adder(increment=3)(stream, 2)
        output_2 = Adder(increment=4)(stream, 3)
        output_3 = Adder(increment=5)(stream, 4)

        test_step = KerasTestGenerator(input_indices=[0, 1], output_indices=[2, 3, 4]) \
            ([input_1, input_2, output_1, output_2, output_3])

        generator = test_step.get_generator()

        output = next(generator)
        self.assertTrue(isinstance(output, tuple))
        self.assertTrue(len(output) == 2)
        self.assertTrue(isinstance(output[0], list))
        self.assertTrue(isinstance(output[1], list))
        self.assertTrue(len(output[0]) == 2)
        self.assertTrue(len(output[1]) == 3)
        self.assertTrue(isinstance(output[0][0], np.ndarray))
        self.assertTrue(isinstance(output[0][1], np.ndarray))
        self.assertTrue(isinstance(output[1][0], np.ndarray))
        self.assertTrue(isinstance(output[1][1], np.ndarray))
        self.assertTrue(isinstance(output[1][2], np.ndarray))
        self.assertEquals(output[0][0].shape, (1,))
        self.assertEquals(output[0][1].shape, (1,))
        self.assertEquals(output[1][0].shape, (1,))
        self.assertEquals(output[1][1].shape, (1,))
        self.assertEquals(output[1][2].shape, (1,))

    def test_test_generator_data(self):
        stream = IntegerStream(nr_outgoing_streams=5)()
        stream = ToNumpyArray()(stream)

        stream = ToNumpyArray()(stream)

        input_1 = Adder(increment=1)(stream, 0)
        input_2 = Adder(increment=2)(stream, 1)
        output_1 = Adder(increment=3)(stream, 2)
        output_2 = Adder(increment=4)(stream, 3)
        output_3 = Adder(increment=5)(stream, 4)

        test_step = KerasTestGenerator(input_indices=[0, 1], output_indices=[2, 3, 4]) \
            ([input_1, input_2, output_1, output_2, output_3])

        generator = test_step.get_generator()

        output = next(generator)
        actual_output = (
            [
                np.array([2]),
                np.array([3])
            ],
            [
                np.array([4]),
                np.array([5]),
                np.array([6]),
            ]
        )

        self.assertListEqual(list(output[0][0]), list(actual_output[0][0]))
        self.assertListEqual(list(output[0][1]), list(actual_output[0][1]))
        self.assertListEqual(list(output[1][0]), list(actual_output[1][0]))
        self.assertListEqual(list(output[1][1]), list(actual_output[1][1]))
        self.assertListEqual(list(output[1][2]), list(actual_output[1][2]))

        output = next(generator)
        actual_output = (
            [
                np.array([3]),
                np.array([4])
            ],
            [
                np.array([5]),
                np.array([6]),
                np.array([7]),
            ]
        )

        self.assertListEqual(list(output[0][0]), list(actual_output[0][0]))
        self.assertListEqual(list(output[0][1]), list(actual_output[0][1]))
        self.assertListEqual(list(output[1][0]), list(actual_output[1][0]))
        self.assertListEqual(list(output[1][1]), list(actual_output[1][1]))
        self.assertListEqual(list(output[1][2]), list(actual_output[1][2]))
