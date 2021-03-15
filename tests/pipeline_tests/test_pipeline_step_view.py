from unittest import TestCase

from pipeline.control_flow import Identity
from tests.helper import IntegerStream, Adder


class TestPipelineStepView(TestCase):

    def test_sequential(self):
        input_stream = IntegerStream()()
        adder = Adder()(input_stream)
        output = Identity()(adder)

        view1 = output.get_view(increment=0)
        view2 = output.get_view(increment=1)
        view3 = output.get_view(increment=2)

        generator1 = view1.get_generator()
        generator2 = view2.get_generator()
        generator3 = view3.get_generator()

        self.assertListEqual(next(generator1), [1])
        self.assertListEqual(next(generator1), [2])

        self.assertListEqual(next(generator2), [4])
        self.assertListEqual(next(generator2), [5])

        self.assertListEqual(next(generator3), [7])
        self.assertListEqual(next(generator3), [8])

        self.assertListEqual(next(generator1), [7])
        self.assertListEqual(next(generator1), [8])

        input_stream = IntegerStream()()
        adder = Adder()(input_stream)
        adder = Adder()(adder)
        output = Identity()(adder)

        view1 = output.get_view(increment=0)
        view2 = output.get_view(increment=1)
        view3 = output.get_view(increment=2)

        generator1 = view1.get_generator()
        generator2 = view2.get_generator()
        generator3 = view3.get_generator()

        self.assertListEqual(next(generator1), [1])
        self.assertListEqual(next(generator1), [2])

        self.assertListEqual(next(generator2), [5])
        self.assertListEqual(next(generator2), [6])

        self.assertListEqual(next(generator3), [9])
        self.assertListEqual(next(generator3), [10])

        self.assertListEqual(next(generator1), [7])
        self.assertListEqual(next(generator1), [8])

    def test_parallel(self):
        input_stream = IntegerStream(nr_outgoing_streams=2)()
        adder = Adder()(input_stream)
        adder1 = Adder()(adder, 0)
        adder1 = Adder()(adder1)
        adder2 = Adder()(adder, 1)
        output = Identity()([adder1, adder2])

        view1 = output.get_view(increment=0)
        view2 = output.get_view(increment=1)
        view3 = output.get_view(increment=2)

        generator1 = view1.get_generator()
        generator2 = view2.get_generator()
        generator3 = view3.get_generator()

        self.assertListEqual(next(generator1), [1, 1])
        self.assertListEqual(next(generator1), [2, 2])
        self.assertListEqual(next(generator2), [6, 5])
        self.assertListEqual(next(generator2), [7, 6])
        self.assertListEqual(next(generator3), [11, 9])
        self.assertListEqual(next(generator3), [12, 10])
        self.assertListEqual(next(generator2), [10, 9])
        self.assertListEqual(next(generator2), [11, 10])
