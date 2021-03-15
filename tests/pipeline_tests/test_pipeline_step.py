from unittest import TestCase

from pipeline.control_flow import Identity
from tests.helper import IntegerStream


class TestPipelineStep(TestCase):

    def test_sequential_connect_to(self):
        # one step
        stream = IntegerStream(nr_outgoing_streams=1)()
        view = stream.get_view()
        generator = view.get_generator()
        self.assertEqual(next(generator), [1])
        self.assertEqual(next(generator), [2])
        self.assertEqual(next(generator), [3])
        self.assertEqual(next(generator), [4])
        self.assertEqual(next(generator), [5])

        # three steps without indices
        stream = IntegerStream(nr_outgoing_streams=1)()
        stream = Identity()(stream)
        stream = Identity()(stream)
        view = stream.get_view()
        generator = view.get_generator()
        self.assertEqual(next(generator), [1])
        self.assertEqual(next(generator), [2])
        self.assertEqual(next(generator), [3])
        self.assertEqual(next(generator), [4])
        self.assertEqual(next(generator), [5])

        # three steps with indices
        stream = IntegerStream(nr_outgoing_streams=1)()
        stream = Identity()(stream, 0)
        stream = Identity()(stream, 0)
        view = stream.get_view()
        generator = view.get_generator()
        self.assertEqual(next(generator), [1])
        self.assertEqual(next(generator), [2])
        self.assertEqual(next(generator), [3])
        self.assertEqual(next(generator), [4])
        self.assertEqual(next(generator), [5])

        # three steps without indices and four parallel streams
        stream = IntegerStream(nr_outgoing_streams=4)()
        stream = Identity()(stream)
        stream = Identity()(stream)
        view = stream.get_view()
        generator = view.get_generator()
        self.assertListEqual(next(generator), [1] * 4)
        self.assertListEqual(next(generator), [2] * 4)
        self.assertListEqual(next(generator), [3] * 4)
        self.assertListEqual(next(generator), [4] * 4)
        self.assertListEqual(next(generator), [5] * 4)

    def test_complicated_graph(self):
        input_stream = IntegerStream(nr_outgoing_streams=5)()

        a = Identity()(input_stream, [0, 1])
        b = Identity()(a)
        c = Identity()([b, input_stream], [None, [2, 3]])
        d = Identity()([c, input_stream], [[0, 1], [3, 4]])
        e = Identity()([c, d], [[2, 3], None])

        view = e.get_view()
        generator = view.get_generator()

        self.assertListEqual(next(generator), [1, 1, 1, 1, 2, 1])
        self.assertListEqual(next(generator), [2, 3, 2, 2, 4, 2])

        input_stream = IntegerStream(nr_outgoing_streams=4)()

        a = Identity()(input_stream, [0, 1])
        b = Identity()(input_stream, [1, 2])

        c = Identity()([a, b, input_stream])
        d = Identity()([a, c, b], [[0], None, [0]])

        view = d.get_view()
        generator = view.get_generator()

        self.assertListEqual(next(generator), [1, 2, 1, 3, 1, 3, 4, 2, 1, 5])
        self.assertListEqual(next(generator), [4, 5, 2, 8, 3, 6, 9, 5, 2, 10])
