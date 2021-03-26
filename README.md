*in development*

## Overview

This project contains the implementation of an ultra-flexible pipeline for pre-processing. It can be used for any data science or machine learning project. However, it is specifically designed to work with the generator interface of tensorflow-keras.

Any pipeline can be represented by a directed acyclic graph. Each node corresponds to an operation and can have multiple input streams and output streams. The data-flow-principle is used: the output node request data from its input nodes, they request data from their inputs, and so on.

## Simple Example

Every pipeline has at least one data source: the node at the root of the tree. It can be defined by inheriting from the FirstPipelineStep class:

```python
class IntegerStream(FirstPipelineStep):
    def __init__(self, **arguments):
        super().__init__(**arguments)
        self.next_number = 0

    def get_next(self, previous: Generator, **arguments) -> Generator:
        self.next_number += 1
        yield [self.next_number]
```

`get_next` is a generator yielding the next element. Since every node can have multiple output streams, `get_next` yields a list with one element per output stream.

After having defined the data source, the easiest thing possible is a *sequential pipeline* with a simple list of steps which are applied to every data element:

```python
pipeline = Block([
    IntegerStream(),
    Adder(),
    FilterEven()    
])
```

In this example, `Adder` and `FilterEven` can be defined as follows:

```python
# we can inherit from FunctionTransformer since we only apply a simple function
class Adder(FunctionTransformer):
    def transform(self, number, increment=0, **arguments):
        return number + increment


# here, we have to apply from PipelineStep since we discard all odd elements
class FilterEven(PipelineStep):
    def get_next(self, previous: Generator, **arguments) -> Generator:
        # previous is a generator yielding a list from every incoming input per input stream
        inputs = next(previous)
        
        # we only yield if all inputs are even
        while not all([i % 2 for i in inputs]):
            inputs = next(previous)

        yield inputs
```

Having defined the pipeline as above, we can use `pipeline.get_generator` which returns a generator yielding the processed elements:

```python
generator = pipeline.get_generator()

for e in generator:
    print(e)
```

There are many predefined pipeline steps and many useful features. A more realistic pipeline for image processing is the following, where we generate data for an autoencoder which reconstructs an image where a random block is cropped out:

```python
# SimpleMRIGenerator has one outputs stream yielding the mri image
generator = SimpleMRIGenerator(n_bins=n_bins, data_loader=load_data)()

image_steps = Block([
    Rescale(),
    Resize()
])(generator)

# DuplicateStream turns every input stream into two output streams
duplicated = DuplicateStream()(image_steps)

# hide a random block of the input image
x = HideRandomBlock()(duplicated, 0) # we only use the first output stream of duplicated
# as this is an autoencoder, we use the full image as target output
y = Identity()(duplicated, 1)

# KerasTrainingGenerator transforms the data so that it can be directly used with a keras model
train_output = KerasTrainingGenerator()([x, y])

# get_view allows us to create multiple views of the same pipeline with different arguments passed 
# to the steps. Here, bins_included is passed to the generator and allows us to do a train-test split
train_pipeline = train_output.get_view(bins_included=list(range(n_bins - 1)), **parameters)

test_output = KerasTestGenerator()([x, y])
test_pipeline = test_output.get_view(bins_included=[n_bins - 1], shuffle=False, **parameters)

# we can cache processed data on disk
train_pipeline.cache_or_load('train.cache')
test_pipeline.cache_or_load('test.cache')

# model is any tf-keras model
history = model.fit(x=train_pipeline,
                    validation_data=test_data,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch)
```

## Conda

The conda environment in conda_env.yml is used. To load all packages in conda_env.yml into your current local environment run:

'conda env export > conda_env.yml'
'pip install -r requirements.txt'

To update conda_env.yml to contain the same packages as your current local environment run:

'conda env update --file conda_env.yml  --prune'
'pip freeze > requirements.txt'