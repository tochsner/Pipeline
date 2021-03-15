from random import randint
from typing import Generator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from cv2 import resize, INTER_AREA, fastNlMeansDenoising, erode, dilate

from pipeline.pipeline_step import PipelineStep
from pipeline.transformer import FunctionTransformer

import cv2

class Rescale(FunctionTransformer):

    def transform(self, img, **arguments):
        if np.max(img - np.min(img)) == 0: return img - np.min(img)
        return (img - np.min(img)) / np.max(img - np.min(img))


class Resize(FunctionTransformer):

    def transform(self, img, width=384, height=384, **arguments):
        return resize(img, (width, height), interpolation=INTER_AREA)


class AddChannel(FunctionTransformer):

    def transform(self, array, **arguments):
        return array.reshape(array.shape + (1,))


class ToRGB(FunctionTransformer):

    def transform(self, array, **arguments):
        return np.repeat(array.reshape(array.shape + (1,)), repeats=3, axis=-1)


class Reshape(FunctionTransformer):

    def transform(self, array, shape=(1,), **arguments):
        return array.reshape(shape)


class RandomlyCrop(FunctionTransformer):

    def transform(self, image, crop_width=128, crop_height=128, **arguments):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = image[y: y + crop_height, x: x + crop_width]

        return crop


class AverageFilter(PipelineStep):

    def get_next(self, previous: Generator, min_avg=0.1, **arguments) -> Generator:
        for stream in previous:
            if all([np.mean(image) > min_avg for image in stream]): yield stream


class Denoising(FunctionTransformer):

    def transform(self, img, h=10, template_window_size=3, search_window_size=7, **arguments):
        if img.shape[-1] <= 3:
            return fastNlMeansDenoising(img, None, h, template_window_size, search_window_size)

        denoised_channels = list()

        for c in range(img.shape[-1]):
            denoised_channels.append(
                fastNlMeansDenoising(img[:, :, c], None, h, template_window_size, search_window_size))
            denoised_channels[-1] = denoised_channels[-1].reshape(denoised_channels[-1].shape + (1,))

        return np.concatenate(denoised_channels, axis=2)


class Dilation(FunctionTransformer):

    def transform(self, img, kernel_size=3, iterations=1, **arguments):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return dilate(img, kernel, iterations=iterations)


class Erosion(FunctionTransformer):

    def transform(self, img, kernel_size=3, iterations=1, **arguments):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return erode(img, kernel, iterations=iterations)


class ShowImage(FunctionTransformer):

    def transform(self, img, **arguments):
        plt.imshow(img)
        plt.show()
        return img


class HideHalfImage(FunctionTransformer):

    def transform(self, img, **arguments):
        covered = np.copy(img)

        if randint(0, 1):
            covered[:, :covered.shape[1] / 2] = 0
        else:
            covered[:, covered.shape[1] / 2:] = 0

        return covered


class HideQuarterImage(FunctionTransformer):

    def transform(self, img, **arguments):
        covered = np.copy(img)

        if randint(0, 1):
            if randint(0, 1):
                covered[:covered.shape[0] // 2, :covered.shape[1] // 2] = 0
            else:
                covered[covered.shape[0] // 2:, :covered.shape[1] // 2] = 0
        else:
            if randint(0, 1):
                covered[:covered.shape[0] // 2, covered.shape[1] // 2:] = 0
            else:
                covered[covered.shape[0] // 2:, covered.shape[1] // 2:] = 0

        return covered


class HideRandomBlock(FunctionTransformer):

    def transform(self, img, min_block_size=(0, 0), max_block_size=(0, 0), **arguments):
        covered = np.copy(img)

        size = randint(min_block_size[0], max_block_size[0]), randint(min_block_size[1], max_block_size[1])

        x = randint(0, img.shape[0] - size[0] - 1)
        y = randint(0, img.shape[1] - size[1] - 1)

        covered[x:x + size[0], y:y + size[1]] = 0

        return covered


class GetNormalizedAxis(FunctionTransformer):

    def transform(self, input, axis='x', **arguments):
        if axis == 'x': return np.array([[i/input.shape[0]] * input.shape[1] for i in range(input.shape[0])])
        if axis == 'y': return np.array([[i/input.shape[1] for i in range(input.shape[1])]] * input.shape[0])


class GaussianPyramid(PipelineStep):

    def get_next(self, previous: Generator, num_layers=1, **arguments) -> Generator:
        input_images = next(previous)

        gaussians = [self._get_gaussian(image.copy(), num_layers) for image in input_images]
        all_gaussians = list(chain.from_iterable(gaussians))

        yield all_gaussians

    def _get_gaussian(self, image, num_layers):
        lower = image
        gaussian_pyr = [image]
        for _ in range(num_layers - 1):
            lower = cv2.pyrDown(lower)
            gaussian_pyr.append(lower)

        return gaussian_pyr


class LaplacianPyramid(FunctionTransformer):

    def get_next(self, previous: Generator, num_layers=1, **arguments) -> Generator:
        input_images = next(previous)

        laplacians = [self._get_laplacian(image.copy(), num_layers) for image in input_images]
        all_laplacian = list(chain.from_iterable(laplacians))

        yield all_laplacian

    def _get_laplacian(self, image, num_layers):
        gaussian_pyr = [image]
        lower = image
        for _ in range(num_layers - 1):
            lower = cv2.pyrDown(lower)
            gaussian_pyr.append(lower)

        laplacian_top = gaussian_pyr[-1]

        laplacian_pyr = [laplacian_top]
        for i in range(num_layers - 1, 0, -1):
            size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
            gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
            laplacian = gaussian_pyr[i - 1] - gaussian_expanded
            laplacian_pyr.append(laplacian)

        return reversed(laplacian_pyr)
