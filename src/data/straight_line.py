import math
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class StraightLine(Dataset):

    def __init__(self,
                 img_h: Optional[int] = 32,
                 img_w: Optional[int] = 32,
                 num_images: Optional[int] = 50,
                 n_black_pixels: Optional[int] = 1,
                 transform: Optional[Callable] = None):
        """
        Dataset that generates images with a straight line.
        :param img_h: Height of the images.
        :param img_w: Width of the images.
        :param num_images: Number of images to generate.
        :param n_black_pixels: The number of black pixels to add to the image.
        :param transform: Optional transform to be applied on a sample.
        """
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.num_images = num_images
        self.transform = transform
        self.n_black_pixels = n_black_pixels

        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        """
        Returns the number of images in the dataset.
        :return: Number of images in the dataset.
        """
        return self.num_images

    def _get_random_line_coords(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns the coordinates of a random straight line (create two random x,y coordinates).
        :return: a Tuple of two Tuples of x,y coordinates.
        """
        if random.random() < 0.5:
            x1 = 2
            x2 = self.img_w - 2
            y1 = random.randint(2, self.img_h - 2)
            y2 = self.img_h - y1
        else:
            y1 = 2
            y2 = self.img_h - 2
            x1 = random.randint(2, self.img_w - 2)
            x2 = self.img_w - x1

        return (x1, y1), (x2, y2)

    def _create_l_image(self, line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Image:
        """
        Creates a black grayscale image with a random straight line in withe drawn on it.
        :param line_coords: The coordinates of the line to draw.
        :return: The image.
        """
        img = Image.new('L', (self.img_w, self.img_h), color=0)
        draw = ImageDraw.Draw(img)
        draw.line(line_coords, fill=255, width=1)
        return img

    def _create_image(
            self,
            line_coords: Tuple[Tuple[int, int], Tuple[int, int]],
            n_black_pixels: Optional[int] = None
    ) -> Image:
        """
        Creates either a RBG or a grayscale image with a random straight line in withe drawn on it.
        :param idx: The index of the image.
        :param line_coords: The coordinates of the line to draw.
        :param n_black_pixels: The number of black pixels to add to the middle of the line.
        :return: The image.
        """
        if n_black_pixels is None:
            n_black_pixels = self.n_black_pixels

        img = self._create_l_image(line_coords)

        # add a black pixel in the middle (discontinuous line)
        if n_black_pixels > 0:
            img = np.array(img)
            line_center = (line_coords[0][0] + line_coords[1][0]) // 2, (line_coords[0][1] + line_coords[1][1]) // 2
            all_line_coords = np.argwhere(img > 128)
            center_point_idx = np.sum(np.abs(
                all_line_coords - np.array([line_center[1], line_center[0]]).reshape(1, 2).repeat(
                    all_line_coords.shape[0], axis=0)), axis=1).argmin()
            n_black = min(n_black_pixels, all_line_coords.shape[0] - 2)
            lower_idx = center_point_idx - n_black // 2
            upper_idx = center_point_idx + (n_black - (center_point_idx - lower_idx))
            idxs = np.array([list(all_line_coords[i]) for i in range(lower_idx, upper_idx)])
            img[idxs[:, 0], idxs[:, 1]] = 0
            img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        return img

    def get_item(
            self,
            idx: int,
            line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
            n_black_pixels: Optional[int] = 0,
    ):
        """
        Returns an image with a random straight line drawn on it.
        :param idx: Index of the image to return (has no effect)
        :param line_coords: The starting coordinates of the line to draw.
        :param n_black_pixels: The number of black pixels to add to the middle of the line.
        :return: The image
        """
        if line_coords is None:
            line_coords = self._get_random_line_coords()

        images = self._create_image(line_coords, n_black_pixels=n_black_pixels)
        return images, {'line_coords': line_coords, 'angle': math.atan(
            (line_coords[1][1] - line_coords[0][1]) / (1e-10 + line_coords[1][0] - line_coords[0][0]))}

    def __getitem__(self, idx: int):
        return self.get_item(idx)
