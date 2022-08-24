from __future__ import annotations

import math
from pathlib import Path
from typing import List, Union

import numpy as np
from manim import ImageMobject, Mobject, Polygon, FadeIn, BLACK, Circle, RED, DOWN, Animation, UP, VGroup


class IsometricNeuronsLookingAtPixelsSubScene:
    def __init__(self):
        self.image_large_size = 6.0
        self.number_of_pixels = 10
        assert self.number_of_pixels % 2 == 0
        self.pixel_size = self.image_large_size / self.number_of_pixels
        self.skip_animations = False
        self.isometric_theta = -math.tau / 8
        self.isometric_scale = 0.5
        self.isometric_y_base_shift = -1
        self.isometric_y_per_z_shift = 1
        self.neuron_radius = self.pixel_size * 0.4
        self.v_group = VGroup()

        self.planetary_nebula_image_mobject = self.create_image()

        self.cartesian_pixel_grid, self.isometric_pixel_grid, self.neuron_groups = self.create_grids_and_neuron_groups()

        self.cartesian_neuron, self.cartesian_neuron_kernel = self.create_cartesian_neuron_collection()
        self.v_group.add(self.cartesian_neuron, self.cartesian_neuron_kernel)
        super().__init__()

    def create_cartesian_neuron_collection(self):
        cartesian_neuron_position = self.pixel_index_to_pixel_center_cartesian_xy(pixel_x_index=1, pixel_y_index=1)
        cartesian_neuron = self.create_neuron([cartesian_neuron_position[0], cartesian_neuron_position[1], 1])
        cartesian_neuron_kernel = self.create_neuron_kernel_centered_on_pixel_index(pixel_x_index=1, pixel_y_index=1,
                                                                                    isometric=False)
        return cartesian_neuron, cartesian_neuron_kernel

    def create_grids_and_neuron_groups(self) -> (List[Mobject], List[Mobject], List[List[NeuronGroup]]):
        cartesian_pixel_grid: List[Mobject] = []
        isometric_pixel_grid: List[Mobject] = []
        neuron_groups: List[List[NeuronGroup]] = []
        for pixel_y_index in range(self.number_of_pixels):
            if pixel_y_index != 0 and pixel_y_index != self.number_of_pixels - 1:
                neuron_groups.append([])
            for pixel_x_index in range(self.number_of_pixels):
                left, top = self.pixel_index_to_pixel_start_cartesian_xy(pixel_x_index, pixel_y_index)
                right = left + self.pixel_size
                bottom = top - self.pixel_size
                polygon_cartesian_positions = [
                    [left, top, 0],
                    [right, top, 0],
                    [right, bottom, 0],
                    [left, bottom, 0],
                ]
                polygon_isometric_positions = self.from_cartesian_position_to_isometric_position(
                    np.array(polygon_cartesian_positions))
                cartesian_polygon = self.create_pixel_grid_polygon(polygon_cartesian_positions)
                cartesian_pixel_grid.append(cartesian_polygon)
                self.v_group.add(cartesian_polygon)
                isometric_polygon = self.create_pixel_grid_polygon(polygon_isometric_positions)
                isometric_pixel_grid.append(isometric_polygon)
                self.v_group.add(isometric_polygon)
                if (pixel_x_index != 0 and pixel_x_index != self.number_of_pixels - 1 and
                        pixel_y_index != 0 and pixel_y_index != self.number_of_pixels - 1):
                    neuron_group = NeuronGroup(self, pixel_x_index, pixel_y_index)
                    neuron_groups[pixel_y_index - 1].append(neuron_group)
                    self.v_group.add(neuron_group.v_group)
        return cartesian_pixel_grid, isometric_pixel_grid, neuron_groups

    def create_image(self) -> ImageMobject:
        planetary_nebula_image_mobject = ImageMobject(
            Path('neural_network_explanation_presentation_animations/images/grayscale_ngc7293_planetary_nebula.jpg'))
        planetary_nebula_image_mobject.scale_to_fit_width(self.image_large_size)
        return planetary_nebula_image_mobject

    def pixel_index_to_pixel_center_cartesian_xy(self, pixel_x_index: float, pixel_y_index: float) -> (float, float):
        cartesian_x_position = (-self.image_large_size / 2) + (self.pixel_size / 2) + (pixel_x_index * self.pixel_size)
        cartesian_y_position = (self.image_large_size / 2) - (self.pixel_size / 2) - (pixel_y_index * self.pixel_size)
        return cartesian_x_position, cartesian_y_position

    def pixel_index_to_pixel_start_cartesian_xy(self, pixel_x_index: float, pixel_y_index: float) -> (float, float):
        cartesian_x_position = (-self.image_large_size / 2) + (pixel_x_index * self.pixel_size)
        cartesian_y_position = (self.image_large_size / 2) - (pixel_y_index * self.pixel_size)
        return cartesian_x_position, cartesian_y_position

    def pixel_index_to_pixel_start_isometric_position(self, pixel_x_index: float, pixel_y_index: float,
                                                      z_position: float = 0) -> np.ndarray:
        cartesian_x_position = (-self.image_large_size / 2) + (pixel_x_index * self.pixel_size)
        cartesian_y_position = (self.image_large_size / 2) - (pixel_y_index * self.pixel_size)
        isometric_position = self.from_cartesian_position_to_isometric_position(
            np.array([cartesian_x_position, cartesian_y_position, z_position]))
        return isometric_position

    def from_cartesian_position_to_isometric_position(self, cartesian_position: np.ndarray) -> np.ndarray:
        if len(cartesian_position.shape) == 1:
            cartesian_xy_position = cartesian_position[:2]
            cartesian_z_position = cartesian_position[[2]]
        else:
            cartesian_xy_position = cartesian_position[:, :2]
            cartesian_z_position = cartesian_position[:, [2]]
        rotation = np.array([[np.cos(self.isometric_theta), - np.sin(self.isometric_theta)],
                             [np.sin(self.isometric_theta), np.cos(self.isometric_theta)]])
        rotated_position = cartesian_xy_position.dot(rotation)
        if len(cartesian_position.shape) == 1:
            non_shifted_isometric_xy_position = np.array([rotated_position[0],
                                                          rotated_position[1] * self.isometric_scale])
            non_shifted_isometric_position = np.concatenate([non_shifted_isometric_xy_position,
                                                             cartesian_z_position], axis=0)
            isometric_position = non_shifted_isometric_position
            y_shift = self.isometric_y_base_shift + (isometric_position[2] * self.isometric_y_per_z_shift)
            isometric_position[1] += y_shift
        else:
            non_shifted_isometric_xy_position = np.stack([rotated_position[:, 0],
                                                          rotated_position[:, 1] * self.isometric_scale], axis=1)
            non_shifted_isometric_position = np.concatenate([non_shifted_isometric_xy_position,
                                                             cartesian_z_position], axis=1)
            isometric_position = non_shifted_isometric_position
            y_shift = self.isometric_y_base_shift + (isometric_position[:, 2] * self.isometric_y_per_z_shift)
            isometric_position[:, 1] += y_shift
        return isometric_position

    @staticmethod
    def create_pixel_grid_polygon(vertexes: Union[np.ndarray, List[List[float]]]) -> Polygon:
        polygon = Polygon(*vertexes, color=BLACK)
        return polygon

    def create_neuron_above_pixel_position(self, pixel_x_index, pixel_y_index) -> Circle:
        isometric_position = self.pixel_index_to_pixel_start_isometric_position(pixel_x_index + 0.5,
                                                                                pixel_y_index + 0.5,
                                                                                z_position=1)
        isometric_position[2] = 1  # Z position for occlusions.
        circle = self.create_neuron(position=isometric_position)
        return circle

    def create_neuron(self, position) -> Circle:
        circle = Circle(radius=self.neuron_radius, stroke_color=BLACK, fill_color=RED, fill_opacity=1.0)
        circle.move_to(position)
        return circle

    def create_neuron_kernel_centered_on_pixel_index(self, pixel_x_index: float, pixel_y_index: float,
                                                     isometric: bool = True) -> Polygon:
        kernel_size = 3
        center_pixel_x_index = pixel_x_index + 0.5
        center_pixel_y_index = pixel_y_index + 0.5
        start_pixel_x_index = center_pixel_x_index - (kernel_size / 2)
        start_pixel_y_index = center_pixel_y_index - (kernel_size / 2)
        end_pixel_x_index = center_pixel_x_index + (kernel_size / 2)
        end_pixel_y_index = center_pixel_y_index + (kernel_size / 2)
        start_cartesian_position = self.pixel_index_to_pixel_start_cartesian_xy(start_pixel_x_index,
                                                                                start_pixel_y_index)
        end_cartesian_position = self.pixel_index_to_pixel_start_cartesian_xy(end_pixel_x_index,
                                                                              end_pixel_y_index)
        left, top = start_cartesian_position[0], start_cartesian_position[1]
        right, bottom = end_cartesian_position[0], end_cartesian_position[1]
        z_position = 0.01
        polygon_positions = np.array([
            [left, top, z_position],
            [right, top, z_position],
            [right, bottom, z_position],
            [left, bottom, z_position],
        ])
        if isometric:
            polygon_positions = self.from_cartesian_position_to_isometric_position(polygon_positions)
        polygon = Polygon(*polygon_positions, color=RED, fill_color=RED, fill_opacity=0.5, stroke_opacity=0.0)
        return polygon

    def create_neuron_output_centered_on_pixel_index(self, pixel_x_index: float, pixel_y_index: float,
                                                     isometric: bool = True) -> Polygon:
        output_size = 1
        center_pixel_x_index = pixel_x_index + 0.5
        center_pixel_y_index = pixel_y_index + 0.5
        start_pixel_x_index = center_pixel_x_index - (output_size / 2)
        start_pixel_y_index = center_pixel_y_index - (output_size / 2)
        end_pixel_x_index = center_pixel_x_index + (output_size / 2)
        end_pixel_y_index = center_pixel_y_index + (output_size / 2)
        start_cartesian_position = self.pixel_index_to_pixel_start_cartesian_xy(start_pixel_x_index,
                                                                                start_pixel_y_index)
        end_cartesian_position = self.pixel_index_to_pixel_start_cartesian_xy(end_pixel_x_index,
                                                                              end_pixel_y_index)
        left, top = start_cartesian_position[0], start_cartesian_position[1]
        right, bottom = end_cartesian_position[0], end_cartesian_position[1]
        z_position = 2.0
        polygon_positions = np.array([
            [left, top, z_position],
            [right, top, z_position],
            [right, bottom, z_position],
            [left, bottom, z_position],
        ])
        if isometric:
            polygon_positions = self.from_cartesian_position_to_isometric_position(polygon_positions)
        polygon = Polygon(*polygon_positions, color=RED, fill_color=RED, fill_opacity=1.0, stroke_color=BLACK,
                          stroke_opacity=1.0)
        return polygon


class NeuronGroup:
    def __init__(self, scene: IsometricNeuronsLookingAtPixelsSubScene, pixel_x_index: float, pixel_y_index: float):
        self.neuron: Circle = scene.create_neuron_above_pixel_position(pixel_x_index, pixel_y_index)
        self.neuron_animation_created: bool = False
        self.kernel: Polygon = scene.create_neuron_kernel_centered_on_pixel_index(pixel_x_index, pixel_y_index)
        self.kernel_animation_created: bool = False
        self.output: Polygon = scene.create_neuron_output_centered_on_pixel_index(pixel_x_index, pixel_y_index)
        self.output_animation_created: bool = False
        self.cartesian_output: Polygon = scene.create_neuron_output_centered_on_pixel_index(
            pixel_x_index, pixel_y_index, isometric=False)
        self.v_group = VGroup(self.neuron, self.kernel, self.output, self.cartesian_output)

    def create_neuron_animation(self) -> Animation:
        self.neuron_animation_created = True
        return FadeIn(self.neuron, shift=DOWN)

    def create_kernel_animation(self) -> Animation:
        self.kernel_animation_created = True
        return FadeIn(self.kernel, shift=DOWN)

    def create_output_animation(self) -> Animation:
        self.output_animation_created = True
        return FadeIn(self.output, shift=UP)
