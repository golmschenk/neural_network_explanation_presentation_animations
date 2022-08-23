from typing import List

import numpy as np
from manim import Polygon, Line, Circle, BLACK, RED, VGroup, RIGHT, RoundedRectangle, LEFT


class PixelGridSquare:
    def __init__(self, number_of_pixels: int):
        self.grid_size: float = 3 * 0.4
        self.number_of_pixels = number_of_pixels
        self.pixel_size = self.grid_size / self.number_of_pixels
        self.v_group: VGroup = VGroup()
        polygon_list_list: List[List[Polygon]] = []
        for pixel_y_index in range(number_of_pixels):
            polygon_list: List[Polygon] = []
            for pixel_x_index in range(number_of_pixels):
                left, top = self.pixel_index_to_pixel_start_cartesian_xy(pixel_x_index, pixel_y_index)
                right = left + self.pixel_size
                bottom = top - self.pixel_size
                polygon_vertexes = [
                    [left, top, 0],
                    [right, top, 0],
                    [right, bottom, 0],
                    [left, bottom, 0],
                ]
                polygon = Polygon(*polygon_vertexes, color=BLACK)
                polygon_list.append(polygon)
                self.v_group.add(polygon)
            polygon_list_list.append(polygon_list)
        self.polygon_array: np.ndarray = np.array(polygon_list_list)

    def pixel_index_to_pixel_start_cartesian_xy(self, pixel_x_index: float, pixel_y_index: float) -> (float, float):
        cartesian_x_position = (-self.grid_size / 2) + (pixel_x_index * self.pixel_size)
        cartesian_y_position = (self.grid_size / 2) - (pixel_y_index * self.pixel_size)
        return cartesian_x_position, cartesian_y_position


class NeuronWithKernel:
    def __init__(self):
        self.neuron_radius: float = 0.2
        self.kernel: PixelGridSquare = PixelGridSquare(number_of_pixels=3)
        neuron_to_grid_distance = self.kernel.grid_size - self.neuron_radius
        self.indicator_lines: VGroup = VGroup(
            Line(np.array([0, 0, -1]), np.array([-neuron_to_grid_distance, self.kernel.grid_size / 2, -1]), stroke_color=BLACK),
            Line(np.array([0, 0, -1]), np.array([-neuron_to_grid_distance, -self.kernel.grid_size / 2, -1]), stroke_color=BLACK)
        )
        self.neuron: Circle = Circle(radius=self.neuron_radius, stroke_color=BLACK, fill_color=RED, fill_opacity=1.0)
        neuron_plus_indicator_lines_v_group: VGroup = VGroup(self.neuron, self.indicator_lines)
        neuron_plus_indicator_lines_v_group.next_to(self.kernel.v_group, direction=RIGHT, buff=0)
        self.v_group = VGroup(self.kernel.v_group, self.indicator_lines, self.neuron)


class LayerBuildingComplexitySubScene:
    def __init__(self):
        self.display_half_x_size: float = 5.3333
        self.display_half_y_size: float = 4
        self.display_x_margin: float = 0.2
        self.display_y_margin: float = 1
        self.layer_margin: float = 0.6
        self.layer_height = 2 * self.display_half_y_size - 2 * self.display_y_margin
        self.layer_width = (2 * self.display_half_x_size - 2 * self.layer_margin - 2 * self.display_x_margin) / 3

    def create_layers(self) -> (RoundedRectangle, RoundedRectangle, RoundedRectangle):
        gradient_layer = self.create_layer_rectangle()
        line_layer = self.create_layer_rectangle()
        corner_layer = self.create_layer_rectangle()
        gradient_layer.next_to(line_layer, direction=LEFT, buff=self.layer_margin)
        corner_layer.next_to(line_layer, direction=RIGHT, buff=self.layer_margin)
        return gradient_layer, line_layer, corner_layer

    def create_layer_rectangle(self) -> RoundedRectangle:
        return RoundedRectangle(width=self.layer_width, height=self.layer_height, corner_radius=0.3, stroke_color=BLACK)
