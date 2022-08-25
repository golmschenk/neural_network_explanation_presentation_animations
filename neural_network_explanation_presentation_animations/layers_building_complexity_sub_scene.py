from typing import List, Union

import numpy as np
from colour import Color
from manim import Polygon, Line, Circle, BLACK, RED, VGroup, RIGHT, RoundedRectangle, LEFT, BLUE, DOWN, GRAY, WHITE, \
    DARK_GRAY, LIGHT_GRAY, GREEN, YELLOW, UP, Arrow, rgba_to_color, Scene, ReplacementTransform, Create, FadeIn, FadeOut


class PixelGridSquare:
    def __init__(self, number_of_pixels: int, color_array: np.ndarray):
        self.grid_size: float = 3 * 0.4
        self.number_of_pixels = number_of_pixels
        self.pixel_size = self.grid_size / self.number_of_pixels
        assert color_array.shape == (number_of_pixels, number_of_pixels)
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
                polygon = Polygon(*polygon_vertexes, stroke_color=BLACK,
                                  fill_color=color_array[pixel_y_index, pixel_x_index], fill_opacity=1.0)
                polygon_list.append(polygon)
                self.v_group.add(polygon)
            polygon_list_list.append(polygon_list)
        self.polygon_array: np.ndarray = np.array(polygon_list_list).squeeze()

    def pixel_index_to_pixel_start_cartesian_xy(self, pixel_x_index: float, pixel_y_index: float) -> (float, float):
        cartesian_x_position = (-self.grid_size / 2) + (pixel_x_index * self.pixel_size)
        cartesian_y_position = (self.grid_size / 2) - (pixel_y_index * self.pixel_size)
        return cartesian_x_position, cartesian_y_position


class NeuronWithKernel:
    def __init__(self, neuron_color: Union[Color, str], color_array: np.ndarray, number_of_pixels: int = 3):
        self.neuron_radius: float = 0.2
        self.kernel: PixelGridSquare = PixelGridSquare(number_of_pixels=number_of_pixels, color_array=color_array)
        neuron_to_grid_distance = self.kernel.grid_size - self.neuron_radius
        self.indicator_lines: VGroup = VGroup(
            self.create_indicator_line(np.array([0, 0, -1]),
                                       np.array([-neuron_to_grid_distance, self.kernel.grid_size / 2, -1])),
            self.create_indicator_line(np.array([0, 0, -1]),
                                       np.array([-neuron_to_grid_distance, -self.kernel.grid_size / 2, -1]))
        )
        self.neuron: Circle = Circle(radius=self.neuron_radius, stroke_color=BLACK, fill_color=neuron_color,
                                     fill_opacity=1.0)
        neuron_plus_indicator_lines_v_group: VGroup = VGroup(self.neuron, self.indicator_lines)
        neuron_plus_indicator_lines_v_group.next_to(self.kernel.v_group, direction=RIGHT, buff=0)
        self.v_group = VGroup(self.kernel.v_group, self.indicator_lines, self.neuron)

    @staticmethod
    def create_indicator_line(start_position: np.ndarray, end_position: np.ndarray) -> Line:
        return Line(start_position, end_position, stroke_color=BLACK, z_index=-1)


class LayerBuildingComplexitySubScene:
    def __init__(self):
        self.display_half_x_size: float = 5.3333
        self.display_half_y_size: float = 4
        self.display_x_margin: float = 0.2
        self.display_y_margin: float = 1
        self.layer_margin: float = 0.6
        self.layer_height = 2 * self.display_half_y_size - 2 * self.display_y_margin
        self.layer_width = (2 * self.display_half_x_size - 2 * self.layer_margin - 2 * self.display_x_margin) / 3
        self.v_group = VGroup()
        gradient_layer, line_layer, corner_layer = self.create_layers()
        self.gradient_layer: RoundedRectangle = gradient_layer
        self.v_group.add(self.gradient_layer)
        self.line_layer: RoundedRectangle = line_layer
        self.v_group.add(self.line_layer)
        self.corner_layer: RoundedRectangle = corner_layer
        self.v_group.add(self.corner_layer)
        self.gradient_layer_to_line_layer_arrow: Arrow = Arrow(
            self.gradient_layer.get_right(), self.line_layer.get_left(), buff=0.0, stroke_color=BLACK, fill_color=BLACK)
        self.v_group.add(self.gradient_layer_to_line_layer_arrow)
        self.line_layer_to_corner_layer_arrow: Arrow = Arrow(
            self.line_layer.get_right(), self.corner_layer.get_left(), buff=0.0, stroke_color=BLACK, fill_color=BLACK)
        self.v_group.add(self.line_layer_to_corner_layer_arrow)
        dark_to_light_gradient_neuron_color = RED
        light_to_dark_gradient_neuron_color = BLUE
        self.dark_to_light_gradient_neuron: NeuronWithKernel = NeuronWithKernel(
            dark_to_light_gradient_neuron_color, np.array([[DARK_GRAY, LIGHT_GRAY, WHITE],
                                                           [DARK_GRAY, LIGHT_GRAY, WHITE],
                                                           [DARK_GRAY, LIGHT_GRAY, WHITE]]))
        self.dark_to_light_gradient_neuron.v_group.move_to(self.gradient_layer)
        self.v_group.add(self.dark_to_light_gradient_neuron.v_group)
        self.light_to_dark_gradient_neuron: NeuronWithKernel = NeuronWithKernel(
            light_to_dark_gradient_neuron_color, np.array([[WHITE, LIGHT_GRAY, DARK_GRAY],
                                                           [WHITE, LIGHT_GRAY, DARK_GRAY],
                                                           [WHITE, LIGHT_GRAY, DARK_GRAY]]))
        self.light_to_dark_gradient_neuron.v_group.next_to(self.dark_to_light_gradient_neuron.v_group, direction=DOWN,
                                                           buff=self.layer_height / 10)
        self.v_group.add(self.light_to_dark_gradient_neuron.v_group)
        vertical_line_neuron_color = GREEN
        self.vertical_line_neuron: NeuronWithKernel = NeuronWithKernel(
            vertical_line_neuron_color,
            np.array([[WHITE, light_to_dark_gradient_neuron_color, dark_to_light_gradient_neuron_color],
                      [WHITE, light_to_dark_gradient_neuron_color, dark_to_light_gradient_neuron_color],
                      [WHITE, light_to_dark_gradient_neuron_color, dark_to_light_gradient_neuron_color]]))
        self.vertical_line_neuron.v_group.move_to(self.line_layer)
        self.v_group.add(self.vertical_line_neuron.v_group)
        horizontal_line_neuron_color = YELLOW
        self.horizontal_line_neuron: NeuronWithKernel = NeuronWithKernel(
            horizontal_line_neuron_color,
            np.array([[WHITE, WHITE, WHITE, WHITE, WHITE],
                      [WHITE, WHITE, WHITE, WHITE, WHITE],
                      [LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY],
                      [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
                      [LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY]]),
            number_of_pixels=5)
        self.horizontal_line_neuron.v_group.next_to(self.vertical_line_neuron.v_group, direction=DOWN,
                                                    buff=self.layer_height / 10)
        self.v_group.add(self.horizontal_line_neuron.v_group)
        self.vertical_line_in_original_image_pixel_grid: PixelGridSquare = PixelGridSquare(
            number_of_pixels=5,
            color_array=np.array([[WHITE, WHITE, LIGHT_GRAY, DARK_GRAY, LIGHT_GRAY],
                                  [WHITE, WHITE, LIGHT_GRAY, DARK_GRAY, LIGHT_GRAY],
                                  [WHITE, WHITE, LIGHT_GRAY, DARK_GRAY, LIGHT_GRAY],
                                  [WHITE, WHITE, LIGHT_GRAY, DARK_GRAY, LIGHT_GRAY],
                                  [WHITE, WHITE, LIGHT_GRAY, DARK_GRAY, LIGHT_GRAY]]))
        self.vertical_line_in_original_image_pixel_grid.v_group.next_to(self.vertical_line_neuron.kernel.v_group,
                                                                        direction=UP, buff=self.layer_height / 10)
        self.v_group.add(self.vertical_line_in_original_image_pixel_grid.v_group)
        self.corner_neuron_kernel: PixelGridSquare = PixelGridSquare(
            number_of_pixels=3,
            color_array=np.array([[WHITE, horizontal_line_neuron_color, horizontal_line_neuron_color],
                                  [vertical_line_neuron_color, WHITE, WHITE],
                                  [vertical_line_neuron_color, WHITE, WHITE]]))
        self.corner_neuron_kernel.v_group.move_to(self.corner_layer.get_center(), aligned_edge=RIGHT)
        self.v_group.add(self.corner_neuron_kernel.v_group)
        self.corner_in_original_image_pixel_grid: PixelGridSquare = PixelGridSquare(
            number_of_pixels=7,
            color_array=np.array([[WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
                                  [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
                                  [WHITE, WHITE, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
                                  [WHITE, WHITE, DARK_GRAY, WHITE, WHITE, WHITE, WHITE],
                                  [WHITE, WHITE, DARK_GRAY, WHITE, WHITE, WHITE, WHITE],
                                  [WHITE, WHITE, DARK_GRAY, WHITE, WHITE, WHITE, WHITE],
                                  [WHITE, WHITE, DARK_GRAY, WHITE, WHITE, WHITE, WHITE]]))
        self.corner_in_original_image_pixel_grid.v_group.next_to(self.corner_neuron_kernel.v_group,
                                                                 direction=UP, buff=self.layer_height / 10)
        self.v_group.add(self.corner_in_original_image_pixel_grid.v_group)

    def create_sections(self, scene: Scene, transition_neuron: Circle, transition_kernel: Polygon):
        scene.next_section()
        scene.play(ReplacementTransform(transition_neuron, VGroup(self.dark_to_light_gradient_neuron.neuron, self.dark_to_light_gradient_neuron.indicator_lines)),
                   ReplacementTransform(transition_kernel, self.dark_to_light_gradient_neuron.kernel.v_group))

        scene.next_section()
        scene.play(FadeIn(self.gradient_layer))

        scene.next_section()
        scene.play(FadeIn(self.light_to_dark_gradient_neuron.v_group))

        scene.next_section()
        scene.play(FadeIn(self.gradient_layer_to_line_layer_arrow, self.line_layer))

        scene.next_section()
        scene.play(FadeIn(self.vertical_line_neuron.v_group))

        scene.next_section()
        scene.play(FadeIn(self.vertical_line_in_original_image_pixel_grid.v_group))

        scene.next_section()
        scene.play(FadeOut(self.vertical_line_in_original_image_pixel_grid.v_group))

        scene.next_section()
        scene.play(FadeIn(self.vertical_line_in_original_image_pixel_grid.v_group))

        scene.next_section()
        scene.play(FadeIn(self.horizontal_line_neuron.v_group))

        scene.next_section()
        scene.play(FadeIn(self.line_layer_to_corner_layer_arrow, self.corner_layer))

        scene.next_section()
        scene.play(FadeIn(self.corner_neuron_kernel.v_group))

        scene.next_section()
        scene.play(FadeIn(self.corner_in_original_image_pixel_grid.v_group))

        scene.next_section()
        scene.remove(self.dark_to_light_gradient_neuron.v_group)  # Hacky remove all.
        scene.wait(1)

    def create_layers(self) -> (RoundedRectangle, RoundedRectangle, RoundedRectangle):
        gradient_layer = self.create_layer_rectangle()
        line_layer = self.create_layer_rectangle()
        corner_layer = self.create_layer_rectangle()
        gradient_layer.next_to(line_layer, direction=LEFT, buff=self.layer_margin)
        corner_layer.next_to(line_layer, direction=RIGHT, buff=self.layer_margin)
        return gradient_layer, line_layer, corner_layer

    def create_layer_rectangle(self) -> RoundedRectangle:
        return RoundedRectangle(width=self.layer_width, height=self.layer_height, corner_radius=0.3, stroke_color=BLACK,
                                fill_color=rgba_to_color([224 / 255, 234 / 255, 246 / 255, 1.0]), fill_opacity=1.0, z_index=-2)

