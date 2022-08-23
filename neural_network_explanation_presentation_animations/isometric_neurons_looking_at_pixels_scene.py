import math
from pathlib import Path
from typing import List, Union

import numpy as np
from manim import Scene, config, ImageMobject, FadeTransform, Mobject, Polygon, FadeOut, FadeIn, ReplacementTransform, \
    BLACK, Circle, RED, DOWN, IN, Animation, LaggedStart


class IsometricNeuronsLookingAtPixelsScene(Scene):
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
        super().__init__()

    def construct(self):
        self.next_section(skip_animations=self.skip_animations)
        planetary_nebula_image_mobject = ImageMobject(
            Path('neural_network_explanation_presentation_animations/images/grayscale_ngc7293_planetary_nebula.jpg'))
        planetary_nebula_image_mobject.scale_to_fit_width(self.image_large_size)
        self.add(planetary_nebula_image_mobject)

        self.next_section(skip_animations=self.skip_animations)
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
                cartesian_pixel_grid.append(self.create_pixel_grid_polygon(polygon_cartesian_positions))
                isometric_pixel_grid.append(self.create_pixel_grid_polygon(polygon_isometric_positions))
                if (pixel_x_index != 0 and pixel_x_index != self.number_of_pixels - 1 and
                        pixel_y_index != 0 and pixel_y_index != self.number_of_pixels - 1):
                    neuron_group = NeuronGroup(self, pixel_x_index, pixel_y_index)
                    neuron_groups[pixel_y_index - 1].append(neuron_group)
        self.play(FadeOut(planetary_nebula_image_mobject), FadeIn(*cartesian_pixel_grid))

        self.next_section(skip_animations=self.skip_animations)
        cartesian_neuron_position = self.pixel_index_to_pixel_center_cartesian_xy(pixel_x_index=1, pixel_y_index=1)
        cartesian_neuron = self.create_neuron([cartesian_neuron_position[0], cartesian_neuron_position[1], 1])
        cartesian_neuron_kernel = self.create_neuron_kernel_centered_on_pixel_index(pixel_x_index=1, pixel_y_index=1,
                                                                                    isometric=False)
        self.play(FadeIn(cartesian_neuron, shift=IN))

        self.next_section(skip_animations=self.skip_animations)
        self.play(FadeIn(cartesian_neuron_kernel, shift=IN))

        self.next_section(skip_animations=self.skip_animations)
        coordinate_swap_animations = []
        for flat_grid_index in range(len(cartesian_pixel_grid)):
            coordinate_swap_animations.append(ReplacementTransform(cartesian_pixel_grid[flat_grid_index],
                                                                   isometric_pixel_grid[flat_grid_index]))
        isometric_neuron = neuron_groups[0][0].neuron
        neuron_groups[0][0].neuron_animation_created = True
        coordinate_swap_animations.append(ReplacementTransform(cartesian_neuron, isometric_neuron))
        isometric_neuron_kernel = neuron_groups[0][0].kernel
        neuron_groups[0][0].kernel_animation_created = True
        coordinate_swap_animations.append(ReplacementTransform(cartesian_neuron_kernel, isometric_neuron_kernel))
        self.play(*coordinate_swap_animations)

        self.next_section(skip_animations=self.skip_animations)
        neuron_animation = neuron_groups[0][1].create_neuron_animation()
        kernel_animation = neuron_groups[0][1].create_kernel_animation()
        self.play(neuron_animation)
        self.play(kernel_animation)

        self.next_section(skip_animations=self.skip_animations)
        section_animations = []
        for y_index in range(len(neuron_groups)):
            for x_index in range(len(neuron_groups[y_index])):
                neuron_group = neuron_groups[y_index][x_index]
                if not neuron_group.neuron_animation_created:
                    section_animations.append(neuron_group.create_neuron_animation())
                if not neuron_group.kernel_animation_created:
                    section_animations.append(neuron_group.create_kernel_animation())
        self.play(LaggedStart(*section_animations, lag_ratio=0.02))

        self.next_section(skip_animations=self.skip_animations)
        self.wait(1)

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

    def create_pixel_grid_polygon(self, vertexes: Union[np.ndarray, List[List[float]]]) -> Polygon:
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
        polygon_positions = np.array([
            [left, top, 0.01],
            [right, top, 0.01],
            [right, bottom, 0.01],
            [left, bottom, 0.01],
        ])
        if isometric:
            polygon_positions = self.from_cartesian_position_to_isometric_position(polygon_positions)
        polygon = Polygon(*polygon_positions, color=RED, fill_color=RED, fill_opacity=0.5, stroke_opacity=0.0)
        return polygon


class NeuronGroup:
    def __init__(self, scene: IsometricNeuronsLookingAtPixelsScene, pixel_x_index: float, pixel_y_index: float):
        self.neuron: Circle = scene.create_neuron_above_pixel_position(pixel_x_index, pixel_y_index)
        self.neuron_animation_created: bool = False
        self.kernel: Polygon = scene.create_neuron_kernel_centered_on_pixel_index(pixel_x_index, pixel_y_index)
        self.kernel_animation_created: bool = False

    def create_neuron_animation(self) -> Animation:
        self.neuron_animation_created = True
        return FadeIn(self.neuron, shift=DOWN)

    def create_kernel_animation(self) -> Animation:
        self.kernel_animation_created = True
        return FadeIn(self.kernel, shift=DOWN)



if __name__ == '__main__':
    config.background_opacity = 0
    config.movie_file_extension = '.mov'
    config.save_sections = True
    config.quality = 'low_quality'
    IsometricNeuronsLookingAtPixelsScene().render(preview=True)
