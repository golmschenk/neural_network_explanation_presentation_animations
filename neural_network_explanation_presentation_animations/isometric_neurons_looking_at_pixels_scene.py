import math
from pathlib import Path
from typing import List, Union

import numpy as np
from manim import Scene, config, ImageMobject, FadeTransform, Mobject, Polygon, FadeOut, FadeIn, ReplacementTransform, \
    BLACK


class ThreeDNeuronsLookingAtPixelsScene(Scene):
    def __init__(self):
        self.image_large_size = 6.0
        self.number_of_pixels = 10
        assert self.number_of_pixels % 2 == 0
        self.pixel_size = self.image_large_size / self.number_of_pixels
        self.skip_animations = False
        self.isometric_theta = math.tau / 8
        self.isometric_scale = 0.5
        self.isometric_shift = [0, -1, 0]
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
        for pixel_x_index in range(self.number_of_pixels):
            for pixel_y_index in range(self.number_of_pixels):
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
        self.play(FadeOut(planetary_nebula_image_mobject), FadeIn(*cartesian_pixel_grid))

        self.next_section(skip_animations=self.skip_animations)
        grid_swap_animations = []
        for flat_grid_index in range(len(cartesian_pixel_grid)):
            grid_swap_animations.append(ReplacementTransform(cartesian_pixel_grid[flat_grid_index],
                                                             isometric_pixel_grid[flat_grid_index]))
        self.play(*grid_swap_animations)
        self.wait(1)

        self.next_section(skip_animations=self.skip_animations)
        self.wait(1)

    def pixel_index_to_pixel_center_cartesian_xy(self, pixel_x_index, pixel_y_index) -> (float, float):
        cartesian_x_position = (-self.image_large_size / 2) + (self.pixel_size / 2) + (pixel_x_index * self.pixel_size)
        cartesian_y_position = (self.image_large_size / 2) - (self.pixel_size / 2) - (pixel_y_index * self.pixel_size)
        return cartesian_x_position, cartesian_y_position

    def pixel_index_to_pixel_start_cartesian_xy(self, pixel_x_index, pixel_y_index) -> (float, float):
        cartesian_x_position = (-self.image_large_size / 2) + (pixel_x_index * self.pixel_size)
        cartesian_y_position = (self.image_large_size / 2) - (pixel_y_index * self.pixel_size)
        return cartesian_x_position, cartesian_y_position

    def pixel_index_to_pixel_start_isometric_position(self, pixel_x_index, pixel_y_index) -> np.ndarray:
        cartesian_x_position = (-self.image_large_size / 2) + (pixel_x_index * self.pixel_size)
        cartesian_y_position = (self.image_large_size / 2) - (pixel_y_index * self.pixel_size)
        isometric_position = self.from_cartesian_position_to_isometric_position(
            np.array([cartesian_x_position, cartesian_y_position, 0]))
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
            non_shifted_isometric_position = np.stack([non_shifted_isometric_xy_position,
                                                       cartesian_z_position], axis=0)
        else:
            non_shifted_isometric_xy_position = np.stack([rotated_position[:, 0],
                                                          rotated_position[:, 1] * self.isometric_scale], axis=1)
            non_shifted_isometric_position = np.concatenate([non_shifted_isometric_xy_position,
                                                             cartesian_z_position], axis=1)
        isometric_position = non_shifted_isometric_position + self.isometric_shift
        return isometric_position

    def create_pixel_grid_polygon(self, vertexes: Union[np.ndarray, List[List[float]]]) -> Polygon:
        polygon = Polygon(*vertexes, color=BLACK)
        return polygon


if __name__ == '__main__':
    config.background_opacity = 0
    config.movie_file_extension = '.mov'
    config.save_sections = True
    config.quality = 'low_quality'
    ThreeDNeuronsLookingAtPixelsScene().render(preview=True)
