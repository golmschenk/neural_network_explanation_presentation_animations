import math
from pathlib import Path

import numpy as np
from manim import Scene, Circle, PINK, Create, config, ImageMobject, RIGHT, Rectangle, FadeIn, FadeOut, BLACK, \
    ReplacementTransform, FadeTransform, DOWN, Dot


class MainScene(Scene):
    def construct(self):
        planetary_nebula_image_mobject = ImageMobject(
            Path('neural_network_explanation_presentation_animations/images/grayscale_ngc7293_planetary_nebula.jpg'))
        image_large_size = 6.0
        number_of_pixels = 10
        planetary_nebula_image_mobject.scale_to_fit_width(image_large_size)
        self.add(planetary_nebula_image_mobject)
        self.next_section()
        # Convert image to grid.
        pixel_size = image_large_size / number_of_pixels
        pixel_grid = Rectangle(width=image_large_size, height=image_large_size, grid_xstep=pixel_size,
                               grid_ystep=pixel_size)
        pixel_grid.stroke_color = BLACK
        self.play(FadeTransform(planetary_nebula_image_mobject, pixel_grid))
        self.next_section()
        # Convert to an isometric grid.
        isometric_pixel_grid = pixel_grid.copy()
        isometric_pixel_grid.rotate(math.tau / 8)
        isometric_pixel_grid.stretch(0.5, 1)
        isometric_pixel_grid.shift(DOWN)
        self.play(ReplacementTransform(pixel_grid, isometric_pixel_grid))
        self.next_section()
        # Show neuron.
        def cartesian_coordinate_to_isometric_coordinate(cartesian_x: float, cartesian_y: float) -> (float, float):
            vector = np.array([cartesian_x, cartesian_y])
            theta = math.tau / 8
            scale = 0.5
            rotation = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rotated_vector = vector.dot(rotation)
            isometric_vector = np.array([rotated_vector[0], rotated_vector[1] * scale])
            return isometric_vector[0], isometric_vector[1]

        cartesian_coordinates = ((-image_large_size / 2) + (pixel_size * 0.5),
                                 (image_large_size / 2) + (pixel_size * 0.5))
        isometric_coordinates = cartesian_coordinate_to_isometric_coordinate(*cartesian_coordinates)
        # self.add(Dot([*isometric_coordinates]))

        self.wait(1)


if __name__ == '__main__':
    config.background_opacity = 0
    config.movie_file_extension = '.mov'
    config.quality = 'low_quality'
    MainScene().render(preview=True)
