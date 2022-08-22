import math
from pathlib import Path

import numpy as np
from manim import Scene, Circle, PINK, Create, config, ImageMobject, RIGHT, Rectangle, FadeIn, FadeOut, BLACK, \
    ReplacementTransform, FadeTransform, DOWN, Dot, ThreeDScene, Sphere, PI, TAU, RED, VGroup, Line3D, IN


class MainScene(ThreeDScene):
    def construct(self):
        planetary_nebula_image_mobject = ImageMobject(
            Path('neural_network_explanation_presentation_animations/images/grayscale_ngc7293_planetary_nebula.jpg'))
        image_large_size = 6.0
        number_of_pixels = 10
        assert number_of_pixels % 2 == 0
        # 32 seems to be the default?
        line3d_resolution = 10
        sphere_resolution = 50
        skip_animations = False
        planetary_nebula_image_mobject.scale_to_fit_width(image_large_size)
        self.add(planetary_nebula_image_mobject)
        self.next_section(skip_animations=skip_animations)
        # Convert image to grid.
        pixel_size = image_large_size / number_of_pixels
        pixel_grid = VGroup()
        for x_position in np.linspace(-image_large_size / 2, image_large_size / 2, num=number_of_pixels + 1):
            line = Line3D([x_position, -image_large_size / 2, 0], [x_position, image_large_size / 2, 0], color=BLACK,
                          resolution=line3d_resolution)
            pixel_grid.add(line)
        for y_position in np.linspace(-image_large_size / 2, image_large_size / 2, num=number_of_pixels + 1):
            line = Line3D([-image_large_size / 2, y_position, 0], [image_large_size / 2, y_position, 0], color=BLACK,
                          resolution=line3d_resolution)
            pixel_grid.add(line)
        self.add(pixel_grid)
        self.play(FadeOut(planetary_nebula_image_mobject), FadeIn(pixel_grid))
        self.next_section(skip_animations=skip_animations)
        # Add a neuron.
        neuron_x_position = (-image_large_size / 2) + (3 * pixel_size / 2)
        neuron_y_position = (image_large_size / 2) - (3 * pixel_size / 2)
        neuron_z_position = 1
        neuron = Sphere(
            center=(neuron_x_position, neuron_y_position, neuron_z_position),
            radius=pixel_size / 2,
            resolution=(sphere_resolution, sphere_resolution),
            sheen_factor=0.0,
            stroke_opacity=0.0,
            fill_opacity=1.0,
            fill_color=RED,
        )
        neuron.set_color(RED)
        self.play(FadeIn(neuron, shift=IN))
        self.next_section(skip_animations=skip_animations)
        # Move to angled view.
        self.move_camera(1.2 * math.tau / 8, -3 * math.tau / 8)
        self.next_section(skip_animations=skip_animations)

        self.wait(1)


if __name__ == '__main__':
    config.background_opacity = 0
    config.movie_file_extension = '.mov'
    config.save_sections = True
    config.quality = 'low_quality'
    MainScene().render(preview=True)
