import math
from pathlib import Path

import numpy as np
from manim import Scene, Circle, PINK, Create, config, ImageMobject, RIGHT, Rectangle, FadeIn, FadeOut, BLACK, \
    ReplacementTransform, FadeTransform, DOWN, Dot, ThreeDScene, Sphere, PI, TAU, RED, VGroup, Line3D, IN, Surface


class MainScene(ThreeDScene):
    def construct(self):
        planetary_nebula_image_mobject = ImageMobject(
            Path('neural_network_explanation_presentation_animations/images/grayscale_ngc7293_planetary_nebula.jpg'))
        image_large_size = 6.0
        number_of_pixels = 10
        assert number_of_pixels % 2 == 0
        # 32 seems to be the default?
        line3d_resolution = 10
        sphere_resolution = 10
        xy_rectangle_resolution = (10, 10)
        skip_animations = True
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
        def create_neuron_kernel(x_center, y_center) -> XyRectangle:
            kernel_size = (3 * pixel_size)
            x = x_center - (kernel_size / 2)
            y = y_center - (kernel_size / 2)
            neuron_kernel = XyRectangle(x=x, y=y, x_size=kernel_size, y_size=kernel_size,
                                        resolution=xy_rectangle_resolution)
            neuron_kernel.set_opacity(0.5)
            neuron_kernel.set_color(RED)
            return neuron_kernel

        def create_neuron(x, y):
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
            return neuron

        neuron_x_position = (-image_large_size / 2) + (3 * pixel_size / 2)
        neuron_y_position = (image_large_size / 2) - (3 * pixel_size / 2)
        neuron = create_neuron(neuron_x_position, neuron_y_position)
        neuron_kernel = create_neuron_kernel(neuron_x_position, neuron_y_position)
        self.play(FadeIn(neuron, shift=IN), FadeIn(neuron_kernel, shift=IN))
        self.next_section(skip_animations=False)
        # Move to angled view.
        self.move_camera(1.2 * math.tau / 8, -3 * math.tau / 8)
        self.next_section(skip_animations=skip_animations)

        self.wait(1)


class XyRectangle(Surface):
    def align_points_with_larger(self, larger_mobject):
        super().align_points_with_larger(larger_mobject)

    def __init__(self, x: float, y: float, x_size: float, y_size: float, z: float = 0.1, resolution=(10, 10)):
        super().__init__(
            func=lambda u, v: np.array([x + (u * x_size), y + (v * y_size), z]),
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=resolution,
        )


if __name__ == '__main__':
    config.background_opacity = 0
    config.movie_file_extension = '.mov'
    config.save_sections = True
    config.quality = 'low_quality'
    MainScene().render(preview=True)
