import math
from pathlib import Path
from typing import List

import numpy as np
from manim import Scene, Circle, PINK, Create, config, ImageMobject, RIGHT, Rectangle, FadeIn, FadeOut, BLACK, \
    ReplacementTransform, FadeTransform, DOWN, Dot, ThreeDScene, Sphere, PI, TAU, RED, VGroup, Line3D, IN, Surface, \
    Mobject, OUT, LaggedStart


class ThreeDNeuronsLookingAtPixelsScene(ThreeDScene):
    def construct(self):
        planetary_nebula_image_mobject = ImageMobject(
            Path('neural_network_explanation_presentation_animations/images/grayscale_ngc7293_planetary_nebula.jpg'))
        image_large_size = 6.0
        number_of_pixels = 10
        assert number_of_pixels % 2 == 0
        # 32 seems to be the default?
        line3d_resolution = 5
        sphere_resolution = 5
        xy_rectangle_resolution = (5, 5)
        skip_animations = True
        # Create image.
        self.next_section(skip_animations=skip_animations)
        planetary_nebula_image_mobject.scale_to_fit_width(image_large_size)
        self.add(planetary_nebula_image_mobject)
        # Convert image to grid.
        self.next_section(skip_animations=skip_animations)
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
        # Create all neurons.
        def pixel_index_to_position(x_index, y_index) -> (float, float):
            x_position_ = (-image_large_size / 2) + (pixel_size / 2) + (x_index * pixel_size)
            y_position_ = (image_large_size / 2) - (pixel_size / 2) - (y_index * pixel_size)
            return x_position_, y_position_
        def create_neuron_kernel(x_center, y_center) -> XyRectangle:
            kernel_size = (3 * pixel_size)
            x = x_center - (kernel_size / 2)
            y = y_center - (kernel_size / 2)
            neuron_kernel_ = XyRectangle(x=x, y=y, x_size=kernel_size, y_size=kernel_size,
                                         resolution=xy_rectangle_resolution)
            neuron_kernel_.set_opacity(0.5)
            neuron_kernel_.set_color(RED)
            return neuron_kernel_

        def create_neuron_output(x_center, y_center) -> XyRectangle:
            size = 0.9 * pixel_size
            x = x_center - (size / 2)
            y = y_center - (size / 2)
            neuron_kernel_ = XyRectangle(x=x, y=y, x_size=size, y_size=size, z=2,
                                         resolution=xy_rectangle_resolution)
            neuron_kernel_.set_opacity(0.5)
            neuron_kernel_.set_color(RED)
            return neuron_kernel_

        def create_neuron(x, y):
            neuron_z_position = 1
            neuron_ = Sphere(
                center=(x, y, neuron_z_position),
                radius=((2 * pixel_size) / 5),
                resolution=(sphere_resolution, sphere_resolution),
                sheen_factor=0.0,
                stroke_opacity=0.0,
                fill_opacity=1.0,
                fill_color=RED,
            )
            neuron_.set_color(RED)
            return neuron_

        neurons: List[List[Mobject]] = []
        neuron_kernels: List[List[Mobject]] = []
        neuron_outputs: List[List[Mobject]] = []
        for pixel_index_x in range(1, number_of_pixels-1):
            neurons.append([])
            neuron_kernels.append([])
            neuron_outputs.append([])
            for pixel_index_y in range(1, number_of_pixels-1):
                x_position, y_position = pixel_index_to_position(pixel_index_x, pixel_index_y)
                neurons[pixel_index_x - 1].append(create_neuron(x_position, y_position))
                neuron_kernels[pixel_index_x - 1].append(create_neuron_kernel(x_position, y_position))
                neuron_outputs[pixel_index_x - 1].append(create_neuron_output(x_position, y_position))

        self.next_section(skip_animations=False)
        self.play(FadeIn(neurons[0][0], shift=IN))
        # Show the neuron kernel.
        self.next_section(skip_animations=False)
        self.play(FadeIn(neuron_kernels[0][0], shift=IN))
        # Move to angled view.
        self.next_section(skip_animations=False)
        self.move_camera(1.2 * math.tau / 8, -3 * math.tau / 8)
        # Show the neuron output.
        self.next_section(skip_animations=False)
        self.play(FadeIn(neuron_outputs[0][0], shift=OUT))
        # Add another neuron
        self.next_section(skip_animations=False)
        self.play(FadeIn(neurons[1][0], shift=IN), FadeIn(neuron_kernels[1][0], shift=IN))
        # Add remaining neurons
        self.next_section(skip_animations=False)
        add_neuron_animations = []
        for x in range(len(neurons)):
            for y in range(len(neurons[x])):
                if (x == 0 or x == 1) and y == 0:
                    continue  # Already added neuron and kernel.
                add_neuron_animations.append(FadeIn(neurons[x][y], shift=IN))
                add_neuron_animations.append(FadeIn(neuron_kernels[x][y], shift=IN))
        self.play(LaggedStart(*add_neuron_animations))

        self.next_section(skip_animations=False)

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
    ThreeDNeuronsLookingAtPixelsScene().render(preview=True)
