from pathlib import Path

import numpy as np
from manim import Scene, LaggedStart, ReplacementTransform, FadeIn, IN, FadeOut, VGroup, ImageMobject, RIGHT, Mobject, \
    ApplyFunction, MoveToTarget, DOWN

from neural_network_explanation_presentation_animations.configuration import set_up_configuration
from neural_network_explanation_presentation_animations.isometric_neurons_looking_at_pixels_sub_scene import \
    IsometricNeuronsLookingAtPixelsSubScene
from neural_network_explanation_presentation_animations.layers_building_complexity_sub_scene import \
    LayerBuildingComplexitySubScene
from neural_network_explanation_presentation_animations.mobject_set_operations import mobjects_from_v_group


class MainScene(Scene):
    def __init__(self):
        super().__init__()
        self.isometric_neurons_looking_at_pixels_sub_scene: IsometricNeuronsLookingAtPixelsSubScene = \
            IsometricNeuronsLookingAtPixelsSubScene()
        self.layers_building_complexity_sub_scene: LayerBuildingComplexitySubScene = LayerBuildingComplexitySubScene()
        self.skip_animations = False

    def construct(self):
        background_mobject = ImageMobject(
            Path('neural_network_explanation_presentation_animations/images/background.jpg'))
        background_mobject.scale_to_fit_height(8)
        background_mobject.z_index = -1_000_000
        self.add(background_mobject)
        self.next_section(skip_animations=self.skip_animations)
        self.add(self.isometric_neurons_looking_at_pixels_sub_scene.planetary_nebula_image_mobject)
        self.wait(1)

        self.next_section(skip_animations=self.skip_animations)
        self.play(FadeOut(self.isometric_neurons_looking_at_pixels_sub_scene.planetary_nebula_image_mobject),
                  FadeIn(*self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_pixel_grid))

        self.next_section(skip_animations=self.skip_animations)

        self.play(FadeIn(self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_neuron, shift=IN))

        self.next_section(skip_animations=self.skip_animations)
        self.play(FadeIn(self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_neuron_kernel, shift=IN))

        self.next_section(skip_animations=self.skip_animations)
        coordinate_swap_animations = []
        for flat_grid_index in range(len(self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_pixel_grid)):
            coordinate_swap_animations.append(ReplacementTransform(
                self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_pixel_grid[flat_grid_index],
                self.isometric_neurons_looking_at_pixels_sub_scene.isometric_pixel_grid[flat_grid_index]))
        isometric_neuron = self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][0].neuron
        self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][0].neuron_animation_created = True
        coordinate_swap_animations.append(
            ReplacementTransform(self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_neuron, isometric_neuron))
        isometric_neuron_kernel = self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][0].kernel
        self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][0].kernel_animation_created = True
        coordinate_swap_animations.append(
            ReplacementTransform(self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_neuron_kernel,
                                 isometric_neuron_kernel))
        self.play(*coordinate_swap_animations)

        self.next_section()
        isometric_neuron_copy = isometric_neuron.copy()
        isometric_neuron_kernel_copy = isometric_neuron_kernel.copy()
        self.add(isometric_neuron_copy, isometric_neuron_kernel_copy)
        def shift_and_scale(mobject: Mobject) -> Mobject:
            mobject.scale(0.8)
            mobject.shift(RIGHT * 2)
            return mobject

        # self.play(FadeOut(*in_scene_isometric_sub_scene_mobjects))

        in_scene_isometric_sub_scene_mobjects = list(
            set(mobjects_from_v_group(self.isometric_neurons_looking_at_pixels_sub_scene.v_group)) & set(self.mobjects))
        in_scene_isometric_sub_scene_v_group = VGroup(*in_scene_isometric_sub_scene_mobjects)
        out_scene_isometric_sub_scene_mobjects = list(
            set(mobjects_from_v_group(self.isometric_neurons_looking_at_pixels_sub_scene.v_group)) - set(self.mobjects))
        out_scene_isometric_sub_scene_v_group = VGroup(*out_scene_isometric_sub_scene_mobjects)
        shift_and_scale(out_scene_isometric_sub_scene_v_group)

        self.next_section()
        self.play(
            ApplyFunction(shift_and_scale, in_scene_isometric_sub_scene_v_group),
            ReplacementTransform(
                isometric_neuron_copy,
                VGroup(
                    self.layers_building_complexity_sub_scene.dark_to_light_gradient_neuron.neuron,
                    self.layers_building_complexity_sub_scene.dark_to_light_gradient_neuron.indicator_lines
                )
            ),
            ReplacementTransform(
                isometric_neuron_kernel_copy,
                self.layers_building_complexity_sub_scene.dark_to_light_gradient_neuron.kernel.v_group
            )
        )
        out_scene_isometric_sub_scene_v_group.align_to(in_scene_isometric_sub_scene_v_group, direction=DOWN)

        self.next_section()
        self.play(self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][0].create_output_animation())

        self.next_section()
        self.play(self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][1].create_neuron_animation(),
                  self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][1].create_kernel_animation())

        self.next_section(skip_animations=self.skip_animations)
        section_animations = []
        for y_index in range(len(self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups)):
            for x_index in range(len(self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[y_index])):
                neuron_group = self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[y_index][x_index]
                if not neuron_group.neuron_animation_created:
                    section_animations.append(neuron_group.create_neuron_animation())
                if not neuron_group.kernel_animation_created:
                    section_animations.append(neuron_group.create_kernel_animation())
        self.play(LaggedStart(*section_animations, lag_ratio=0.01))

        self.next_section()
        self.play(FadeIn(self.layers_building_complexity_sub_scene.gradient_layer))

        self.next_section()
        self.play(FadeIn(self.layers_building_complexity_sub_scene.gradient_layer_to_line_layer_arrow))

        self.next_section(skip_animations=self.skip_animations)
        section_animations = []
        for y_index in range(len(self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups)):
            for x_index in range(len(self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[y_index])):
                neuron_group = self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[y_index][x_index]
                if not neuron_group.output_animation_created:
                    section_animations.append(neuron_group.create_output_animation())
        self.play(LaggedStart(*section_animations, lag_ratio=0.01))

        self.next_section(skip_animations=self.skip_animations)
        self.play(FadeOut(self.isometric_neurons_looking_at_pixels_sub_scene.isometric_pixel_grid_v_group,
                          self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups_v_group))

        self.layers_building_complexity_sub_scene.create_later_sections(scene=self)

        self.next_section(skip_animations=self.skip_animations)
        self.wait(1)


if __name__ == '__main__':
    set_up_configuration()
    MainScene().render(preview=True)
