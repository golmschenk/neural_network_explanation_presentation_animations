from manim import Scene, LaggedStart, ReplacementTransform, FadeIn, IN, FadeOut

from neural_network_explanation_presentation_animations.configuration import set_up_configuration
from neural_network_explanation_presentation_animations.isometric_neurons_looking_at_pixels_sub_scene import \
    IsometricNeuronsLookingAtPixelsSubScene
from neural_network_explanation_presentation_animations.layers_building_complexity_sub_scene import \
    LayerBuildingComplexitySubScene


class MainScene(Scene):
    def __init__(self):
        super().__init__()
        self.isometric_neurons_looking_at_pixels_sub_scene: IsometricNeuronsLookingAtPixelsSubScene = \
            IsometricNeuronsLookingAtPixelsSubScene()
        self.layers_building_complexity_sub_scene: LayerBuildingComplexitySubScene = LayerBuildingComplexitySubScene()

    def construct(self):
        self.next_section(skip_animations=self.skip_animations)
        self.add(self.isometric_neurons_looking_at_pixels_sub_scene.planetary_nebula_image_mobject)

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
            coordinate_swap_animations.append(ReplacementTransform(self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_pixel_grid[flat_grid_index],
                                                                   self.isometric_neurons_looking_at_pixels_sub_scene.isometric_pixel_grid[flat_grid_index]))
        isometric_neuron = self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][0].neuron
        self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][0].neuron_animation_created = True
        coordinate_swap_animations.append(ReplacementTransform(self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_neuron, isometric_neuron))
        isometric_neuron_kernel = self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][0].kernel
        self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][0].kernel_animation_created = True
        coordinate_swap_animations.append(ReplacementTransform(self.isometric_neurons_looking_at_pixels_sub_scene.cartesian_neuron_kernel, isometric_neuron_kernel))
        self.play(*coordinate_swap_animations)

        self.next_section()
        isometric_neuron_copy = isometric_neuron.copy()
        isometric_neuron_kernel_copy = isometric_neuron_kernel.copy()
        self.add(isometric_neuron_copy, isometric_neuron_kernel_copy)
        self.remove(isometric_neuron, isometric_neuron_kernel)
        self.play(FadeOut(self.isometric_neurons_looking_at_pixels_sub_scene.isometric_pixel_grid_v_group))

        self.layers_building_complexity_sub_scene.create_sections(scene=self, transition_neuron=isometric_neuron_copy,
                                                                  transition_kernel=isometric_neuron_kernel_copy)


        self.next_section(skip_animations=self.skip_animations)
        # Hacky go back to old scene.
        self.add(isometric_neuron, isometric_neuron_kernel)
        self.add(self.isometric_neurons_looking_at_pixels_sub_scene.isometric_pixel_grid_v_group)
        neuron_animation = self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][1].create_neuron_animation()
        kernel_animation = self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[0][1].create_kernel_animation()
        self.play(neuron_animation)
        self.play(kernel_animation)

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

        self.next_section(skip_animations=self.skip_animations)
        section_animations = []
        for y_index in range(len(self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups)):
            for x_index in range(len(self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[y_index])):
                neuron_group = self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups[y_index][x_index]
                if not neuron_group.output_animation_created:
                    section_animations.append(neuron_group.create_output_animation())
        self.play(LaggedStart(*section_animations, lag_ratio=0.01))

        self.play(FadeOut(self.isometric_neurons_looking_at_pixels_sub_scene.isometric_pixel_grid_v_group,
                          self.isometric_neurons_looking_at_pixels_sub_scene.neuron_groups_v_group))

        self.next_section(skip_animations=self.skip_animations)
        self.wait(1)


if __name__ == '__main__':
    set_up_configuration()
    MainScene().render(preview=True)