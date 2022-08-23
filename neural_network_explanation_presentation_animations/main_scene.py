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
        planetary_nebula_image_mobject = self.isometric_neurons_looking_at_pixels_sub_scene.create_image()
        self.add(planetary_nebula_image_mobject)

        self.next_section(skip_animations=self.skip_animations)
        cartesian_pixel_grid, isometric_pixel_grid, neuron_groups = \
            self.isometric_neurons_looking_at_pixels_sub_scene.create_grids_and_neuron_groups()
        self.play(FadeOut(planetary_nebula_image_mobject), FadeIn(*cartesian_pixel_grid))

        self.next_section(skip_animations=self.skip_animations)
        cartesian_neuron, cartesian_neuron_kernel = \
            self.isometric_neurons_looking_at_pixels_sub_scene.create_cartesian_neuron_collection()
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
        section_animations = []
        for y_index in range(len(neuron_groups)):
            for x_index in range(len(neuron_groups[y_index])):
                neuron_group = neuron_groups[y_index][x_index]
                if not neuron_group.output_animation_created:
                    section_animations.append(neuron_group.create_output_animation())
        self.play(LaggedStart(*section_animations, lag_ratio=0.01))

        self.next_section(skip_animations=self.skip_animations)
        self.wait(1)


if __name__ == '__main__':
    set_up_configuration()
    MainScene().render(preview=True)