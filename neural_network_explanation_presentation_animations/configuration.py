from manim import config, rgb_to_color


def set_up_configuration():
    config.background_color = rgb_to_color([235 / 255, 235 / 255, 235 / 255])
    config.quality = 'low_quality'
    # config.background_opacity = 0
    # config.movie_file_extension = '.mov'
    config.save_sections = True

