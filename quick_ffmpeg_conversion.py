import subprocess
from pathlib import Path

input_video_root_path = Path('/Users/golmschenk/Desktop/layers_sections')
output_video_root_path = Path('/Users/golmschenk/Desktop/layers_sections_out')
output_video_root_path.mkdir(exist_ok=True, parents=True)
for path in input_video_root_path.glob('*.mov'):
    output_path = output_video_root_path.joinpath(path.name)
    subprocess.run(['ffmpeg', '-y', '-i', str(path), '-c:v', 'prores_ks', str(output_path)])
