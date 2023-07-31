import argparse
from pathlib import Path
from moviepy.editor import VideoFileClip

def extract_audio(video_path, out_path=None):

    # If user does not provide path for audio file, use video path with changed extension
    if out_path is None:
        out_path = Path(video_path).with_suffix('.mp3')

    # If desired audio file already exists, raise exception
    if Path(out_path).is_file():
        raise ValueError(f'File {out_path} already exists')

    video_obj = VideoFileClip(video_path)
    video_obj.audio.write_audiofile(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('--out', required=False)
    args = parser.parse_args()
    extract_audio(args.video_path, args.out)