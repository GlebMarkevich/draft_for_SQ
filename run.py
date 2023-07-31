import argparse
from pathlib import Path
from extract_audio import extract_audio

from recognize_audio import recognize_audio
from summarize import summarize
from text_to_speech import synthesize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    args = parser.parse_args()

    video_path = Path(args.video_path)
    audio_path = video_path.with_suffix('.mp3')
    text_path = video_path.with_suffix('.txt')
    summary_path = video_path.with_suffix('.summary.mp3')
    try:
        extract_audio(args.video_path, out_path=audio_path)
    except ValueError:
        print(f"{audio_path} exists, using existing file.")

    print("Speech recognition...")
    try:
        recognize_audio(audio_path, text_path)
        print("Speech recognition finished.")
    except ValueError:
        print(f"{text_path} exists, using existing file.")

    print("Summarizing...")
    result = summarize(text_path)
    print(result)

    print("Running speech synthesis...")
    audio_bytes = synthesize(result)
    summary_path.write_bytes(audio_bytes.getbuffer())
