import argparse
import os
from pathlib import Path
import grpc
from reprint import output
import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc
from pydub import AudioSegment
from pydub.utils import make_chunks
import glob
import moviepy.editor as mp

CHUNK_SIZE = 4000
CHUNK_LENGTH_MS = 4*60*1000  # 4 minutes

def extract_audio_from_mov(mov_path, mp3_path):
    clip = mp.VideoFileClip(str(mov_path))
    clip.audio.write_audiofile(mp3_path)

def chunk_audio(mp3_path):
    audio = AudioSegment.from_file(mp3_path, "mp3")
    chunks = make_chunks(audio, CHUNK_LENGTH_MS)
    chunk_files = []

    for i, chunk in enumerate(chunks):
        chunk_name = f"{mp3_path.stem}_chunk{i}.mp3"
        chunk.export(chunk_name, format="mp3")
        chunk_files.append(chunk_name)

    return chunk_files

def read_audio(audio_file_name):
    recognize_options = stt_pb2.StreamingOptions(
        recognition_model=stt_pb2.RecognitionModelOptions(
            audio_format=stt_pb2.AudioFormatOptions(
                container_audio=stt_pb2.ContainerAudio(
                    container_audio_type=stt_pb2.ContainerAudio.MP3,
                )
            ),
            text_normalization=stt_pb2.TextNormalizationOptions(
                text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                profanity_filter=False,
                literature_text=False,
            ),
            language_restriction=stt_pb2.LanguageRestrictionOptions(
                restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                language_code=["en-US"],
            ),
            audio_processing_type=stt_pb2.RecognitionModelOptions.FULL_DATA,
        ),
        eou_classifier=stt_pb2.EouClassifierOptions(
            default_classifier=stt_pb2.DefaultEouClassifier(
                type=stt_pb2.DefaultEouClassifier.DEFAULT,
                max_pause_between_words_hint_ms=2500,
            ),
        ),
    )

    yield stt_pb2.StreamingRequest(session_options=recognize_options)

    with open(audio_file_name, "rb") as f:
        data = f.read(CHUNK_SIZE)
        while data != b"":
            yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))
            data = f.read(CHUNK_SIZE)

def recognize_audio(audio_file_name, out_file_name):
    cred = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel("api.speechkit.cloudil.com:443", cred)
    stub = stt_service_pb2_grpc.RecognizerStub(channel)

    api_key = os.environ["SPEECHKIT_API_KEY"]

    it = stub.RecognizeStreaming(
        read_audio(audio_file_name), metadata=(("authorization", f"Api-Key {api_key}"),)
    )

    try:
        with output(initial_len=1) as output_lines:
            for r in it:
                event_type, alternatives = r.WhichOneof("Event"), None
                if event_type == "partial" and len(r.partial.alternatives) > 0:
                    alternatives = [a.text for a in r.partial.alternatives]
                elif event_type == "final":
                    alternatives = [a.text for a in r.final.alternatives]
                    with open(out_file_name, "a") as f:
                        for a in r.final.alternatives:
                            f.write(f"{a.text} {a.start_time_ms} {a.end_time_ms}\n")
                elif event_type == "final_refinement":
                    alternatives = [a.text for a in r.final_refinement.normalized_text.alternatives]
                    output_lines.append("")
                    with open(out_file_name, "a") as f:
                        f.write(alternatives[0])
                else:
                    continue
                output_lines[-1] = alternatives[0]

    except grpc._channel._Rendezvous as err:
        print(f"Error code {err._state.code}, message: {err._state.details}")
        raise err

def process_directory(input_dir, output_dir):
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)

    if not output_dir_path.is_dir():
        output_dir_path.mkdir()

    for video_file in input_dir_path.glob("*.mov"):
        mp3_file = output_dir_path / (video_file.stem + ".mp3")
        txt_file = output_dir_path / (video_file.stem + ".txt")

        extract_audio_from_mov(video_file, mp3_file)

        chunk_files = chunk_audio(mp3_file)

        for chunk_file in chunk_files:
            recognize_audio(chunk_file, txt_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
