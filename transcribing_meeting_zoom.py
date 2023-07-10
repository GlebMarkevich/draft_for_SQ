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

CHUNK_SIZE = 4000
CHUNK_LENGTH_MS = 4*60*1000  # 4 minutes

def convert_m4a_to_mp3(m4a_path, mp3_path):
    audio = AudioSegment.from_file(m4a_path, "m4a")
    audio.export(mp3_path, format="mp3")

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
    # Specify the recognition settings.
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
                language_code=["he-IL"],
            ),

            audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
        ),

        eou_classifier=stt_pb2.EouClassifierOptions(
            default_classifier=stt_pb2.DefaultEouClassifier(
                type=stt_pb2.DefaultEouClassifier.DEFAULT,
                max_pause_between_words_hint_ms=2500,
            ),
        ),
    )

    # Send a message with recognition settings.
    yield stt_pb2.StreamingRequest(session_options=recognize_options)
    
    # Read the audio file and send its contents in portions.
    with open(audio_file_name, "rb") as f:
        data = f.read(CHUNK_SIZE)
        while data != b"":
            yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))
            data = f.read(CHUNK_SIZE)

def recognize_audio(audio_file_name, out_file_name):
    # Establish a connection with the server.
    cred = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel("api.speechkit.cloudil.com:443", cred)
    stub = stt_service_pb2_grpc.RecognizerStub(channel)

    api_key = os.environ["SPEECHKIT_API_KEY"]

    # Send data for recognition.
    it = stub.RecognizeStreaming(
        read_audio(audio_file_name), metadata=(("authorization", f"Api-Key {api_key}"),)
    )

    # Process the server responses and output the result to the console and to the file.
    try:
        with output(initial_len=1) as output_lines:
            for r in it:
                event_type, alternatives = r.WhichOneof("Event"), None
                if event_type == "partial" and len(r.partial.alternatives) > 0:
                    alternatives = [a.text for a in r.partial.alternatives]
                elif event_type == "final":
                    alternatives = [a.text for a in r.final.alternatives]
                    # Save phrase timestamps to the output file
                    with open(out_file_name, "a") as f:
                        for a in r.final.alternatives:
                            # Writing the phrase and its start and end times
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

    for audio_file in input_dir_path.glob("*.m4a"):
        mp3_file = output_dir_path / (audio_file.stem + ".mp3")
        txt_file = output_dir_path / (audio_file.stem + ".txt")
        
        convert_m4a_to_mp3(audio_file, mp3_file)

        chunk_files = chunk_audio(mp3_file)

        for chunk_file in chunk_files:
            recognize_audio(chunk_file, txt_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)