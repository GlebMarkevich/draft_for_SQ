import argparse
import os
import pyaudio
import pkg_resources
import subprocess
import sys
import grpc
from reprint import output

import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ["pyaudio", "grpcio", "reprint"]
for package in required_packages:
    try:
        dist = pkg_resources.get_distribution(package)
        print('{} ({}) is installed'.format(dist.key, dist.version))
    except pkg_resources.DistributionNotFound:
        print('{} is NOT installed'.format(package))
        install(package)

def generate_audio_stream():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Start speaking...")

    recognize_options = stt_pb2.StreamingOptions(
        recognition_model=stt_pb2.RecognitionModelOptions(
            audio_format=stt_pb2.AudioFormatOptions(
                pcm_audio=stt_pb2.PcmAudio(
                    sample_rate_hertz=RATE,
                    num_channels=CHANNELS,
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
            audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
        )
    )

    yield stt_pb2.StreamingRequest(session_options=recognize_options)

    try:
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK)
                print(f"Data Chunk: {data}") # log data
                yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))
            except IOError as e:
                print(f"IOError during read: {e}")
                pass
    except Exception as e:
        print(f"Exception during audio stream generation: {e}")
        stream.stop_stream()
        stream.close()
        p.terminate()
        raise e
    
    print("* Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

def recognize_audio_mic(out_file_name):
    if os.path.isfile(out_file_name):
        raise ValueError(f"{out_file_name} exists.")
        
    cred = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel('api.speechkit.cloudil.com:443', cred)
    stub = stt_service_pb2_grpc.RecognizerStub(channel)

    api_key = os.environ['SPEECHKIT_API_KEY']

    it = stub.RecognizeStreaming(generate_audio_stream(), metadata=(('authorization', f'Api-Key {api_key}'),))
    
    try:
        with output(initial_len=1) as output_lines:
            for r in it:
                event_type, alternatives = r.WhichOneof("Event"), None
                if event_type == 'partial' and len(r.partial.alternatives) > 0:
                    alternatives = [a.text for a in r.partial.alternatives]
                elif event_type == 'final':
                    alternatives = [a.text for a in r.final.alternatives]
                elif event_type == 'final_refinement':
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", default="recognizer_output.txt")
    args = parser.parse_args()
    recognize_audio_mic(args.out_path)


