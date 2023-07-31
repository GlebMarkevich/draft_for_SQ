import argparse
import os
from pathlib import Path

import grpc
from reprint import output

import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc

CHUNK_SIZE = 4000


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
                language_code=["en-US"],
            ),
            audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
        )
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
    if Path(out_file_name).is_file():
        raise ValueError(f"{out_file_name} exists.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("out_path", default="recognizer_output.txt")
    args = parser.parse_args()
    recognize_audio(args.path, args.out_path)
