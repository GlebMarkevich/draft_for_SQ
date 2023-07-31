import io
from pathlib import Path
import re
import grpc
import argparse
import os

import yandex.cloud.ai.tts.v3.tts_pb2 as tts_pb2
import yandex.cloud.ai.tts.v3.tts_service_pb2_grpc as tts_service_pb2_grpc

TEXT_LENGTH_LIMIT = 240


def synthesize_batch(text):
    api_key = os.environ["SPEECHKIT_API_KEY"]

    # Define request parameters.
    request = tts_pb2.UtteranceSynthesisRequest(
        text=text,
        # Set output audio format
        output_audio_spec=tts_pb2.AudioFormatOptions(
            container_audio=tts_pb2.ContainerAudio(
                container_audio_type=tts_pb2.ContainerAudio.MP3
            )
        ),
        loudness_normalization_type=tts_pb2.UtteranceSynthesisRequest.LUFS,
        # Set voice. Full list here: https://cloudil.co.il/docs/speechkit/tts/voices
        hints=[tts_pb2.Hints(voice="john")],
    )

    # Establish connection with server.
    cred = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel("api.speechkit.cloudil.com:443", cred)
    stub = tts_service_pb2_grpc.SynthesizerStub(channel)

    # Send data for synthesis.
    it = stub.UtteranceSynthesis(
        request, metadata=(("authorization", f"Api-Key {api_key}"),)
    )

    # Merge chunks into BytesIO buffer chunks.
    try:
        audio = io.BytesIO()
        for response in it:
            audio.write(response.audio_chunk.data)
        audio.seek(0)
        return audio
    except grpc._channel._Rendezvous as err:
        print(f"Error code {err._state.code}, message: {err._state.details}")
        raise err


def split_batches(text):
    sentences = re.split(r"(?<=[\.\!\?])", text)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= TEXT_LENGTH_LIMIT:
            yield sent
        else:
            words = sent.split()
            batch = ""
            for word in words:
                if len(batch + word) > TEXT_LENGTH_LIMIT and batch:
                    yield batch
                    batch = word
                else:
                    batch += " " + word
            if batch:
                yield batch


def synthesize(text):
    merged_bytes = io.BytesIO()
    for batch in split_batches(text):
        print(batch)
        batch_bytes = synthesize_batch(batch)
        merged_bytes.write(batch_bytes.read())
    merged_bytes.seek(0)
    return merged_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file")
    parser.add_argument("--output", default="output.mp3")
    args = parser.parse_args()

    with open(args.text_file) as f:
        input_text = f.read()
    audio_bytes = synthesize(input_text)
    Path(args.output).write_bytes(audio_bytes.getbuffer())
