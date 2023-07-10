# Import necessary libraries
import argparse
import os
from pathlib import Path
import grpc
from reprint import output
import json
import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc
 
# Define the chunk size for audio processing
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
                language_code=["he-IL"],
            ),
            audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
        ),
        eou_classifier=stt_pb2.EouClassifierOptions(
            default_classifier=stt_pb2.DefaultEouClassifier(
                type=stt_pb2.DefaultEouClassifier.DEFAULT,
                max_pause_between_words_hint_ms=500,  # Adjust this value to increase sensitivity to pauses
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

    results = []

    # Process the server responses and output the result to the console and to the file.
    try:
        with output(initial_len=1) as output_lines:
            for r in it:
                event_type, alternatives = r.WhichOneof("Event"), None
                if event_type == "partial" and len(r.partial.alternatives) > 0:
                    alternatives = [a.text for a in r.partial.alternatives]
                elif event_type == "final":
                    alternatives = [a.text for a in r.final.alternatives]
                    # Save word timestamps to the output file
                    for a in r.final.alternatives:
                        for w in a.words:
                            results.append({
                                "word": w.text,
                                "startMS": w.start_time_ms,
                                "endMS": w.end_time_ms
                            })
                elif event_type == "final_refinement":
                    alternatives = [a.text for a in r.final_refinement.normalized_text.alternatives]
                    output_lines.append("")
                    results.append({
                        "word": alternatives[0],
                        "startMS": None,  # You need to determine these values
                        "endMS": None     # You need to determine these values
                    })
                else:
                    continue
                output_lines[-1] = alternatives[0]

        with open(out_file_name, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
    except grpc.RpcError as err:
        print(f"Error code {err.code()}, message: {err.details()}")
        raise err

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out_path", default="recognizer_output.txt")
    args = parser.parse_args()
    recognize_audio(args.path, args.out_path)