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

# Define the chunk size for reading the audio file
CHUNK_SIZE = 4000

# Function to convert and chunk audio from m4a to mp3 format
def convert_and_chunk_audio(m4a_path, mp3_path):
    # Load audio file
    audio = AudioSegment.from_file(m4a_path, "m4a")

    # Define chunk length in milliseconds
    chunk_length_ms = 4.5 * 60 * 1000  # 4.5 minutes in milliseconds
    chunks = make_chunks(audio, chunk_length_ms)  # Make chunks of 4.5 mins

    # Export all of the individual chunks as .mp3 files
    for i, chunk in enumerate(chunks):
        chunk_name = f"{mp3_path.stem}_{i}.mp3"
        chunk.export(chunk_name, format="mp3")
        yield chunk_name

# Function to read audio file and generate streaming requests for recognition
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
            audio_processing_type=stt_pb2.RecognitionModelOptions.FULL_DATA,
        ),
        eou_classifier=stt_pb2.EouClassifierOptions(
            default_classifier=stt_pb2.DefaultEouClassifier(
                type=stt_pb2.DefaultEouClassifier.DEFAULT,
                max_pause_between_words_hint_ms=2500,  # Adjust this value to increase sensitivity to pauses
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

# Function to recognize audio and store the transcription
def recognize_audio(audio_file_name, out_file_name):
    # Establish a connection with the server.
    cred = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel("api.speechkit.cloudil.com:443", cred)
    stub = stt_service_pb2_grpc.RecognizerStub(channel)

    # Get the API key from environment variable
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
                    # Append phrase timestamps to the output file
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

# Function to process all audio files in a directory
def process_directory(input_dir, output_dir):
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)

    # Check if output directory exists, if not, create it
    if not output_dir_path.is_dir():
        output_dir_path.mkdir()

    # Process all .m4a audio files in the input directory
    for audio_file in input_dir_path.glob("*.m4a"):
        mp3_file = output_dir_path / (audio_file.stem + ".mp3")
        txt_file = output_dir_path / (audio_file.stem + ".txt")

        # Instead of just converting and recognizing, we now chunk the audio as well
        for chunk_name in convert_and_chunk_audio(audio_file, mp3_file):
            recognize_audio(chunk_name, txt_file)

# Function to get speaker name from file path
def get_speaker(file_path):
    """
    Extract the speaker name from the file path. 
    This assumes that the speaker name is contained in the filename.
    """
    filename = Path(file_path).stem  # Get the filename without extension
    speaker = filename.split("_")[-1]  # Assumes speaker is the last part of filename after splitting by "_"
    return speaker

# Function to split transcriptions from a single file into separate entries
def split_transcriptions(file_path):
    """
    Split transcriptions from a single file into separate entries,
    maintaining the order of the timestamps.
    """
    transcriptions = []
    final_lines = []
    with open(file_path, "r", encoding='utf-8-sig') as file:
        for line in file:
            line_parts = line.strip().split()
            time_stamp_segments = line_parts[-2:]

            # Check if the last two parts can be timestamps
            if len(time_stamp_segments) == 2 and all(ts.isdigit() for ts in time_stamp_segments):
                time_stamp1, time_stamp2 = map(int, time_stamp_segments)
                text = " ".join(line_parts[:-2])
                transcriptions.append((get_speaker(file_path), text, time_stamp1, time_stamp2))
            else:
                final_lines.append((get_speaker(file_path), line.strip()))

    return transcriptions, final_lines

# Function to merge all transcriptions into a single file
def merge_files(output_dir):
    txt_files = glob.glob(os.path.join(output_dir, "*.txt"))

    # Each element of this list will be a tuple: (speaker, text, time_stamp1, time_stamp2)
    transcriptions_per_file = []
    final_lines_per_file = []

    # Store unique phrases per speaker
    unique_phrases_per_speaker = {}

    # Process all text files in the output directory
    for txt_file in txt_files:
        speaker = get_speaker(txt_file)
        transcriptions, final_line = split_transcriptions(txt_file)

        # Store unique phrases
        for transcription in transcriptions:
            # Initialize speaker's unique phrases set if it doesn't exist yet
            if speaker not in unique_phrases_per_speaker:
                unique_phrases_per_speaker[speaker] = set()

            # If the phrase is not repeated, add it to transcriptions_per_file and to the unique phrases set
            if transcription[1] not in unique_phrases_per_speaker[speaker]:
                transcriptions_per_file.append(transcription)
                unique_phrases_per_speaker[speaker].add(transcription[1])

        # Add the final line without checking for repetitions
        final_lines_per_file.append((speaker, final_line))

    # Sort the transcriptions by time_stamp1
    transcriptions_per_file.sort(key=lambda x: x[2])

    # Write to the merged text file
    with open(os.path.join(output_dir, "merged.txt"), "w") as f:
        # Write the transcriptions
        for speaker, text, time_stamp1, time_stamp2 in transcriptions_per_file:
            f.write(f"{speaker}: {text} {time_stamp1} {time_stamp2}\n")

        # Write the final lines
        for speaker, text in final_lines_per_file:
            f.write(f"{speaker}: {text}\n")

# Main script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")  # Define command-line argument for input directory
    parser.add_argument("output_dir")  # Define command-line argument for output directory
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)  # Process all audio files in the input directory
    merge_files(args.output_dir)  # Merge all transcriptions into a single file
