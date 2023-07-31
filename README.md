
# Video summarization demo

In this repo, we implement a script to create a short text/audio summary of a video/audio. 


# Quickstart

Prerequisites:  
 - Python 3.7 or higher. Installation instructions for [Linux](https://docs.python-guide.org/starting/install3/linux/), [MacOS](https://docs.python-guide.org/starting/install3/osx/), [Windows](https://www.digitalocean.com/community/tutorials/install-python-windows-10)
 - FFMpeg. Installation instructions for [Linux](https://www.tecmint.com/install-ffmpeg-in-linux/), [MacOS](https://phoenixnap.com/kb/ffmpeg-mac), [Windows](https://phoenixnap.com/kb/ffmpeg-windows)

Note that installation can take up to ~20-30 mins, so make sure to install the requirements before a demonstration call if any.

Run demo: 

1. Clone this repository
2. Run the terminal from this folder
3. Run following commands in the terminal:
```
python3 -m venv venv
export SPEECHKIT_API_KEY=<your SpeechKit key>
export OPENAI_API_KEY=<your OpenAI key>
pip3 install -r requirements.txt
python3 run.py <path to the video file>
```
You can find how to get the Speechkit API key [here](https://cloudil.co.il/docs/iam/operations/api-key/create) and the OpenAI API key [here](https://platform.openai.com/account/api-keys)

# Full walkthrough

The processing pipeline consists of three steps:
1. Extracting audio from video files using `FFmpeg` and `moviepy`
2. Speech recognition using CloudIL
3. Text summarization using OpenAI API
4. Text-to-speech with CloudIL.

We will create a summary of the ["How do airplanes stay in the air?"](https://www.ted.com/talks/raymond_adkins_how_do_airplanes_stay_in_the_air) TED talk as an example. You can download the video of the talk with [this](https://download.ted.com/products/162986.mp4) direct link.

## Step 1: Extracting audio from the video

Requirements: 
1. `FFmpeg`.  Installation instructions for [Linux](https://www.tecmint.com/install-ffmpeg-in-linux/), [MacOS](https://phoenixnap.com/kb/ffmpeg-mac), [Windows](https://phoenixnap.com/kb/ffmpeg-windows)  
FFmpeg is a package for video editing. We will use it to extract audio from video.  
2. `moviepy`
```
pip install moviepy
```
Moviepy is a Python package for processing video files. It can be used for video editing, composing and creating effects, and acts as Python bindings to `ffmpeg` executive. Extracting audio to `.mp3` can be done with the following code:
```py
from moviepy.editor import VideoFileClip

video_path = "2021e-raymond-adkins-aerodynamic-lift-002-5000k.mp4"
out_path = "audio.mp3"

video_obj = VideoFileClip(video_path)
video_obj.audio.write_audiofile(out_path)
```

The full script for this task is located at `extract_audio.py`. It includes a CLI interface, so you can use it directly from a terminal:
```
python extract_audio.py 2021e-raymond-adkins-aerodynamic-lift-002-5000k.mp4 --out audio.mp3
```
This will create a file `audio.mp3` with the audio from the specified video file. Also, if `audio.mp3` already exists, the script will raise an error to avoid overwriting.

## Step 2: Speech recognition

At this stage, we will use CloudIL for performing speech recognition.

Requirements: 
1. `yandexcloud` Python package
```
pip install yandexcloud
```
CloudIL uses the same API as Yandex Cloud, so we can use this package as the SDK for CloudIL.

2. `reprint` Python package
```
pip install reprint
```
This package is used to print recognized text to the terminal in real-time. In a real production environment, this would not be required.

To use CloudIL services, you will need an [API Key](https://cloudil.co.il/docs/iam/concepts/authorization/api-key) or an [IAM Token](https://cloudil.co.il/docs/iam/operations/iam-token/create-for-federation). For testing purposes, it is simpler to use an API Key:
1. In the [CloudIL management console](https://console.cloudil.co.il), select the folder the service account belongs to.
2. Go to the ***Service accounts*** tab.
3. Choose a service account and click the line with its name.
4. Click ***Create new key*** in the top panel.
5. Click ***Create API key***.
6. Save the private key. 

To perform speech recognition with CloudIL, you need to do the following: 
 - Send recognition configuration
 - Send audio chunks
 - Receive recognition results and process them

CloudIL API works on gRPC protocol. At the same time, you can use `yandexcloud` Python SDK so that you don't need to dive into the details of the protocol. Let's walk through the implementation:

### Setup recognition configuration:
```py
recognize_options = stt_pb2.StreamingOptions(
  recognition_model=stt_pb2.RecognitionModelOptions(
    # Declare audio format as MP3 container    
    audio_format=stt_pb2.AudioFormatOptions(
      container_audio=stt_pb2.ContainerAudio(
        container_audio_type=stt_pb2.ContainerAudio.MP3,
      )
    ),
    # Enable text normalization. This enables text postprocessing:
    # punctuation, converting numerals into numbers etc.
    text_normalization=stt_pb2.TextNormalizationOptions(
      text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
      profanity_filter=False,
      literature_text=False
    ),
    # Set recognition language as English. If you omit parameter 
    # language_restriction, the languade will be detected automatically.
    language_restriction=stt_pb2.LanguageRestrictionOptions(
      restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
      language_code=['en-US']
    ),
    # We want to receive real-time updates, this is optional
    audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME
  )
)
``` 

The full list of the configuration parameters can be found in the [CloudIL documentation](https://cloudil.co.il/docs/speechkit/stt-v3/api-ref/grpc/stt_service#RecognitionModelOptions).

### Sending configuration and audio chunks

First, we create a Python generator that will produce requests to send to the API. The first request is a configuration request, all the following ones are audio chunks.

```py
import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc

# Size of an audio buffer in bytes that will be sent to the API
CHUNK_SIZE = 4000

def read_audio(audio_file_name):
  recognize_options = # See above

  # Yield a message with recognition settings.
  yield stt_pb2.StreamingRequest(session_options=recognize_options)

  # Read the audio file and yield requests with its contents in portions.
  with open(audio_file_name, 'rb') as f:
    data = f.read(CHUNK_SIZE)
    while data != b'':
      yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))
      data = f.read(CHUNK_SIZE)
```
Note that since we send the audio in chunks, the API can work not only with full recorded files, but with the streaming audio as well. 

Now, we will create a method to establish a connection and send the prepared requests to CloudIL.

```py
def recognize_audio(audio_file_name, out_file_name):
  # Establish a connection with the server.
  cred = grpc.ssl_channel_credentials()
  channel = grpc.secure_channel("api.speechkit.cloudil.com:443", cred)
  stub = stt_service_pb2_grpc.RecognizerStub(channel)

  # In this example, we get the API key from the environment variable
  api_key = os.environ["SPEECHKIT_API_KEY"]
  # Send data for recognition. 
  # Recognize streaming is an object that consumes requests 
  # from the read_audio generator and yields responses
  it = stub.RecognizeStreaming(
    read_audio(audio_file_name), metadata=(("authorization", f"Api-Key {api_key}"),)
  )

  try:
    for r in it:
      # Process responses, more on that later
      pass
  # Handle errors
  except grpc._channel._Rendezvous as err:
    print(f"Error code {err._state.code}, message: {err._state.details}")
    raise err
```
The final piece of the recognition puzzle is processing responses from the CloudIL API. Three types of response events contain recognition results:
 - `partial`: These are the events that represent partial recognition results where the speaker has not finished their speech yet, but it's still can be useful to show the recognition results in real-time. When processing static audio, that is not that necessary, but in some cases, for example, voice assistants, showing current recognition results to a user is vital.
 - `final`: This event is emitted when the speaker finished their speech, so you can process the full recognition result.
 - `final_refinement`: The event is emitted after the `final` event and contains the same recognized text, but normalized. This event is sent by the server only if normalization is enabled in the configuration.

So, we will process the responses as follows:
```py
from reprint import output
it = stub.RecognizeStreaming(...) # See above

# output() is a context manager that allows editing printed text
with output(initial_len=1) as output_lines:
  for r in it:
    # "alternatives" object will contain the list of the possible
    # recognized texts returned by the API
    event_type, alternatives = r.WhichOneof("Event"), None
    
    # For "partial" and "final" responses just save the alternatives
    if event_type == "partial" and len(r.partial.alternatives) > 0:
      alternatives = [a.text for a in r.partial.alternatives]
    elif event_type == "final":
      alternatives = [a.text for a in r.final.alternatives]

    # When the "final_refinement" event is received, save 
    # current block to the file and add a line for stdout  
    elif event_type == "final_refinement":
      alternatives = [a.text for a in r.final_refinement.normalized_text.alternatives]
      output_lines.append("")
      with open(out_file_name, "a") as f:
        f.write(alternatives[0])
    else:
      continue

    # change output, such that "partial" updates will overwrite previous
    # updates in the terminal, not append to them.
    output_lines[-1] = alternatives[0]
```

The full script for the speech recognition task is `recognize_audio.py`. You can use it from the terminal directly:
```
 python recognize_audio.py --path audio.mp3
```

### Step 3. Summarization with the OpenAI API

First, you will need to get the API key. You can do it [here](https://platform.openai.com/account/api-keys).

The integration with the OpenAI services can be implemented with the official Python SDK: 
```
pip install openai
```

Using this SDK is extremely clear: 
```py
prompt = "<text to summarize>"
augmented_prompt = f"summarize this text: {prompt}"

# Send a request to OpenAI 
resp = openai.Completion.create(

  # Model ID. Da Vinci is the most advanced model available via API so far.
  model="text-davinci-003",

  # Prompt to send to the API
  prompt=augmented_prompt,

  # A temperature is a number between 0 and 1 and it can be defined as 
  # the parameter of "creativeness": if it is too low, 
  # the model will just copy text from the prompt.
  # If the temperature is too high, the model can produce text 
  # that is not related to the original
  temperature=.5,

  # Maximum number of the tokens in the response. One token is about 4 characters
  # 
  max_tokens=1000,
)
print(resp["choices"][0]["text"])
```

The full script to use from the terminal is located at `summarize.py`:
```
python summarize.py <path_to_text_file>
```

### Step 4. Text-to-speech

Text-to-speech API works pretty much the same as the speech recognition API:

```py
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
```

However, there is a limit of 250 characters that can be sent via API with a single request. Therefore, it is needed to split text into chunks. 
We will split the text into sentences, and if the sentence is too long, we split by words:

```py
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
```

Now we need to process the batches and merge them into single `.mp3` file.

```py
def synthesize(text):
    merged_bytes = io.BytesIO()
    for batch in split_batches(text):
        audio_bytes = synthesize_batch(batch)
        merged_bytes.write(audio_bytes.read())
    merged_bytes.seek(0)
    return merged_bytes
```

The full script to use from the terminal is located at `text_to_speech.py`:
```
python text_to_speech.py <path_to_text_file>
```

# Results

Here is an example of a summary that was produced from the TED talk. Note that your result can be different because of the randomness in the text generation.

> Albert Einstein attempted to design a flawed airplane wing in 1917 based on an incomplete theory of flight. Though inaccurate explanations still circulate today, lift is actually generated by the wings and air flow around them, with the pressure difference resulting in an upward force. Engineers use the Navier-Stokes equations to precisely model air flow around a wing and detail how lift is generated. Over a century later, lift is still a complex concept, but the physics of fluid in motion can help explain it.


Compare with the summary handcrafted by TED editors:


>By 1917, Albert Einstein had explained the relationship between space and time. But, that year, he designed a flawed airplane wing. His attempt was based on an incomplete theory of how flight works. Indeed, insufficient and inaccurate explanations still circulate today. So, where did Einstein go wrong? And how exactly do planes fly? Raymond Adkins explains the concept of aerodynamic lift.
