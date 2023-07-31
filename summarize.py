import argparse
import openai

def summarize(input_file):
    with open(input_file) as f:
        prompt = f.read()
    augmented_prompt = f"summarize this text: {prompt}"
    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=augmented_prompt,
        temperature=.5,
        max_tokens=2000,
    )
    return resp["choices"][0]["text"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    args = parser.parse_args()
    print(summarize(args.input_file))
