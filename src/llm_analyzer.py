import argparse
import base64
import json
from PIL import Image
import io
import boto3
from anthropic import Anthropic

# Constants
MEDIA_TYPE = "image/jpeg"
MODEL_NAME = "anthropic.claude-3-5-sonnet-20240620-v1:0"
TEMPERATURE = 0.0  # Lower temperature for more deterministic outputs
TOP_P = 0.3

prompt_text = """
You are an expert in analyzing a stylized representation of wooden wheel images. Your task is to examine the wheel in 
the provided image and give detailed information about its spokes and condition. The lines which extend from the center
of the wheel to the outer rim are called spokes. Some of these spokes may be broken or damaged. Your analysis should
include the total number of spokes, the number of broken spokes, and the number of completed (undamaged) spokes.


Please follow these steps to analyze the wheel:

1. Carefully examine the entire wheel image, paying close attention to the spokes and their condition.
2. Count the total number of spokes in the wheel.
3. Identify any broken or damaged spokes. A spoke is considered broken if it is visibly interrupted or disconnected.
4. Count the number of broken spokes.
5. Calculate the number of completed (undamaged) spokes by subtracting the number of broken spokes from the total 
number of spokes.
6. Before providing your final answer, use the <thinking> tags to briefly explain your reasoning and how you arrived at 
your counts.
7. Provide your final analysis in the following format, enclosed in <analysis> tags:

<analysis>
Total spokes: [Insert number]
Broken spokes: [Insert number]
Completed spokes: [Insert number]
</analysis>

Remember to be thorough and accurate in your examination. If any part of the image is unclear or you're unsure about a 
particular spoke, mention this in your <thinking> section.

In examples you will see broken spokes are marked with red line, and completed spokes are marked with green line.
"""


def load_image_as_base64(image_path):
    """Load an image file and encode it as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_message_list(example_1_image_base64, example_2_image_base64, image_base64):
    """Create a list of messages for the model input."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": MEDIA_TYPE,
                                             "data": example_1_image_base64}},
                {"type": "text", "text": prompt_text}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<thinking> Upon careful examination of the wheel image, I can see that this "
                                         "wheel has a unique design. There are clear lines extending from the center to"
                                         " the rim, which represent the spokes. I'll count these carefully:"
                                         "1. There are 5 full-length spokes that reach from the center to the rim "
                                         "without interruption."
                                         "2. There are 1 protrusions from the center that appear to be partial or"
                                         "broken spokes. It's not connected to the rim directly."
                                         "3. There are 4 protrusions on the rim that could be considered the outer "
                                         "ends of broken spokes. They are not fully connected to the center."
                                         "4. It seems that one of the partial spokes is opposite the other on the rim,"
                                         "indicating that they may be part of the same spoke."
                                         "For the total count, I'll include all of these elements as they all appear to"
                                         "be part of the wheel's spoke system. The broken spokes are those that don't "
                                         "fully connect the center to the rim, considering previous analysis that some "
                                         "of them may be part of the same spoke. The completed spokes are those that "
                                         "extend fully from the center to the rim without interruption."
                                         "5 of the spokes are completed. One of the partial spokes is opposite the "
                                         "other which indicates that they are parts of the same spoke. That makes the "
                                         "broken spokes count 4."
                                         "</thinking>"
                                         "<analysis>\nTotal spokes: 9\nBroken spokes: 4\nCompleted spokes: 5\n"
                                         "</analysis>"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": MEDIA_TYPE,
                                             "data": example_2_image_base64}},
                {"type": "text", "text": prompt_text}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<thinking> Upon careful examination of the wheel image, I can see that this "
                                         "wheel has a unique design. There are clear lines extending from the center to"
                                         " the rim, which represent the spokes. I'll count these carefully:"
                                         "1. There are 7 full-length spokes that reach from the center to the rim "
                                         "without interruption."
                                         "2. There are 2 protrusions from the center that appear to be partial or"
                                         "broken spokes. They are not connected to the rim directly."
                                         "3. There are 2 protrusions on the rim that could be considered the outer "
                                         "ends of broken spokes. They are not fully connected to the center."
                                         "4. It seems that one of the partial spokes is opposite the other on the rim,"
                                         "indicating that they may be part of the same spoke."
                                         "For the total count, I'll include all of these elements as they all appear to"
                                         "be part of the wheel's spoke system. The broken spokes are those that don't "
                                         "fully connect the center to the rim, considering previous analysis that some "
                                         "of them may be part of the same spoke. The completed spokes are those that "
                                         "extend fully from the center to the rim without interruption."
                                         "7 of the spokes are completed. One of the partial spokes is opposite the "
                                         "other which indicates that they are parts of the same spoke. That makes the "
                                         "broken spokes count 3."
                                         "</thinking>"
                                         "<analysis>\nTotal spokes: 10\nBroken spokes: 3\nCompleted spokes: 7\n"
                                         "</analysis>"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": MEDIA_TYPE, "data": image_base64}},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]


def analyze_with_bedrock(image_base64, example_1_image_base64, example_2_image_base64):
    """
    Analyze the image using AWS Bedrock with Claude 3 Sonnet, including a few-shot example with an actual image.

    Args:
    image_base64 (str): Base64 encoded image string of the wheel to analyze.
    example_1_image_base64 (str): Base64 encoded image string of the first example wooden wheel.
    example_2_image_base64 (str): Base64 encoded image string of the second example wooden wheel.

    Returns:
    str: Analysis result from the model.
    """
    try:
        bedrock = boto3.client(service_name='bedrock-runtime')

        message_list = create_message_list(example_1_image_base64, example_2_image_base64, image_base64)

        prompt = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "messages": message_list
        }

        # Serialize the request body to JSON and encode as bytes
        body = json.dumps(prompt).encode('utf-8')

        # Make the API call to Bedrock
        response = bedrock.invoke_model(
            modelId=MODEL_NAME,
            accept='application/json',
            contentType='application/json',
            body=body
        )

        # Parse and return the response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body['content'][0]['text']

    except json.JSONDecodeError as e:
        return f"Error decoding JSON response: {str(e)}"
    except KeyError as e:
        return f"Unexpected response structure: {str(e)}"
    except Exception as e:
        return f"Unexpected error in Bedrock analysis: {str(e)}"


def analyze_with_anthropic(image_base64, api_key, example_1_image_base64, example_2_image_base64):
    """
    Analyze the image using Anthropic API directly, including a few-shot example with an actual image.

    Args:
    image_base64 (str): Base64 encoded image string of the wheel to analyze.
    api_key (str): Anthropic API key.
    example_1_image_base64 (str): Base64 encoded image string of the first example wooden wheel.
    example_2_image_base64 (str): Base64 encoded image string of the second example wooden wheel.

    Returns:
    str: Analysis result from the model.
    """
    try:
        client = Anthropic(api_key=api_key)

        message_list = create_message_list(example_1_image_base64, example_2_image_base64, image_base64)

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            messages=message_list
        )

        return response.content[0].text

    except Exception as e:
        return f"Error in Anthropic analysis: {str(e)}"


def main(image_path, method, anthropic_api_key=None):
    """
    Main function to orchestrate the image analysis process.

    Args:
    image_path (str): Path to the image file to analyze.
    method (str): Analysis method ('bedrock' or 'anthropic').
    anthropic_api_key (str, optional): Anthropic API key for direct API usage.
    """
    # Encode the image to analyze
    encoded_image = load_image_as_base64(image_path)
    # Encode the example image
    # example_1_image_base64 = load_image_as_base64("../data/example_1_w0.png")
    # example_2_image_base64 = load_image_as_base64("../data/example_2_w9.png")
    example_1_image_base64 = load_image_as_base64("../data/wooden_wheel_0.png")
    example_2_image_base64 = load_image_as_base64("../data/wooden_wheel_9.png")

    # Analyze based on the chosen method
    if method == 'bedrock':
        result = analyze_with_bedrock(encoded_image, example_1_image_base64, example_2_image_base64)
    elif method == 'anthropic':
        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required for the Anthropic method")
        result = analyze_with_anthropic(encoded_image,
                                        anthropic_api_key,
                                        example_1_image_base64,
                                        example_2_image_base64)
    else:
        raise ValueError("Invalid method. Choose 'bedrock' or 'anthropic'")

    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze wheel images using LLMs")
    parser.add_argument("image_path", help="Path to the wheel image to analyze")
    parser.add_argument("method", choices=['bedrock', 'anthropic'], help="Analysis method")
    parser.add_argument("--api_key", help="Anthropic API key (required for 'anthropic' method)")

    args = parser.parse_args()

    main(args.image_path, args.method, args.api_key)
