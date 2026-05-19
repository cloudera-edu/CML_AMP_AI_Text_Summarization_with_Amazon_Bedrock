import gradio as gr
import os
import json
from utils import bedrock

accept = 'application/json'
contentType = 'application/json'

# Nova Micro supports on-demand inference directly
NOVA_MODEL_ID = "amazon.nova-micro-v1:0"

with open('amp_2_app/example.txt', 'r') as file:
    example_text = file.read()

examples = {
    'CML Documentation': example_text
}


def example_lookup(text):
    if text:
        return examples[text]
    return ''


example_instruction = (
    "Please provide a summary of the following text. "
    "Do not add any information that is not mentioned in the text below."
)


def clear_out():
    cleared_tuple = (
        gr.Textbox.update(value=""),
        gr.Textbox.update(value=""),
        gr.Textbox.update(value=""),
        gr.Textbox.update(value="")
    )
    return cleared_tuple


# List of LLM models to use for text summarization
models = [
    NOVA_MODEL_ID,
    'anthropic.claude-3-haiku-20240307-v1:0'
]


# Setting up the prompt syntax for the corresponding model
def prompt_construction(
    modelId,
    instruction="[instruction]",
    prompt="[input_text]"
):

    if modelId == NOVA_MODEL_ID:
        full_prompt = instruction + "\n\n" + prompt

    elif modelId == 'anthropic.claude-3-haiku-20240307-v1:0':
        full_prompt = instruction + "\n" + prompt

    return full_prompt


# Setting up the API call in the correct format for the corresponding model
def json_format(
    modelId,
    tokens,
    temperature,
    top_p,
    full_prompt="[input text]"
):

    if modelId == NOVA_MODEL_ID:

        body = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": full_prompt
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": tokens,
                "temperature": temperature,
                "topP": top_p
            }
        })

    elif modelId == 'anthropic.claude-3-haiku-20240307-v1:0':

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": tokens,
            "temperature": temperature,
            "top_k": 250,
            "top_p": top_p,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_prompt
                        }
                    ]
                }
            ]
        })

    return body


def display_format(modelId):

    if modelId == NOVA_MODEL_ID:

        body = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "[input_text]"
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": "[max_tokens]",
                "temperature": "[temperature]",
                "topP": "[top_p]"
            }
        })

    elif modelId == 'anthropic.claude-3-haiku-20240307-v1:0':

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": "[max_tokens]",
            "temperature": "[temperature]",
            "top_k": 250,
            "top_p": "[top_p]",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "[input_text]"
                        }
                    ]
                }
            ]
        })

    return body


def summarize(
    modelId,
    input_text,
    instruction_text,
    max_tokens,
    temperature,
    top_p
):

    # Initializing the bedrock client using AWS credentials
    boto3_bedrock = bedrock.get_bedrock_client(
        region=os.environ.get("AWS_DEFAULT_REGION", None)
    )

    full_prompt = prompt_construction(
        modelId,
        instruction_text,
        input_text
    )

    body = json_format(
        modelId,
        max_tokens,
        temperature,
        top_p,
        full_prompt
    )

    # Foundation model is invoked here to generate a response
    response = boto3_bedrock.invoke_model(
        body=body,
        modelId=modelId,
        accept=accept,
        contentType=contentType
    )

    response_body = json.loads(response.get('body').read())

    # Extract the output from the API response
    if modelId == NOVA_MODEL_ID:

        result = response_body['output']['message']['content'][0]['text']

    elif modelId == 'anthropic.claude-3-haiku-20240307-v1:0':

        result = response_body['content'][0]['text']

    return result.strip('\n')


with gr.Blocks() as demo:

    with gr.Row():

        gr.Markdown("# Amazon Bedrock Text Summarization")

        example_holder = gr.Textbox(
            visible=False,
            label="Input Text",
            value="example"
        )

    with gr.Row():

        modelId = gr.Dropdown(
            label="Choose a Bedrock Model:",
            choices=models,
            value=NOVA_MODEL_ID
        )

    with gr.Row():

        with gr.Column(scale=4):

            custom_instruction = gr.Textbox(
                label="Input your prompt instruction:",
                value=example_instruction
            )

            input_text = gr.Textbox(
                label="Input your text",
                placeholder="Insert some long text here..."
            )

            example = gr.Examples(
                examples=[
                    [example_instruction, "CML Documentation"]
                ],
                inputs=[
                    custom_instruction,
                    example_holder
                ]
            )

        with gr.Column(scale=4):

            with gr.Accordion(
                "Advanced Generation Options",
                open=False
            ):

                max_new_tokens = gr.Slider(
                    minimum=0,
                    maximum=4096,
                    step=1,
                    value=512,
                    label="Max Tokens"
                )

                temperature = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=0.5,
                    label="Temperature"
                )

                top_p = gr.Slider(
                    minimum=0,
                    maximum=1.0,
                    step=0.01,
                    value=1.0,
                    label="Top P"
                )

            with gr.Accordion(
                "Bedrock API Request Details",
                open=False
            ):

                instruction_prompt = gr.Code(
                    label="Instruction Prompt",
                    value=prompt_construction(
                        NOVA_MODEL_ID
                    )
                )

                input_format = gr.JSON(
                    label="Input Format",
                    value=display_format(
                        NOVA_MODEL_ID
                    )
                )

                with gr.Accordion(
                    "AWS Credentials",
                    open=False
                ):

                    label = gr.Markdown(
                        "These can be set from the project env vars"
                    )

                    region = gr.Markdown(
                        "**Region**: " +
                        str(os.getenv('AWS_DEFAULT_REGION'))
                    )

                    access_key = gr.Markdown(
                        "**Access Key**: " +
                        str(os.getenv('AWS_ACCESS_KEY_ID'))
                    )

                    secret_key = gr.Markdown(
                        "**Secret Key**: *****"
                    )

            summarize_btn = gr.Button(
                "Summarize",
                variant='primary'
            )

            reset_btn = gr.Button("Reset")

        with gr.Column(scale=4):

            output = gr.Textbox(
                label="Output Text"
            )

    summarize_btn.click(
        fn=summarize,
        inputs=[
            modelId,
            input_text,
            custom_instruction,
            max_new_tokens,
            temperature,
            top_p
        ],
        outputs=output,
        api_name="summarize"
    )

    reset_btn.click(
        fn=clear_out,
        inputs=[],
        outputs=[
            input_text,
            output,
            example_holder,
            custom_instruction
        ],
        show_progress=False
    )

    modelId.change(
        fn=prompt_construction,
        inputs=[modelId],
        outputs=instruction_prompt
    )

    modelId.change(
        fn=display_format,
        inputs=modelId,
        outputs=input_format
    )

    example_holder.change(
        fn=example_lookup,
        inputs=example_holder,
        outputs=input_text,
        show_progress=False
    )

demo.launch(
    server_port=int(os.getenv('CDSW_APP_PORT')),
    enable_queue=True,
    show_error=True,
    server_name='127.0.0.1',
)
