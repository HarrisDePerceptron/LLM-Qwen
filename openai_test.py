from openai import OpenAI

# from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

prompts = [
    #"how many planets in our solor system. name them all. return a well formed json response ",
    #"your favourite one and why",
    #"write python code to calulate the factorial of an integer",
   #"now write it in rust without using recursion",
    #"now write it in c++",
    "my name is haris. what is the meaning of this name in arabic"
]

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]


for p in prompts:
    message = {"role": "user", "content": p}
    messages.append(message)
    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2-1.5B-Instruct",
        messages=messages,  # type: ignore
        stream=True,

    )
    # response = chat_response.choices[0].message.content
    #

    print("AI: ", end="")
    response = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content is not None:
            res = chunk.choices[0].delta.content
            print(res, end="")
            response += res

    print("\n")

    response_message = {"role": "system", "content": response}
    messages.append(response_message)
