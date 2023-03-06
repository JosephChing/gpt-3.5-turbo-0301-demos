# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import tiktoken

# Don't give this to anyone.
openai.api_key = ''

messageList = [
    {"role": "system", "content": "You are a helpful assistant."},
]


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += -1
        num_tokens += 2
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


while True:
    # Read user Input
    userInput = input("\n(user) \n")
    if userInput == "exit":
        break
    messageList.append({"role": "user", "content": userInput})

    # Count the number of prompt tokens, and the number of completion tokens allowed.
    promptTokens = num_tokens_from_messages(messageList, 'gpt-3.5-turbo-0301')
    completionTokens = 4095 - promptTokens
    print(f"{promptTokens} prompt tokens. {completionTokens} completion tokens available.")
    
    response = openai.ChatCompletion.create(
        # Required.
        model="gpt-3.5-turbo-0301",
        messages=messageList,

        # Optional
        user="Joseph",
        n=1,
        
        stop=["10"], # Stops once these sequences are detected. Up to 4 sequences allowed.
        max_tokens=completionTokens,  # Model by default can handle 4096 tokens. This is prompt tokens + completion tokens.
        presence_penalty=0,
        frequency_penalty=0,
        # logit_bias = ,
        stream=False,
        temperature=0,
        top_p=1,
        
    )
    messageList.append(response['choices'][0]['message'])

    print("\n(assistant) gpt-3.5-turbo-0301 ")
    print(response['choices'][0]['message']['content'] + "\n")
    print("Token Usage Info: ")
    print(f"    prompt_tokens        {response['usage']['prompt_tokens']}")
    print(f"    completion_tokens    {response['usage']['completion_tokens']}")
    print(f"    total_tokens         {response['usage']['total_tokens']}")



