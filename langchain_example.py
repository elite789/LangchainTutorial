from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

# response = model.invoke("What color is the sky?")
# print(response)

# for chunk in model.stream("Why do parrots have colorful feathers?"):
#     print(chunk.text, end="|", flush=True)

# full = None  # None | AIMessageChunk
# for chunk in model.stream("What color is the sky?"):
#     full = chunk if full is None else full + chunk
#     # print(full.text)

# The
# The sky
# The sky is
# The sky is typically
# The sky is typically blue
# ...

# print(full.content_blocks)
# [{"type": "text", "text": "The sky is typically blue..."}]


# async for event in model.astream_events("Hello"):

#     if event["event"] == "on_chat_model_start":
#         print(f"Input: {event['data']['input']}")

#     elif event["event"] == "on_chat_model_stream":
#         print(f"Token: {event['data']['chunk'].text}")

#     elif event["event"] == "on_chat_model_end":
#         print(f"Full message: {event['data']['output'].text}")

#     else:
#         pass

# responses = model.batch([
#     "Why do parrots have colorful feathers?",
#     "How do airplanes fly?",
#     "What is quantum computing?"
# ])
# for response in responses:
#     print(response)

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


# model_with_tools = model.bind_tools([get_weather])  

# response = model_with_tools.invoke("What's the weather like in Boston?")
# for tool_call in response.tool_calls:
#     # View tool calls made by the model
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")

# Bind (potentially multiple) tools to the model
model_with_tools = model.bind_tools([get_weather])

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "The current weather in Boston is 72°F and sunny."