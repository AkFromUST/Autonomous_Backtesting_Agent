from __future__ import annotations

from langchain_openai import ChatOpenAI


def _strip_channel_token(name: str) -> str:
    if "<|channel|>" in name:
        return name.split("<|channel|>")[0].strip()
    return name


class PatchedChatOpenAI(ChatOpenAI):
    """ChatOpenAI that strips Harmony <|channel|> bleed from tool call names.

    gpt-oss models embed thinking tokens into tool_calls[].function.name:
        'tool_name<|channel|>commentary about what I am doing'
    LangChain rejects the malformed name. This subclass cleans it before
    LangChain builds the AIMessage, preserving the model's reasoning otherwise.
    """

    def _create_chat_result(self, response, *args, **kwargs):
        for choice in response.choices:
            if choice.message and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    if tc.function and tc.function.name:
                        tc.function.name = _strip_channel_token(tc.function.name)
        return super()._create_chat_result(response, *args, **kwargs)
