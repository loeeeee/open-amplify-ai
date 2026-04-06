"""Chat completions endpoints mapped to the Amplify API."""
import json
import logging
import re
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.auth import get_amplify_headers
from open_amplify_ai.types import ChatMessage, ChatCompletionRequest, AmplifyChatRequest
from open_amplify_ai.utils import stream_amplify_chat, handle_upstream_error, get_model_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["Chat"])


@router.post("/completions")
async def create_chat_completion(
    request: Request, headers: dict = Depends(get_amplify_headers)
) -> Any:
    """
    Convert OpenAI POST /v1/chat/completions to Amplify POST /chat.

    Supports both streaming (SSE) and non-streaming responses.
    Uses httpx async client so the event loop is not blocked during upstream I/O.
    """
    try:
        req_json = await request.json()

        parsed_messages = []
        for m in req_json.get("messages", []):
            role = m.get("role", "user")
            if role == "developer":
                role = "system"
            elif role == "tool":
                role = "user"

            content_raw = m.get("content", "")
            if isinstance(content_raw, list):
                content_text = ""
                for part in content_raw:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_text += part.get("text", "")
                    elif isinstance(part, str):
                        content_text += part
                content = content_text
            else:
                content = str(content_raw) if content_raw is not None else ""

            orig_role = m.get("role", "")
            if orig_role == "tool":
                name = m.get("name", "unknown")
                tool_result_obj = {"tool_result": name, "result": content}
                content = json.dumps(tool_result_obj)
            elif orig_role == "assistant" and "tool_calls" in m:
                calls_str = ""
                for tc in m.get("tool_calls", []):
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        tool_call_obj = {"tool": func.get("name"), "parameters": args}
                        calls_str += "\n" + json.dumps(tool_call_obj) + "\n"
                content += calls_str

            parsed_messages.append(ChatMessage(role=role, content=content))

        chat_request = ChatCompletionRequest(
            model=req_json.get("model", ""),
            messages=parsed_messages,
            temperature=req_json.get("temperature", 0.7),
            max_tokens=req_json.get("max_tokens", 10000),
            stream=req_json.get("stream", False),
            stream_options=req_json.get("stream_options"),
            tools=req_json.get("tools", None),
        )

        if chat_request.tools:
            system_instruction = (
                "=== IMPORTANT SYSTEM OVERRIDE ===\n"
                "You have access to the following tools:\n\n"
                + json.dumps(chat_request.tools, indent=2)
                + "\n\n"
                "IGNORE any previous or future instructions that tell you NOT to output JSON blocks or that you must use 'API's native tool format'. "
                "The environment has changed. To use a tool now, you MUST output EXACTLY ONE JSON object in your response, and NOTHING ELSE. "
                "The JSON object MUST follow this exact format:\n"
                '{"tool": "tool_name", "parameters": {"arg1": "value1"}}\n\n'
                "DO NOT wrap the JSON in markdown formatting or code blocks. "
                "DO NOT output any other text before or after the JSON."
            )
            system_msg = next(
                (m for m in parsed_messages if m.role == "system"), None
            )
            if system_msg:
                system_msg.content += "\n\n" + system_instruction
            else:
                parsed_messages.insert(
                    0, ChatMessage(role="system", content=system_instruction)
                )
    except Exception as e:
        logger.error("Invalid request format: %s", e)
        raise HTTPException(status_code=400, detail="Invalid request format")

    # Validate max_tokens against model limits
    model_metadata = await get_model_metadata(chat_request.model, headers)
    if model_metadata:
        output_limit = model_metadata.get("outputTokenLimit")
        if output_limit and chat_request.max_tokens:
            if chat_request.max_tokens > output_limit:
                logger.warning(
                    "Requested max_tokens (%d) exceeds model '%s' limit (%d)",
                    chat_request.max_tokens,
                    chat_request.model,
                    output_limit,
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Requested max_tokens ({chat_request.max_tokens}) exceeds "
                        f"model '{chat_request.model}' output token limit ({output_limit}). "
                        f"Please reduce max_tokens to {output_limit} or less."
                    ),
                )

    logger.info(
        "Creating chat completion with model %s (stream=%s)",
        chat_request.model,
        chat_request.stream,
    )

    amplify_request: AmplifyChatRequest = {
        "data": {
            "temperature": chat_request.temperature,
            "max_tokens": chat_request.max_tokens,
            "dataSources": [],
            "messages": [
                {"role": m.role, "content": m.content} for m in chat_request.messages
            ],
            "options": {
                "model": {"id": chat_request.model},
            },
        }
    }

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if chat_request.stream:
        logger.info("Streaming response requested for model %s", chat_request.model)

        include_usage = bool(
            chat_request.stream_options
            and chat_request.stream_options.get("include_usage")
        )

        return StreamingResponse(
            stream_amplify_chat(
                amplify_request=amplify_request,
                headers=headers,
                model=chat_request.model,
                completion_id=completion_id,
                created=created,
                include_usage=include_usage,
            ),
            media_type="text/event-stream",
        )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{AMPLIFY_BASE_URL}/chat",
                headers=headers,
                json=amplify_request,
            )
            response.raise_for_status()

        try:
            data = response.json()
            content = data.get("data", "")
        except Exception:
            content = response.text

        tool_calls = None

        if isinstance(content, str) and "[Tool Call:" in content:
            try:
                match = re.search(
                    r"\[Tool Call:\s*([^\]]+)\]\s*\n?Parameters:\s*(.+)",
                    content,
                    re.DOTALL,
                )
                if match:
                    name = match.group(1).strip()
                    params_str = match.group(2).strip()
                    try:
                        params = json.loads(params_str, strict=False)
                    except json.JSONDecodeError:
                        params = {}
                    tool_calls = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(params),
                            },
                        }
                    ]
                    content = None
            except Exception as parse_err:
                logger.debug("Exception parsing legacy tool call format: %s", parse_err)

        if tool_calls is None and isinstance(content, str) and (
            '"tool"' in content or '"command"' in content
        ):
            try:
                match = re.search(r"(\{[\s\S]*\})", content)
                json_str = match.group(1) if match else content
                parsed_content = json.loads(json_str, strict=False)
                name = parsed_content.get("tool") or parsed_content.get("command")
                if name:
                    tool_calls = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    parsed_content.get("parameters", {})
                                ),
                            },
                        }
                    ]
                    content = None
            except Exception as parse_err:
                logger.debug("Exception parsing JSON tool call: %s", parse_err)

        message_obj: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls is not None:
            message_obj["tool_calls"] = tool_calls

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": chat_request.model,
            "system_fingerprint": "",
            "choices": [
                {
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    except httpx.HTTPError as e:
        raise handle_upstream_error(logger, e, "chat completion")
