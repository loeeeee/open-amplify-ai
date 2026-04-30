"""Streaming state machine for OpenAI-compatible responses."""
import json
import logging
import uuid
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.token_counting import calculate_cost, count_completion_tokens
from open_amplify_ai.tool_parsing import parse_tool_calls
from open_amplify_ai.types import AmplifyChatRequest, ToolDefinition

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Current streaming mode."""
    INIT = "init"  # Initial state
    CONTENT = "content"  # Streaming normal content
    TOOL_CALL = "tool_call"  # Streaming tool call
    DONE = "done"  # Stream complete


class StreamingStateMachine:
    """
    State machine for managing streaming responses.
    
    Handles transitions between content mode and tool-call mode,
    buffering output to determine mode before emitting.
    """
    
    def __init__(
        self,
        model: str,
        completion_id: str,
        created: int,
        tools: Optional[List[ToolDefinition]] = None,
        prompt_tokens: int = 0,
        input_cost_per_million: Optional[float] = None,
        output_cost_per_million: Optional[float] = None,
    ):
        self.model = model
        self.completion_id = completion_id
        self.created = created
        self.tools = tools
        self.prompt_tokens = prompt_tokens
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        
        self.mode = StreamingMode.INIT
        self.buffer = ""
        self.tool_call_detected = False
        self.content_emitted = False
        self.accumulated_content = ""  # Track all emitted content for token counting
        self.accumulated_tool_args = ""  # Track tool call arguments for token counting
    
    def process_delta(self, delta: str) -> List[Dict[str, Any]]:
        """
        Process a content delta and return OpenAI chunks to emit.
        
        Returns a list of chunks (may be empty if buffering).
        """
        chunks = []
        
        # Add to buffer
        self.buffer += delta
        
        # In INIT or CONTENT mode, check if we should transition to TOOL_CALL
        if self.mode in (StreamingMode.INIT, StreamingMode.CONTENT):
            # Try to detect tool call in buffer
            # Check if buffer looks like it might have a tool call
            should_check = (
                len(self.buffer) > 20 or  # Some minimum content
                self._looks_like_tool_call_start(self.buffer)  # Or has tool markers
            )
            
            if should_check:
                parse_result = parse_tool_calls(self.buffer, self.tools)
                
                if parse_result.is_tool_call:
                    # Transition to TOOL_CALL mode
                    self.mode = StreamingMode.TOOL_CALL
                    self.tool_call_detected = True
                    
                    # Handle mixed content: emit remaining_content first if present
                    if parse_result.remaining_content:
                        if not self.content_emitted:
                            # Emit role first if this is the first content
                            chunk = self._create_chunk({"role": "assistant", "content": ""})
                            chunks.append(chunk)
                        
                        # Emit the text content that was mixed with tool call
                        chunk = self._create_chunk({"content": parse_result.remaining_content})
                        chunks.append(chunk)
                        self.content_emitted = True
                        self.accumulated_content += parse_result.remaining_content
                    
                    # Emit tool call chunks
                    for tool_call in parse_result.tool_calls:
                        # First chunk with index and id
                        chunk = self._create_chunk({
                            "role": "assistant" if not self.content_emitted else None,
                            "content": None,
                            "tool_calls": [{
                                "index": 0,
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function_name,
                                    "arguments": "",
                                },
                            }],
                        })
                        chunks.append(chunk)
                        
                        # Second chunk with arguments
                        chunk = self._create_chunk({
                            "tool_calls": [{
                                "index": 0,
                                "function": {
                                    "arguments": tool_call.function_arguments,
                                },
                            }],
                        })
                        chunks.append(chunk)
                        self.accumulated_tool_args += tool_call.function_arguments
                    
                    # Clear buffer
                    self.buffer = ""
                    
                elif not self._looks_like_tool_call_start(self.buffer):
                    # Definitely not a tool call, emit as content
                    if self.mode == StreamingMode.INIT:
                        self.mode = StreamingMode.CONTENT
                        # Emit role first
                        chunk = self._create_chunk({"role": "assistant", "content": ""})
                        chunks.append(chunk)
                    
                    # Emit buffered content
                    chunk = self._create_chunk({"content": self.buffer})
                    chunks.append(chunk)
                    self.content_emitted = True
                    self.accumulated_content += self.buffer
                    self.buffer = ""
        
        elif self.mode == StreamingMode.TOOL_CALL:
            # Already in tool call mode, don't emit more content
            pass
        
        return chunks
    
    def flush_buffer(self) -> List[Dict[str, Any]]:
        """
        Flush any remaining buffer at end of stream.
        """
        chunks = []
        
        if self.buffer:
            # If we have buffer and haven't detected tool call, emit as content
            if not self.tool_call_detected:
                if self.mode == StreamingMode.INIT:
                    # Emit role first
                    chunk = self._create_chunk({"role": "assistant", "content": ""})
                    chunks.append(chunk)
                
                chunk = self._create_chunk({"content": self.buffer})
                chunks.append(chunk)
                self.content_emitted = True
                self.accumulated_content += self.buffer
            
            self.buffer = ""
        
        return chunks
    
    def finalize(self) -> Dict[str, Any]:
        """
        Create final chunk with finish_reason.
        """
        finish_reason = "tool_calls" if self.tool_call_detected else "stop"
        
        return self._create_chunk(
            delta={},
            finish_reason=finish_reason,
        )
    
    def create_usage_chunk(self) -> Dict[str, Any]:
        """Create usage chunk with estimated token counts, cache details, and cost.

        Completion tokens are computed from the accumulated content and
        tool-call arguments emitted during the stream.  Cost is estimated
        when model pricing is available.
        """
        completion_text = self.accumulated_content + self.accumulated_tool_args
        completion_tokens = count_completion_tokens(completion_text)
        total_tokens = self.prompt_tokens + completion_tokens
        request_cost = calculate_cost(
            self.prompt_tokens,
            completion_tokens,
            self.input_cost_per_million,
            self.output_cost_per_million,
        )
        usage: Dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": 0,
            },
        }
        if request_cost is not None:
            usage["cost"] = request_cost
        return {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": "",
            "choices": [],
            "usage": usage,
        }
    
    def _create_chunk(
        self,
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an OpenAI-compatible streaming chunk."""
        return {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": "",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
    
    def _looks_like_tool_call_start(self, text: str) -> bool:
        """
        Check if text looks like it might be the start of a tool call.
        
        This prevents premature emission of tool call patterns.
        """
        # Check for common tool call starters
        starters = [
            '{"_tool_call"',
            "{'_tool_call'",
            '[Tool Call:',
            '<tool_call>',
            '<tool_use>',
            '{"tool"',
            '{"command"',
        ]
        
        text_lower = text.lower().strip()
        for starter in starters:
            if starter.lower() in text_lower:
                return True
        
        return False


async def stream_amplify_response(
    amplify_request: AmplifyChatRequest,
    headers: Dict[str, str],
    model: str,
    completion_id: str,
    created: int,
    tools: Optional[List[ToolDefinition]] = None,
    include_usage: bool = False,
    prompt_tokens: int = 0,
    input_cost_per_million: Optional[float] = None,
    output_cost_per_million: Optional[float] = None,
) -> AsyncIterator[str]:
    """
    Stream Amplify response with state machine handling.
    
    Yields OpenAI-compatible SSE events.
    """
    state_machine = StreamingStateMachine(
        model=model,
        completion_id=completion_id,
        created=created,
        tools=tools,
        prompt_tokens=prompt_tokens,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )
    
    streaming_error = None
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=None)) as client:
            async with client.stream(
                "POST",
                f"{AMPLIFY_BASE_URL}/chat",
                headers=headers,
                json=amplify_request,
            ) as resp:
                resp.raise_for_status()
                
                async for line_str in resp.aiter_lines():
                    if not line_str:
                        continue
                    
                    # Parse Amplify delta
                    content_delta = ""
                    if line_str.startswith("data: "):
                        payload_str = line_str[6:]
                        if payload_str == "[DONE]":
                            break
                        try:
                            parsed = json.loads(payload_str)
                            content_delta = (
                                parsed.get("data", "")
                                or parsed.get("content", "")
                                or parsed.get("message", "")
                            )
                            # If content_delta is not a string, convert to JSON string
                            if not isinstance(content_delta, str):
                                content_delta = json.dumps(content_delta)
                            # If content looks like escaped JSON, try to unescape it
                            elif '\\"' in content_delta:
                                try:
                                    # Attempt to decode JSON string (handles escaped quotes)
                                    unescaped = json.loads(f'"{content_delta}"')
                                    if isinstance(unescaped, str):
                                        content_delta = unescaped
                                except (json.JSONDecodeError, ValueError):
                                    # Keep original if unescaping fails
                                    pass
                        except json.JSONDecodeError:
                            content_delta = payload_str
                    else:
                        content_delta = line_str
                    
                    if not content_delta:
                        continue
                    
                    # Process through state machine
                    chunks = state_machine.process_delta(content_delta)
                    
                    # Emit chunks
                    for chunk in chunks:
                        yield f"data: {json.dumps(chunk)}\n\n"
    
    except httpx.HTTPError as e:
        streaming_error = e
    
    # Flush any remaining buffer
    chunks = state_machine.flush_buffer()
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # Emit final chunk
    final_chunk = state_machine.finalize()
    yield f"data: {json.dumps(final_chunk)}\n\n"
    
    # Emit usage if requested
    if include_usage:
        usage_chunk = state_machine.create_usage_chunk()
        yield f"data: {json.dumps(usage_chunk)}\n\n"
    
    # Emit done marker
    yield "data: [DONE]\n\n"
    
    if streaming_error is not None:
        logger.warning("Stream disconnected early: %s", streaming_error)
