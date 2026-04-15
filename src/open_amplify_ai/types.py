"""Data structures used for requests, responses, and API mapping."""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

# ---------------------------------------------------------------------------
# Internal Representation (IR) - Core Types
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    """Explicit message roles with provenance preservation."""
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentPartType(str, Enum):
    """Content part types with explicit support matrix."""
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_FILE = "image_file"
    AUDIO = "audio"
    FILE = "file"


@dataclass
class ContentPart:
    """Single content part with type information."""
    type: ContentPartType
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    image_file: Optional[str] = None
    audio: Optional[Dict[str, Any]] = None
    file: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """Tool call with preserved ID for deterministic mapping."""
    id: str
    type: str  # Always "function" for now
    function_name: str
    function_arguments: str  # JSON string


@dataclass
class ToolResult:
    """Tool result with provenance and linkage."""
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False


@dataclass
class InternalMessage:
    """Internal message representation preserving all semantic information."""
    role: MessageRole
    content_parts: List[ContentPart] = field(default_factory=list)
    tool_calls: Optional[List[ToolCall]] = None
    tool_result: Optional[ToolResult] = None
    name: Optional[str] = None  # For tool messages
    
    def get_text_content(self) -> str:
        """Extract concatenated text from content parts."""
        return "".join(
            part.text for part in self.content_parts 
            if part.type == ContentPartType.TEXT and part.text
        )
    
    def has_unsupported_content(self) -> bool:
        """Check if message contains unsupported content types."""
        return any(
            part.type != ContentPartType.TEXT 
            for part in self.content_parts
        )


@dataclass
class ToolDefinition:
    """Tool definition with schema for validation."""
    type: str
    function: Dict[str, Any]
    
    def get_name(self) -> str:
        """Extract tool name."""
        return self.function.get("name", "")
    
    def get_schema(self) -> Dict[str, Any]:
        """Extract parameter schema."""
        return self.function.get("parameters", {})


@dataclass
class InternalRequest:
    """Internal request IR preserving all OpenAI request information."""
    model: str
    messages: List[InternalMessage]
    temperature: float = 0.7
    max_tokens: int = 10000
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    user: Optional[str] = None
    logprobs: Optional[bool] = None
    logit_bias: Optional[Dict[str, float]] = None
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    
    # Unsupported parameters tracking
    unsupported_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalResponse:
    """Internal response IR before OpenAI formatting."""
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    finish_reason: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: Optional[float] = None


# ---------------------------------------------------------------------------
# Error Types
# ---------------------------------------------------------------------------

class ErrorType(str, Enum):
    """OpenAI-compatible error types."""
    INVALID_REQUEST_ERROR = "invalid_request_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    SERVICE_UNAVAILABLE_ERROR = "service_unavailable_error"


@dataclass
class ErrorResponse:
    """Structured error response matching OpenAI format."""
    message: str
    type: ErrorType
    param: Optional[str] = None
    code: Optional[str] = None
    internal_request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Kilo-compatible model metadata
# ---------------------------------------------------------------------------

@dataclass
class ModelCost:
    """Per-model pricing in dollars per million tokens."""
    input: Optional[float] = None
    output: Optional[float] = None
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, omitting None values."""
        result: Dict[str, Any] = {}
        if self.input is not None:
            result["input"] = self.input
        if self.output is not None:
            result["output"] = self.output
        if self.cache_read is not None:
            result["cache_read"] = self.cache_read
        if self.cache_write is not None:
            result["cache_write"] = self.cache_write
        return result


@dataclass
class ModelLimit:
    """Per-model token limits."""
    context: Optional[int] = None
    output: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, omitting None values."""
        result: Dict[str, Any] = {}
        if self.context is not None:
            result["context"] = self.context
        if self.output is not None:
            result["output"] = self.output
        return result


@dataclass
class ModelCapabilities:
    """Feature flags describing what a model supports."""
    images: Optional[bool] = None
    system_prompt: Optional[bool] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, omitting None values."""
        result: Dict[str, Any] = {}
        if self.images is not None:
            result["images"] = self.images
        if self.system_prompt is not None:
            result["system_prompt"] = self.system_prompt
        if self.description is not None:
            result["description"] = self.description
        return result


# ---------------------------------------------------------------------------
# OpenAI Data structures (Legacy - kept for compatibility)
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """OpenAI-compatible model object with Kilo-consumable extensions."""

    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = "amplify-ai"

    # Kilo-consumable structured metadata
    cost: Optional[ModelCost] = None
    limit: Optional[ModelLimit] = None
    capabilities: Optional[ModelCapabilities] = None

    # Display name from Amplify
    display_name: Optional[str] = None

    # Legacy flat fields (kept for backward compatibility)
    max_output_tokens: Optional[int] = None  # Amplify's outputTokenLimit
    context_length: Optional[int] = None     # Amplify's inputContextWindow
    max_model_len: Optional[int] = None      # Total tokens (context + output)


@dataclass
class ChatMessage:
    """Single chat message with role and content."""

    role: str
    content: str


@dataclass
class ChatCompletionRequest:
    """Parsed OpenAI chat completion request body."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 10000  # Raised from 4000 to 10000
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Amplify request/response typed dicts 
# ---------------------------------------------------------------------------

class AmplifyModelOption(TypedDict):
    """Model selector object for Amplify chat options."""

    id: str


class AmplifyChatOptions(TypedDict, total=False):
    """Options block inside an Amplify chat request."""

    model: AmplifyModelOption
    assistantId: str
    prompt: str


class AmplifyChatMessage(TypedDict):
    """Single message in an Amplify chat request."""

    role: str
    content: str


class AmplifyChatData(TypedDict, total=False):
    """Data block for Amplify chat request."""

    temperature: Optional[float]
    max_tokens: Optional[int]
    dataSources: List[str]
    messages: List[AmplifyChatMessage]
    options: AmplifyChatOptions


class AmplifyChatRequest(TypedDict):
    """Top-level Amplify chat request payload."""

    data: AmplifyChatData


class AmplifyFileUploadData(TypedDict, total=False):
    """Data block for Amplify file upload request."""

    type: str
    name: str
    knowledgeBase: str
    tags: List[str]
    data: Dict[str, Any]
    actions: List[Dict[str, Any]]


class AmplifyFileUploadRequest(TypedDict):
    """Top-level Amplify file upload request payload."""

    data: AmplifyFileUploadData


class AmplifyFilesQueryData(TypedDict, total=False):
    """Data block for Amplify files/query request."""

    pageSize: int
    forwardScan: bool
    sortIndex: str
    tags: List[str]
    pageKey: Optional[Dict[str, Any]]


class AmplifyFilesQueryRequest(TypedDict):
    """Top-level Amplify files/query request payload."""

    data: AmplifyFilesQueryData


class AmplifyAssistantCreateData(TypedDict, total=False):
    """Data block for Amplify assistant create/update request."""

    name: str
    description: str
    assistantId: str
    tags: List[str]
    instructions: str
    disclaimer: str
    dataSources: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]


class AmplifyAssistantCreateRequest(TypedDict):
    """Top-level Amplify assistant create request payload."""

    data: AmplifyAssistantCreateData


class AmplifyKeyData(TypedDict):
    """Generic {key} data block used for delete and download endpoints."""

    key: str


class AmplifyKeyRequest(TypedDict):
    """Top-level wrapper for key-based Amplify requests."""

    data: AmplifyKeyData


class AmplifyTagsData(TypedDict, total=False):
    """Data block for Amplify tag operations."""

    tags: List[str]
    tag: str
    id: str


class AmplifyTagsRequest(TypedDict):
    """Top-level wrapper for tag operation requests."""

    data: AmplifyTagsData
