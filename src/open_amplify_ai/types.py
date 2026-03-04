"""Data structures used for requests, responses, and API mapping."""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

# ---------------------------------------------------------------------------
# OpenAI Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """OpenAI-compatible model object."""

    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = "amplify-ai"


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
    max_tokens: Optional[int] = 4000
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None


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
