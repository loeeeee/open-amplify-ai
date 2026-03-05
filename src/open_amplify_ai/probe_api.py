from open_amplify_ai.utils import handle_upstream_error
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm


os.makedirs("logs", exist_ok=True)
os.makedirs("docs", exist_ok=True)
os.makedirs("docs-vibe", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/probe_api.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

BASE_URL = "https://prod-api.vanderbilt.ai"


@dataclass
class APIEndpoint:
    """Represents a single API endpoint to probe."""
    name: str
    method: str
    path: str
    body: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, str]] = None
    variant_note: Optional[str] = None


class ProbeConfig(NamedTuple):
    """Holds authentication configuration loaded from the environment."""
    token: str
    email: str


def load_config() -> ProbeConfig:
    """Load token and email from environment. Fails fast if either is missing."""
    load_dotenv()
    token = os.getenv('AMPLIFY_AI_TOKEN')
    email = os.getenv('AMPLIFY_AI_EMAIL')
    if not token:
        logger.error("AMPLIFY_AI_TOKEN not found in .env file.")
        sys.exit(1)
    if not email:
        logger.error("AMPLIFY_AI_EMAIL not found in .env file.")
        sys.exit(1)
    logger.info("Configuration loaded successfully.")
    return ProbeConfig(token=token, email=email)


def build_endpoints(email: str) -> List[APIEndpoint]:
    """
    Build the full list of endpoints to probe, including conflict variants.

    Conflict variants are marked with a variant_note describing which source
    document the variant corresponds to.
    """
    # A placeholder file key composed from the user email, used in bodies.
    file_key = f"{email}/2024-07-15/example-file-id.json"
    assistant_id_prefix = f"{email}/ast/example-assistant-id"
    thread_id = f"{email}/thr/example-thread-id"

    endpoints: List[APIEndpoint] = [
        # 1. GET /available_models
        APIEndpoint(
            name='/available_models',
            method='GET',
            path='/available_models',
        ),

        # 2. POST /chat — canonical (PDF/JSON): model only inside options
        APIEndpoint(
            name='/chat',
            method='POST',
            path='/chat',
            body={'data': {
                'temperature': 0.7,
                'max_tokens': 100,
                'dataSources': [],
                'messages': [{'role': 'user', 'content': 'What is the capital of France?'}],
                'options': {
                    'ragOnly': False,
                    'skipRag': True,
                    'model': {'id': 'gpt-4o'},
                    'prompt': 'What is the capital of France?'
                }
            }},
            variant_note='Canonical (PDF/JSON): model as object inside options only',
        ),

        # 2b. POST /chat — conflict variant (CSV): model as top-level string too
        APIEndpoint(
            name='/chat [conflict: CSV top-level model]',
            method='POST',
            path='/chat',
            body={'data': {
                'model': 'gpt-4o',
                'temperature': 0.7,
                'max_tokens': 100,
                'dataSources': [],
                'messages': [{'role': 'user', 'content': 'What is the capital of France?'}],
                'options': {
                    'ragOnly': False,
                    'skipRag': True,
                    'model': {'id': 'gpt-4o'},
                    'prompt': 'What is the capital of France?'
                }
            }},
            variant_note='Conflict (CSV): model also as top-level string field',
        ),

        # 3. GET /state/share
        APIEndpoint(
            name='/state/share',
            method='GET',
            path='/state/share',
        ),

        # 4. POST /state/share/load
        APIEndpoint(
            name='/state/share/load',
            method='POST',
            path='/state/share/load',
            body={'data': {'key': f'{email}/example-shared-key.json'}},
        ),

        # 5. POST /files/upload — canonical (PDF/JSON): no actions field
        APIEndpoint(
            name='/files/upload',
            method='POST',
            path='/files/upload',
            body={'data': {
                'type': 'application/pdf',
                'name': 'probe-test-file.pdf',
                'knowledgeBase': 'default',
                'tags': [],
                'data': {}
            }},
            variant_note='Canonical (PDF/JSON): no actions field',
        ),

        # 5b. POST /files/upload — conflict variant (CSV): includes actions array
        APIEndpoint(
            name='/files/upload [conflict: CSV with actions]',
            method='POST',
            path='/files/upload',
            body={'data': {
                'actions': [
                    {'name': 'saveAsData', 'params': {}},
                    {'name': 'createChunks', 'params': {}},
                    {'name': 'ingestRag', 'params': {}},
                    {'name': 'makeDownloadable', 'params': {}},
                    {'name': 'extractText', 'params': {}}
                ],
                'type': 'application/pdf',
                'name': 'probe-test-file.pdf',
                'knowledgeBase': 'default',
                'tags': [],
                'data': {}
            }},
            variant_note='Conflict (CSV): includes actions array in body',
        ),

        # 6. POST /files/query
        APIEndpoint(
            name='/files/query',
            method='POST',
            path='/files/query',
            body={'data': {'pageSize': 10, 'sortIndex': '', 'forwardScan': False}},
        ),

        # 6b. POST /files (op=/delete) — base64-encoded data variant (observed in web UI)
        # The web UI encodes {"key": file_key} as base64 and passes it as the "data" field
        # alongside dispatch fields: method, path, op, service.
        APIEndpoint(
            name='/files [op=/delete, base64 data]',
            method='POST',
            path='/files',
            body={
                'data': base64.b64encode(json.dumps({'key': file_key}).encode()).decode(),
                'method': 'POST',
                'op': '/delete',
                'path': '/files',
                'service': 'file',
            },
            variant_note='File delete dispatch (UI-observed): base64-encoded {key} in data field',
        ),

        # 6c. POST /files (op=/delete) — plain JSON data variant (alternative to probe)
        APIEndpoint(
            name='/files [op=/delete, plain JSON data]',
            method='POST',
            path='/files',
            body={
                'data': {'key': file_key},
                'method': 'POST',
                'op': '/delete',
                'path': '/files',
                'service': 'file',
            },
            variant_note='File delete dispatch (alternative): plain JSON {key} in data field',
        ),

        # 7. GET /files/tags/list — canonical (PDF/JSON)
        APIEndpoint(
            name='/files/tags/list',
            method='GET',
            path='/files/tags/list',
            variant_note='Canonical (PDF/JSON): GET method',
        ),

        # 7b. POST /files/tags/list — conflict variant (CSV says POST)
        APIEndpoint(
            name='/files/tags/list [conflict: CSV POST]',
            method='POST',
            path='/files/tags/list',
            body=None,
            variant_note='Conflict (CSV): POST method',
        ),

        # 8. POST /files/tags/create
        APIEndpoint(
            name='/files/tags/create',
            method='POST',
            path='/files/tags/create',
            body={'data': {'tags': ['probe-test-tag']}},
        ),

        # 9. POST /files/tags/delete
        APIEndpoint(
            name='/files/tags/delete',
            method='POST',
            path='/files/tags/delete',
            body={'data': {'tag': 'probe-test-tag'}},
        ),

        # 10. POST /files/set_tags
        APIEndpoint(
            name='/files/set_tags',
            method='POST',
            path='/files/set_tags',
            body={'data': {'id': file_key, 'tags': ['probe-test-tag']}},
        ),

        # 11. POST /embedding-dual-retrieval
        APIEndpoint(
            name='/embedding-dual-retrieval',
            method='POST',
            path='/embedding-dual-retrieval',
            body={'data': {
                'userInput': 'Can you describe the policies outlined in the document?',
                'dataSources': ['global/example-document-id.content.json'],
                'limit': 10
            }},
        ),

        # 12. POST /assistant/create
        APIEndpoint(
            name='/assistant/create',
            method='POST',
            path='/assistant/create',
            body={'data': {
                'name': 'Probe Test Assistant',
                'description': 'Created by probe_api.py for testing purposes.',
                'assistantId': '',
                'tags': ['probe-test'],
                'instructions': 'Respond to user queries about general knowledge topics.',
                'disclaimer': 'This assistant is for probe testing only.',
                'dataSources': [],
                'tools': []
            }},
        ),

        # 13. GET /assistant/list
        APIEndpoint(
            name='/assistant/list',
            method='GET',
            path='/assistant/list',
        ),

        # 14. POST /assistant/share
        APIEndpoint(
            name='/assistant/share',
            method='POST',
            path='/assistant/share',
            body={'data': {
                'assistantId': 'ast/example-assistant-id',
                'recipientUsers': [email],
                'note': 'Probe test share'
            }},
        ),

        # 15. POST /assistant/delete
        APIEndpoint(
            name='/assistant/delete',
            method='POST',
            path='/assistant/delete',
            body={'data': {'assistantId': 'astp/example-assistant-id'}},
        ),

        # 16. POST /assistant/create/codeinterpreter
        APIEndpoint(
            name='/assistant/create/codeinterpreter',
            method='POST',
            path='/assistant/create/codeinterpreter',
            body={'data': {
                'name': 'Probe Code Interpreter Assistant',
                'description': 'Code interpreter assistant for probe testing.',
                'tags': ['probe-test'],
                'instructions': 'Analyze data and create visualizations.',
                'dataSources': [file_key],
                'fileKeys': [],
                'tools': [{'type': 'code_interpreter'}]
            }},
        ),

        # 17. POST /assistant/files/download/codeinterpreter
        APIEndpoint(
            name='/assistant/files/download/codeinterpreter',
            method='POST',
            path='/assistant/files/download/codeinterpreter',
            body={'data': {'key': assistant_id_prefix}},
        ),

        # 18. DELETE /assistant/openai/thread/delete (query param: threadId)
        APIEndpoint(
            name='/assistant/openai/thread/delete',
            method='DELETE',
            path='/assistant/openai/thread/delete',
            params={'threadId': thread_id},
        ),

        # 19. DELETE /assistant/openai/delete (query param: assistantId)
        APIEndpoint(
            name='/assistant/openai/delete',
            method='DELETE',
            path='/assistant/openai/delete',
            params={'assistantId': assistant_id_prefix},
        ),

        # 20. POST /assistant/chat/codeinterpreter
        APIEndpoint(
            name='/assistant/chat/codeinterpreter',
            method='POST',
            path='/assistant/chat/codeinterpreter',
            body={'data': {
                'assistantId': assistant_id_prefix,
                'messages': [{
                    'role': 'user',
                    'content': 'What can you help me with?',
                    'dataSourceIds': [file_key]
                }]
            }},
        ),
    ]

    return endpoints


def probe_endpoint(endpoint: APIEndpoint, config: ProbeConfig) -> Dict[str, Any]:
    """
    Probe a single API endpoint and return the result dict.

    The result includes the HTTP method, path, status code, response body,
    and any request or network error encountered.
    """
    url = BASE_URL + endpoint.path if not endpoint.path.startswith('http') else endpoint.path
    headers = {
        'Authorization': f'Bearer {config.token}',
        'Content-Type': 'application/json'
    }

    result: Dict[str, Any] = {
        'name': endpoint.name,
        'method': endpoint.method,
        'path': endpoint.path,
        'url': url,
        'variant_note': endpoint.variant_note,
        'status_code': None,
        'response': None,
        'error': None
    }

    try:
        if endpoint.method == 'GET':
            response = requests.get(url, headers=headers, params=endpoint.params, timeout=15)
        elif endpoint.method == 'POST':
            response = requests.post(url, headers=headers, json=endpoint.body, timeout=15)
        elif endpoint.method == 'PUT':
            response = requests.put(url, headers=headers, json=endpoint.body, timeout=15)
        elif endpoint.method == 'DELETE':
            response = requests.delete(url, headers=headers, params=endpoint.params, timeout=15)
        else:
            result['error'] = f"Unsupported method {endpoint.method}"
            return result

        result['status_code'] = response.status_code
        try:
            result['response'] = response.json()
        except json.JSONDecodeError:
            result['response'] = response.text

    except requests.exceptions.RequestException as e:
        result['error'] = str(e)

    return result


def redact_email(text: str, email: str) -> str:
    """Replace occurrences of the user's email in a string with a redaction marker."""
    return text.replace(email, '<user-email>')


def write_detailed_report(results: List[Dict[str, Any]], email: str) -> None:
    """Write the full diagnostic report to docs-vibe/."""
    report_path = 'docs-vibe/17_amplify_api_report.md'
    try:
        with open(report_path, 'w') as f:
            f.write("# Amplify AI API Endpoint Probing Report\n\n")
            f.write("This report documents all probed API endpoints, including conflict variants.\n\n")
            f.write(f"Total probed: {len(results)}\n\n")
            for res in results:
                f.write(f"## {res['method']} {res['path']}\n")
                if res.get('variant_note'):
                    f.write(f"*Variant: {res['variant_note']}*\n\n")
                f.write(f"- **URL**: `{res['url']}`\n")
                f.write(f"- **Status Code**: {res['status_code']}\n")
                if res['error']:
                    f.write(f"- **Error**: {res['error']}\n")
                f.write("### Response\n")
                f.write("```json\n")
                raw = json.dumps(res['response'], indent=2) if isinstance(res['response'], (dict, list)) else str(res['response'])
                f.write(redact_email(raw, email))
                f.write("\n```\n\n")
        logger.info("Detailed report written to %s", report_path)
    except Exception as e:
        logger.error("Failed to write detailed report: %s", e)


def write_api_reference(results: List[Dict[str, Any]], email: str) -> None:
    """Write the concise probed API reference to docs/."""
    report_path = 'docs/amplify_api_probed.md'
    try:
        with open(report_path, 'w') as f:
            f.write("# Amplify AI API Probed Reference\n\n")
            f.write("Verified API surface based on live probes. Conflict variants are shown with notes.\n\n")
            for res in results:
                header = f"{res['method']} {res['path']}"
                if res.get('variant_note'):
                    header += f" ({res['variant_note']})"
                f.write(f"## {header}\n")
                f.write(f"- **Status**: {res['status_code']}\n")
                if res['status_code'] is None:
                    f.write("- **Analysis**: NETWORK ERROR.\n\n")
                elif res['status_code'] < 400:
                    f.write("- **Analysis**: WORKING.\n\n")
                elif res['status_code'] == 401:
                    f.write("- **Analysis**: UNAUTHORIZED (valid endpoint, token rejected).\n\n")
                elif res['status_code'] == 403:
                    f.write("- **Analysis**: FORBIDDEN.\n\n")
                elif res['status_code'] == 404:
                    f.write("- **Analysis**: NOT FOUND (bad path or bad IDs in body).\n\n")
                elif res['status_code'] == 502:
                    f.write("- **Analysis**: BAD GATEWAY (gateway/auth error).\n\n")
                else:
                    f.write(f"- **Analysis**: FAILED ({res['status_code']}).\n\n")
                f.write("```json\n")
                raw = json.dumps(res['response'], indent=2) if isinstance(res['response'], (dict, list)) else str(res['response'])
                raw = redact_email(raw, email)
                if len(raw) > 600:
                    raw = raw[:600] + "\n... [truncated]"
                f.write(raw)
                f.write("\n```\n\n")
        logger.info("API reference written to %s", report_path)
    except Exception as e:
        logger.error("Failed to write API reference: %s", e)


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a concise execution summary to stdout."""
    total = len(results)
    success = sum(1 for r in results if r['status_code'] is not None and 200 <= r['status_code'] < 300)
    client_err = sum(1 for r in results if r['status_code'] is not None and 400 <= r['status_code'] < 500)
    server_err = sum(1 for r in results if r['status_code'] is not None and r['status_code'] >= 500)
    network_err = sum(1 for r in results if r['status_code'] is None)

    logger.info("=== Probe Summary ===")
    logger.info("Total probed : %d", total)
    logger.info("2xx Success  : %d", success)
    logger.info("4xx Client   : %d", client_err)
    logger.info("5xx Server   : %d", server_err)
    logger.info("Network err  : %d", network_err)
    logger.info("=====================")


def main() -> None:
    """Entry point for the Amplify AI API prober."""
    logger.info("Starting Amplify AI API Probe")
    config = load_config()
    endpoints = build_endpoints(config.email)
    logger.info("Probing %d endpoints (including conflict variants)...", len(endpoints))

    results: List[Dict[str, Any]] = []
    for endpoint in tqdm(endpoints, desc="Probing endpoints"):
        res = probe_endpoint(endpoint, config)
        results.append(res)

    write_detailed_report(results, config.email)
    write_api_reference(results, config.email)
    print_summary(results)


if __name__ == "__main__":
    main()
