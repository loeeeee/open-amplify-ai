import os
import re

FILES_TO_EDIT = [
    "src/open_amplify_ai/routers/assistants.py",
    "src/open_amplify_ai/routers/files.py",
    "src/open_amplify_ai/routers/models.py",
    "src/open_amplify_ai/routers/threads.py",
    "src/open_amplify_ai/routers/vector_stores.py",
    "src/open_amplify_ai/probe_api.py",
    "src/open_amplify_ai/server.py",
]

def refactor_file(filepath):
    if not os.path.exists(filepath):
        return
        
    with open(filepath, "r") as f:
        content = f.read()

    # Need to make sure handle_upstream_error is imported
    if "handle_upstream_error" not in content and "utils import" in content:
        content = re.sub(
            r"(from open_amplify_ai\.utils import[^\n]*)",
            r"\1, handle_upstream_error",
            content
        )
    elif "handle_upstream_error" not in content and "from open_amplify_ai.utils" not in content:
        content = "from open_amplify_ai.utils import handle_upstream_error\n" + content

    # The pattern matches:
    #     except requests.exceptions.RequestException as e:
    #         logger.error("Error <action>...", ...)
    #         raise HTTPException(...)
    
    # We'll use a regex that matches the except block, extracts the basic action
    # and replaces the contents.
    
    pattern = re.compile(
        r'( +)except requests\.exceptions\.RequestException as e:\n'
        r'( +)logger\.error\("Error ([^"]+?)[\: "].*?\n'
        r'( +)raise HTTPException\(status_code=500.*?\n',
        re.MULTILINE
    )
    
    def replacer(match):
        indent = match.group(1)
        action = match.group(3)
        # e.g action might be "fetching models" or "creating assistant %s"
        # We strip formatting symbols
        clean_action = action.replace("%s", "").strip()
        return f'{indent}except requests.exceptions.RequestException as e:\n{indent}    raise handle_upstream_error(logger, e, "{clean_action}")\n'

    new_content = pattern.sub(replacer, content)
    
    with open(filepath, "w") as f:
        f.write(new_content)
        
for f in FILES_TO_EDIT:
    refactor_file(f)
