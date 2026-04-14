# Python Rules

Expect python 3.13 and above.

Always use virtual environment at .venv/

Always run python using nix-shell.

## Coding

Always Python logging system, and default log output to console, in addition to
files.

Always use Python typing system.

Always test run the code after each task is finished.

Always define data structure with dataclass.

When data type is too simple to be defined as dataclass, always create a
NamedDict for dictionary.

Always add tqdm progress bar to the process that would take a long time. (>10s)

Always follow the existing code structure.

Always exam the existing code logic first with a global view of the code, and
state how the implementation would change the code logic.

When optimizing CPU utilization, always use concurrent.futures instead of
multiprocessing, unless the user explicit agree the usage of multiprocessing.

Never use module level constant.

Always import module at the top of the file, unless specifically consent by the
user.

Always use fail-fast design.

Always use a modular design with helper functions.

Always print a concise report after script execution.

Always write doc string for each function.
