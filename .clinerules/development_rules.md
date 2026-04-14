# Development Rules

Always write a concise report at docs-vibe on what you have implemented and what are the remaining or unsolved issues.

Keep project simple.

Never use emoji in the code and documentation.

Always use `nix-shell` to run scripts and commands.

Always add a test when user encountered a bug or issue.

## Environment Management

The development environment is NixOS.

Always use shell.nix to config project environment.

## Bash

Always check and remove dead code in the script.

## Documentation

### README

Always update README.md after each task.

### Docs-vibe

Always document concise implementation details and basic usage in docs-vibe/
with a sequential file name.

Always create documentation before start writing any code.

When writing documentation, always document the user's intent in
its original words, in addition to a more logical and concise rephrasing.

Always update the newly created documentation at the end of the task to reflect
the latest status of the code.

If the task could not be finished in one-shot, always create a reflection
documentation in docs-vibe/reflection after each task. The reflection document
needs to describe the circumstance that is causing the issues, and the
solution, in addition to other important information. When a reflection document
is created, never repeat its content in development report in docs-vibe.

Never update the original path in the documentation. Always note the rename or
change in path.

Always document what files are deleted at the end of the development, even
though some are not caused by the agents.
