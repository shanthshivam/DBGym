#!/bin/bash

# Run YAPF (Python code formatter)
yapf --in-place $(git diff --cached --name-only)
isort .

# Set pre-commit check
# chmod +x pre-commit.sh
# cp pre-commit.sh .git/hooks/pre-commit