#!/bin/bash

# Run YAPF (Python code formatter)
isort .
yapf --in-place $(git diff --cached --name-only -- '*.py')

# Set pre-commit check
# chmod +x pre-commit.sh
# cp pre-commit.sh .git/hooks/pre-commit