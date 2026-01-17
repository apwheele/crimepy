#!/bin/bash

# Run pytest with verbose output
output=$(pytest -v 2>&1)
exit_code=$?

# If tests failed, send output to stderr and exit with code 2
if [ $exit_code -ne 0 ]; then
  echo "$output" >&2
  exit 2
fi

# Tests passed - silent success
exit 0