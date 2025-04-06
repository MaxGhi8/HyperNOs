#!/bin/bash
# This is a script to remove empty directories in the current directory and its subdirectories.
# It will print the names of the directories it removes.

find . -type d -empty -print -delete