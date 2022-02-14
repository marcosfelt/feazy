#!/usr/bin/env bash
source $(poetry env info --path)/bin/activate
python run.py
terminal-notifier -title "Feazy" -message "PhD schedule optimization finished"
