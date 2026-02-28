#!/bin/bash
# Run BOC API with cuML support
# Usage: ./run_api_wsl.sh

cd /mnt/c/Users/saisa.DESKTOP-IRA1I5U/Documents/GAN/skill_palavar/project/api

# Install required packages if not present
pip install flask flask-cors numpy --quiet 2>/dev/null

# Run the API
python app.py
