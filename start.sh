#!/bin/bash



echo "Setting up pre-requisites..."
if [ -d "myenv" ]; then
    echo "Virtual environment 'myenv' already exists."
    source myenv/bin/activate
else
    echo "Creating virtual environment 'myenv'..."
    python3 -m venv myenv 
    source myenv/bin/activate
    echo "Virtual environment 'myenv' created."
fi



echo "Starting RAG AI"
python localragV8.py
