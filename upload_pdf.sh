#!/bin/bash



# Function to check if a virtual environment is active
function is_venv_active() {
    # Check if the VIRTUAL_ENV variable is set and points to the myenv directory
    if [[ -z "$VIRTUAL_ENV" || "$VIRTUAL_ENV" != */myenv ]]; then
        return 1  # Not active
    else
        return 0  # Active
    fi
}

# Function to activate the virtual environment
function activate_venv() {
    source myenv/bin/activate
    echo "Virtual environment 'myenv' activated."
}

# Main script
if is_venv_active; then
    echo "Virtual environment 'myenv' is already active."
else
    echo "Virtual environment 'myenv' is not active. Activating..."
    activate_venv
fi


echo "PDF Upload To vault.txt"


python uploadV2.py 
