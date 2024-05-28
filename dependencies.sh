#!/bin/bash

eval "$(command conda "shell.bash" "hook" 2> /dev/null)"
eval "$(command conda "shell.zsh" "hook" 2> /dev/null)"

conda env create -f environment.yml
conda activate gdp6h_duacs_jaxparrow
pip install --upgrade pip
pip install -r requirements.txt --upgrade