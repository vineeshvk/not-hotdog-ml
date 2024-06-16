#!/bin/bash

dir_name='logs/'

# Check whether the file exists or not
if [ -d "$dir_name" ]; then
    rm -r "$dir_name/"
    echo "$dir_name is removed"
else
    echo "$dir_name already removed"
fi
