#!/bin/bash
for dir in *-s*; do
	new_name=$(echo "$dir" | sed -E 's/-s([0-9]+)/\1/')
	mv "$dir" "new_name"
done
