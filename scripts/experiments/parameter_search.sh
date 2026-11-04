#!/bin/bash

# Run in chunks of 10
BATCH_SIZE=10

# Total parameter combos
TOTAL=81000

START=0
while [ $START -lt $TOTAL ]
do
  END=$((START + BATCH_SIZE))
  if [ $END -gt $TOTAL ]; then
    END=$TOTAL
  fi

  echo "Running parameters from $START to $((END-1))"
  uv run scripts/experiments/parameter_search.py \
    --path-config config.toml \
    --path-results output \
    --batch-start $START \
    --batch-end $END

  START=$END
done
