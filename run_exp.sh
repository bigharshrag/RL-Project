#! /bin/bash
for i in {1..10}
do
    echo "Run " $i
    python3 run_sarsa_lambda.py
done