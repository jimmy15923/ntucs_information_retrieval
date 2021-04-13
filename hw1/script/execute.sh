#!/bin/bash
# Put your command below to execute your program.
# Replace "./my-program" with the command that can execute your program.
# Remember to preserve " $@" at the end, which will be the program options we give you.

while getopts r:i:o:m:d: option
do
case "${option}"
in
r) ROCHIOO=${OPTARG};;
i) INPUT=${OPTARG};;
o) OUTPUT=${OPTARG};;
m) MODEL=${OPTARG};;
d) DATA=${OPTARG};;
esac
done

python main.py $@
