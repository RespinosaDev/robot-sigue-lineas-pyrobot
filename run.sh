#!/bin/sh
# Lanza el simulador Stage con el mundo del examen final y el cerebro del robot
cd "$(dirname "$0")"
pyrobot -s StageSimulator.py -w ./mundo/finalExam-202526-line1-v01.world -r Stage.py -b ./BrainFinalExam.py
