#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python red_team_parallel.py 0 4 &
CUDA_VISIBLE_DEVICES=1 python red_team_parallel.py 1 4 &
CUDA_VISIBLE_DEVICES=2 python red_team_parallel.py 2 4 &    
CUDA_VISIBLE_DEVICES=3 python red_team_parallel.py 3 4 &