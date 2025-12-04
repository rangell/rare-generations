#!/bin/sh


# CUDA_VISIBLE_DEVICES=2 python red_team_parallel.py 0 2 &    
# CUDA_VISIBLE_DEVICES=3 python red_team_parallel.py 1 2 &


CUDA_VISIBLE_DEVICES=2 python red_team_parallel.py 0 1 &    
