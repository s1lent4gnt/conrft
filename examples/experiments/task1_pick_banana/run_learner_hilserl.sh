export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=task1_pick_banana \
    --checkpoint_path=/root/online_rl/conrft/examples/experiments/task1_pick_banana/debug_hilserl \
    --demo_path=./demo_data/task1_pick_banana_30_demos.pkl  \
    --learner \