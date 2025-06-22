export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd_sim.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=six_run \
    --demo_path=../../../demo_data/pick_cube_sim_30_demos_2025-01-22_11-04-51.pkl\
    --learner \
