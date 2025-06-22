export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo_sim.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=first_run \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=../../../demo_data/pick_cube_sim_30_demos_2025-06-22_11-19-44.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner \
