export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_conrft_octo_sim.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=first_run \
    --actor \
    # --eval_checkpoint_step=26000 \