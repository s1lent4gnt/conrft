# Training on Franka Arm Walkthrough

We demonstrate how to use our code with real robot manipulators with a representative task in the paper: Pick Banana. We provide detailed instructions and tips the entire pipeline for using ConRFT to fine-tune Octo in a real-world environemnt. 

## Pick Banana
### Procedure
#### Setup Franka Arm Control Server
1. To setup the Python environment and install the Franka controllers, please refer to [README.md](../README.md).

2. To setup the workspace, please refer to the image of our workspace setup in our paper.

3. Adjust for the weight of the wrist cameras by editing `Desk > Settings > End-effector > Mechanical Data > Mass`.

4. Unlock the robot and activate FCI in the Franka Desk. The `franka_server` launch file is found at [serl_robot_infra/robot_servers/launch_right_server.sh](../serl_robot_infra/robot_servers/launch_right_server.sh). You will need to edit the `setup.bash` path as well as the flags for the `python franka_server.py` command. You can refer to the [README.md](../serl_robot_infra/README.md) for `serl_robot_infra` for instructions on setting these flags. To launch the server, run:
   
```bash
bash serl_robot_infra/robot_servers/launch_right_server.sh
```

#### Specify Training Configuration for Your Workplace
For each task, we create a folder in the experiments folder to store data (i.e. task demonstrations, reward classifier data, training run checkpoints), launch scripts, and training configurations (see [experiments/task1_pick_banana](../examples/experiments/task1_pick_banana/)). Next, we will walkthrough all of the changes you need to make the training configuration in [experiments/task1_pick_banana/config.py](../examples/experiments/task1_pick_banana/config.py)) to begin training:

1. First, in the `EnvConfig` class, change `SERVER_URL` to the URL of the running Franka server.

2. Next, we need to configure the cameras. For this task, we used two wrist cameras. All cameras used for a task (both for the reward classifier and policy training) are listed in `REALSENSE_CAMERAS` and their corresponding image crops are set in `IMAGE_CROP` in the `EnvConfig` class. The camera keys used for policy training and for the reward classifier are listed in `TrainConfig` class in `image_keys` and `classifier_keys` respectively. Change the serial numbers in `REALSENSE_CAMERAS` to the serial numbers of the cameras in your setup (this can be found in the RealSense Viewer application). To adjust the image crops (and potentially the exposure), you can run the reward classifier data collection script (see step 6) or the demonstration data collection script (see step 8) to visualize the camera inputs.

3. Finally, we need to collect some poses for the training process. For this task, `TARGET_POSE` refers to the arm pose when putting the banana to the plate, and `RESET_POSE` refers to the arm pose to reset to. `ABS_POSE_LIMIT_HIGH` and `ABS_POSE_LIMIT_LOW` determine the bounding box for the policy. We have `RANDOM_RESET` enabled, meaning there is randomization around the `RESET_POSE` for every reset (`RANDOM_XY_RANGE` and `RANDOM_RZ_RANGE` control the amount of randomization). You should recollect `TARGET_POSE`, and ensure the bounding box is set for safe exploration. To collect the current pose of the Franka arm, you can run:
    ```bash
    curl -X POST http://<FRANKA_SERVER_URL>:5000/getpos_euler
    ```

#### Training Reward Classifier
The reward for this task is given via a reward classifier trained on camera images. For this task, we use the same specified images in `classifier_keys` for training the policy to train the reward classifier. The following steps goes through collecting classifier data and training the reward classifier.

1. First, we need to collect training data for the classifier. Navigate into the examples folder and run:
    ```bash
    cd examples
    python record_success_fail.py --exp_name task1_pick_banana --successes_needed 200
    ```
   While the script is running, all transitions recorded are marked as negative (or no reward) by default. If the space bar is held during a transition, that transition will be marked as positive. The script will terminate when enough positive transitions have been collected (defaults to 200, but can be set via the successes_needed flag). For this task, you should collect negative transitions of the RAM stick held in various locations in the workspace and during the insertion process, and pressing the space bar when the RAM is fully inserted. The classifier data will be saved to the folder `experiments/task1_pick_banana/classifier_data`.

   > **TIP**: To train a classifier robust against false positives (this is important for training a successful policy), we've found it helpful to collect 2-3x more negative transitions as positive transitions to cover all failure modes. 

2. To train the reward classifier, navigate to this task's experiment folder and run:
    ```bash
    cd experiments/task1_pick_banana
    python ../../train_reward_classifier.py --exp_name task1_pick_banana
    ```
    The reward classifier will be trained on the camera images specified by the classifier keys in the training config. The trained classifier will be saved to the folder `experiments/task1_pick_banana/classifier_ckpt`.

#### Recording Demonstrations
A small number of human demonstrations is crucial for stage I (Cal-ConRFT), and for this task, we use 30 demonstrations. 

1. To record the 30 demonstrations with the spacemouse, run:
    ```bash
    python ../../record_demos_octo.py --exp_name task1_pick_banana --successes_needed 30
    ```
    Once the episode is deemed successful by the reward classifier or the episode times out, the robot will reset. The script will terminate once 30 successful demonstrations have been collected, which will be saved to the folder `experiments/task1_pick_banana/demo_data`.

     > **TIP**: During the demo data collection progress, you may notice the reward classifier outputting false positives (episode terminating with reward given without a successful insertion) or false negatives (no reward given despite successful insertion). In that case, you should collect additional classifier data to target the classifier failure modes observed (i.e., if the classifier is giving false positives for holding RAM stick in the air, you should collect more negative data of that occurring). Alternatively, you can also adjust the reward classifier threshold, although we strongly recommend collecting additional classifier data (or even adding more classifier cameras/images if needed) before doing this.

#### Policy Training
Policy training is done asynchronously via an actor thread, responsible for rolling out the policy in the environment and sending the collected transitions to the learner thread, responsible for training the policy and sending the updated policy back to the actor. Both the actor and the learner should be running during policy training.

1. Inside the folder corresponding to the Pick Banana experiment ([experiments/task1_pick_banana](../examples/experiments/task1_pick_banana/)), you will find `run_actor_conrft.sh`, `run_learner_conrft_pretrain.sh` and `run_learner_conrft.sh`. In both scripts, edit `checkpoint_path` to point to the folder where checkpoints and other data generated in the training process will be saved to and in `run_learner_conrft_pretrain.sh` and `run_learner_conrft.sh`, edit `demo_path` to point to the path of the recorded demonstrations (if there are multiple demonstration files, you can provide multiple `demo_path` flags). Firstly, begin stage I (Cal-ConRFT):
    ```bash
    bash run_learner_conrft_pretrain.sh
    ```

   Then, to begin stage II (HIL-ConRFT), launch both threads:
    ```bash
    bash run_actor_conrft.sh
    bash run_learner_conrft.sh
    ```

2.  During online training, you should give some interventions as necessary with the spacemouse to speed up the training run, particularly closer to the beginning of the run or when the policy is exploring an incorrect behavior repeatedly. For reference, with the randomizations on and giving occasional interventions, the policy took around 1 hours to converge to 100% success rate.

3.  To evaluate the trained policy, add the flags `--eval_checkpoint_step=CHECKPOINT_NUMBER_TO_EVAL` and `--eval_n_trajs=N_TIMES_TO_EVAL` to `run_actor_conrft.sh`. Then, launch the actor:
    ```bash
    bash run_actor_conrft.sh
    ```
