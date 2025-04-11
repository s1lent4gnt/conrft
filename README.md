# ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://cccedric.github.io/conrft/)

We provide examples to fine-tune Octo, on the top of [HIL-SERL](https://github.com/rail-berkeley/hil-serl) that provides the base environment to perform robotic manipulation tasks with human interventions. The following sections describe how to use our code. 


**Table of Contents**
- [ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy](#conrft-a-reinforced-fine-tuning-method-for-vla-models-via-consistency-policy)
  - [🛠️ Installation Instructions](#️-installation-instructions)
  - [💻 Overview and Code Structure](#-overview-and-code-structure)
  - [✉️ Contact](#️-contact)
  - [🙏 Acknowledgement](#-acknowledgement)
  - [📝 Citation](#-citation)

## 🛠️ Installation Instructions
1. **Setup Conda Environment:**
    create an environment with
    ```bash
    conda create -n conrft python=3.10
    ```

2. **Install Jax as follows:**
    - For CPU (not recommended):
        ```bash
        pip install --upgrade "jax[cpu]"
        ```

    - For GPU:
        ```bash
        pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```

    - For TPU
        ```bash
        pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ```
    - See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

3. **Install the Octo**
    ```bash
    git clone git@github.com:octo-models/octo.git
    cd octo
    pip install -e .
    pip install -r requirements.txt
    ```

4. **Install the serl_launcher**
    ```bash
    cd serl_launcher
    pip install -e .
    pip install -r requirements.txt
    ```

5. **Install for serl_robot_infra** 
   
   Please refer to the [README](./serl_robot_infra/README.md) in the `serl_robot_infra` directory for installation instructions and details on operating the Franka robot arm. This document includes guidance on setting up the impedance-based [serl_franka_controllers](https://github.com/rail-berkeley/serl_franka_controllers). After completing the installation, you should be able to start the robot server and interact with the `franka_env` gym for hardware control.


## 💻 Overview and Code Structure

We offers a set of code for fine-tuning Octo in robotic manipulation tasks. The approach's pipeline consists of an actor thread and a learner thread, both of which interact with the robot gym environment. These two threads operate asynchronously, with data transmitted from the actor to the learner node over the network using [agentlace](https://github.com/youliangtan/agentlace). The learner thread periodically updates the policy and syncs it with the actor. 

**Table for code structure**

| Code Directory | Description |
| --- | --- |
| examples | Scripts for policy training, demonstration data collection, reward classifier training |
| serl_launcher | Main code for HIL-SERL |
| serl_launcher.agents | Agent Policies (e.g. SAC, BC) |
| serl_launcher.wrappers | Gym env wrappers |
| serl_launcher.data | Replay buffer and data store |
| serl_launcher.vision | Vision related models and utils |
| serl_robot_infra | Robot infra for running with real robots |
| serl_robot_infra.robot_servers | Flask server for sending commands to robot via ROS |
| serl_robot_infra.franka_env | Gym env for Franka robot |

We provide a step-by-step guide in [franka_walkthrough](/docs/franka_walkthrough.md) to fine-tune VLA with ConRFT on a Franka robot.

## ✉️ Contact
For any questions, please feel free to email [chenyuhui2022@ia.ac.cn](mailto:chenyuhui2022@ia.ac.cn).

## 🙏 Acknowledgement
Our code is built upon [CPQL](https://github.com/cccedric/cpql/), [Octo](https://github.com/octo-models/octo), [HIL-SERL](https://github.com/rail-berkeley/hil-serl). We thank all these authors for their nicely open sourced code and their great contributions to the community.

## 📝 Citation

If you find our research helpful and would like to reference it in your work, please consider the following citations:

```bibtex
@article{chen2025conrft,
  title={ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy},
  author={Chen, Yuhui and Tian, Shuai and Liu, Shugao and Zhou, Yingting and Li, Haoran and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2502.05450},
  year={2025}
}
```
