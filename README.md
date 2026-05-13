

Uploading fast_overtake.mp4…

# Hierarchical Navigation Under STL Constrains: RL Global Planner & CBF-MPPI Local Controller

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green)
![OS](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange)

## 📖 Introduction

This repository presents a novel hierarchical control framework designed for complex autonomous navigation tasks subject to **Signal Temporal Logic (STL)** specifications in complex environments. 

Rather than treating global and local planning as isolated modules, our architecture tightly couples a low-frequency Reinforcement Learning (RL) planner with a high-frequency Model Predictive Path Integral (MPPI) controller, utilizing an innovative **Dual Time-Varying Control Barrier Function (Dual TV-CBF)** approach.

### 🌟 Core Architecture & Innovations:

* **Low-Frequency Top-Level Controller (TGPO):** Acting as the global planner, the TGPO algorithm samples the optimal arrival time points for various sub-target regions at the initial stage of the task. It simultaneously generates a reference trajectory in real time to guide the lower-level system.
* **Dual TV-CBF for STL Satisfaction:** To strictly satisfy the temporal logic requirements defined by STL, the bottom-level controller constructs two distinct Time-Varying CBFs:
  1. A TV-CBF based on the sub-target arrival time points optimized by the RL planner.
  2. A TV-CBF based on the final deadline specified by the overall task requirements.
* **Guided MPPI for High-Efficiency Sampling:** A known bottleneck of standard MPPI is its sampling inefficiency in complex spaces. Our framework utilizes the reference trajectory provided by the top-level RL to actively guide and bias the MPPI's sampling distribution. This significantly enhances the sampling efficiency and computational real-time performance of the bottom-level controller.

## 🎥 Demonstration

*Watch the robot execute dynamic overtaking maneuvers among target vehicles with varying speeds using our hierarchical framework:*

<video src="./assets/fast_overtake.mp4" controls="controls" width="100%" autoplay loop></video>

*(Note: If the video does not render in your Markdown viewer, please find the source file in the `./assets/` directory.)*

## ⚙️ Prerequisites

The framework relies heavily on GPU acceleration for real-time MPPI sampling. The codebase has been extensively tested on the following configuration:

* **OS:** Ubuntu 22.04
* **Hardware:** NVIDIA GPU (Tested on GeForce RTX 3070Ti Laptop)
* **NVIDIA Driver:** >= 580.159
* **CUDA Version:** 13.0 
* **Python:** 3.10

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/luoyushuang11/Hierarchical-Navigation-Under-STL-Constrains.git
cd CBF_MPPI
```

**2. Create and activate the Conda environment**
```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate cbf_mppi_env
```
*(Alternatively, you can install dependencies via pip: `pip install -r requirements.txt`)*

## 🚀 Quick Start

The project is modularized into highly independent simulation scenarios. To run a simulation, simply navigate to the desired scenario directory and execute the `rlmppi.py` script.

For example, to run the dynamic overtaking scenario shown in the demonstration:
```bash
cd simulation_overtake_fast
python3 rlmppi.py
```

## 📁 Project Structure & Scenarios

The repository includes 9 comprehensive benchmarks covering mazes, autonomous overtaking, and reach-avoid tasks:

```text
CBF_MPPI/
├── simulation_maze_dualdyn/           # Maze scenario with 2 dynamic obstacles
├── simulation_maze_simple/            # Basic static maze scenario
├── simulation_maze_singledyn/         # Maze scenario with 1 dynamic obstacle
├── simulation_overtake_fast/          # Fully heterogeneous-speed dynamic obstacles scenario
├── simulation_overtake_normal/        # Partially heterogeneous-speed dynamic obstacles scenario
├── simulation_overtake_slow/          # Homogeneous-speed dynamic obstacles scenario
├── simulation_reach_avoid_disordered/ # Reach-avoid task with disordered obstacles
├── simulation_reach_avoid_ordered9/   # Reach-avoid task with 9 ordered obstacles
├── simulation_reach_avoid_ordered13/  # Reach-avoid task with 13 ordered obstacles
├── assets/                            # Media assets (e.g., demo videos)
├── environment.yml                    # Conda environment configuration
└── requirements.txt                   # Pip dependencies
```



