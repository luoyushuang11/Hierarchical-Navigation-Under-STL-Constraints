# A Bi-Level Neural-guided Path Integral Framework for Real-Time Full-Horizon Signal Temporal Logic

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green)
![OS](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange)

## 📖 Introduction

This repository presents a novel bi-level control framework designed for complex autonomous navigation tasks subject to **Signal Temporal Logic (STL)** specifications in complex environments. 

Rather than treating high-level planning and low-level control as isolated modules, our architecture tightly couples a low-frequency Reinforcement Learning (RL) planner with a high-frequency Model Predictive Path Integral (MPPI) controller, utilizing an innovative bi-level, constraint-satisfaction approach.

### 🌟 Core Architecture & Innovations:

* **High-Level Planner:** Acting as the high-level planner, the algorithm samples the optimal arrival time points for various sub-target regions at the initial stage of the task. It simultaneously generates a reference trajectory in real time to guide the low-level system.
* **Low-level Controller:** To strictly satisfy the temporal logic requirements defined by STL, the low-level controller constructs two distinct Time-Varying constraints:
  1. A time-varying constraint based on the sub-target arrival time points optimized by the high-level RL planner.
  2. A time-varying constraint based on the final deadline specified by the overall task requirements.
* **Guided MPPI for High-Efficiency Sampling:** A known bottleneck of standard MPPI is its sampling inefficiency in complex spaces. Our framework utilizes the reference trajectory provided by the high-level RL planner to actively guide and bias the MPPI's sampling distribution. This significantly enhances the sampling efficiency and computational real-time performance of the low-level controller.

## 🎥 Demonstration

*Watch the robot execute dynamic overtaking maneuvers among target vehicles with varying speeds using our hierarchical framework:*

https://github.com/user-attachments/assets/7401dfcf-27c4-434a-815d-17efebcf354a

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
git clone [https://github.com/luoyushuang11/A-Bi-Level-Neural-guided-Path-Integral-Framework.git](https://github.com/luoyushuang11/A-Bi-Level-Neural-guided-Path-Integral-Framework.git)
cd NG_PI
