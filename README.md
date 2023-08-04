This repository is a fork of [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) to enable training in dynamic wind conditions.

We contribute the following:
- A Wind class that defines wind (gust) behavior and models blade flapping and aerodynamic drag effects: [wind.py](https://github.com/irom-lab/gym-pybullet-drones/blob/train/gym_pybullet_drones/assets/wind.py),
- An instantiation of the popular open-source PX4 controller in Python based on [bobzwik](https://github.com/bobzwik/Quadcopter_SimCon/blob/master/Simulation/ctrl.py)'s,
- A framework for learning a residual input on top fo the PX4 controller.

## Installation

`requirements.txt` has been modified to include additional dependencies which are needed for learning the residual policy. These can be installed with:

```
pip3 install -e .
```

## Training in Windy Conditions

The main training script is [scripts/learn_hover_wind_residual.py](https://github.com/irom-lab/gym-pybullet-drones/blob/train/script/learn_hover_wind_residual.py).

A `cfg.yml` file in `scripts` provies the training parameters; make sure to update `cfg.yml` to your specifications. (Including making the `log_dir` in `cfg.yml`)

To begin the training process, run the following:
```
~/gym-pybullet-drones$ python3 script/learn_hover_wind_residual.py -cf cfg.yaml 
```

### Citation
Make sure to cite the original work behind [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones).
```
@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      volume={},
      number={},
      pages={},
      doi={}
}
```
If you make use of this fork and the added wind functionality, you can additionally cite our work:
```
@inproceedings{simon2023flowdrone,
  title={FlowDrone: wind estimation and gust rejection on UAVs using fast-response hot-wire flow sensors},
  author={Simon, Nathaniel and Ren, Allen Z and Piqu{\'e}, Alexander and Snyder, David and Barretto, Daphne and Hultmark, Marcus and Majumdar, Anirudha},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5393--5399},
  year={2023},
  organization={IEEE}
}
```