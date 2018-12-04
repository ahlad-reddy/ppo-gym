# Proximal Policy Optimization (PPO) in TensorFlow for OpenAI Gym.

PPO is a policy gradient algorithm for reinforcement learning agents. PPO has a relatively simple implementation compared to other policy gradient methods. For more information on PPO, check out OpenAI's [blog](https://blog.openai.com/openai-baselines-ppo/) or their [research paper](https://arxiv.org/pdf/1707.06347.pdf).

This implementation is built in [TensorFlow](https://www.tensorflow.org/) and integrates with OpenAI's [Gym](https://github.com/openai/gym) and can be used with Pybullet environments. Much of the implementation parallels the one in [Baselines](https://github.com/openai/baselines), but is written in a much smaller codebase making it easier for newcomers to reinforcement learning and TensorFlow to understand.

## Requirements

- tensorflow
- gym
- opencv

Additional requirements are Gym Retro and Pybullet for their respective environments. (In Progress)

## Usage

The model can be trained by running

```
python ppo.py --train --env [ENVIRONMENT NAME]
```

Additional parameters and flags can be specified by consulting ppo.gym or using the `-h` flag

Results including a tensorboard file, checkpoint files, and a video (post-training) will be generated in the logdir folder. 

The model can be evaluated by specifying the checkpoint file and exempting the `--train` flag

```
python ppo.py --env [ENVIRONMENT NAME] --model_path [/path/to/model]
```

## Results

Sample results including tensorboard files, checkpoints, and videos for various environments can be found in this [drive](https://drive.google.com/open?id=1loDFC9sYeuiTQXSbR94iHEH3A-P2ihcy) folder. More to come!

## To Do

- Test more benchmarks
- Integrate Retro
- Create default hyperparameters for different environment types
- Replace OpenCV dependency?
