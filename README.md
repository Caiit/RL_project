# RL_project
Project for the Reinforcement Learning Course

In this project we tried to implement demonstrations as was done in
[Learning montezumaâĂŹs revenge from a single demonstration](https://blog.openai.com/learning-montezumas-revenge-from-a-single-demonstration/)
for the [Mountain-Car-v0](https://github.com/openai/gym/wiki/MountainCar-v0) environment from OpenAi gym.

The PPO2 algorithm from [openai baselines](https://github.com/openai/baselines) is adjusted in order to perform this task.
We tried extracting only the relevant files from baselines, but it turned out to be many files,
so we assume you have the baselines from openai in another folder as well as follows:
`baselines/`
`RL_Project/`


To run the code:
`python run.py --env=MountainCar-v0 --seed=42 --alg=ppo2 --num_timesteps=200000 --demo --win_percentage=0.5`

where `--demo` means that we use the demonstrations and `--win_percentage` is the percentage that has to be reached
before switching to another start state.

To run the visualisation code:
Make sure you either know where the openai logs are save (somewhere in `/tmp/` or change the log folder with:
`export OPENAI_LOGDIR=$HOME/Documents/rl/RL_project/new_logs/mountain_car/50`

We uploaded our logfiles as well to enable you to run our visualisation
without training the model several times (and saving it in the same way we did).
