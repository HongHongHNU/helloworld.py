import gym
from gym.utils.env_checker import check_env

env = gym.make('CartPole-v1', render_mode="human")
env.unwrapped
env.reset()
check_env(env, skip_render_check=False)
for _ in range(1000):
    env.render()

    env.step(env.action_space.sample())  # take a random action
env.close()

print('hello world')
print('try_2')
