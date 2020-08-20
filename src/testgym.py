import gym
import roboschool
import time
env = gym.make("RoboschoolHalfCheetah2-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  time.sleep(1.3)

  if done:
    observation = env.reset()
env.close()
