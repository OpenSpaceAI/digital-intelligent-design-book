import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class DeepSpaceProbeEnv(gym.Env):
    """
    一个简化的深空探测器行星表面科考任务环境。
    继承自 gym.Env。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DeepSpaceProbeEnv, self).__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.grid_size, self.grid_size])
        self.agent_pos = [0, 0]
        self.goal_pos = [4, 4]
        self.hazard_pos = [2, 2]
        
    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.agent_pos = [0, 0]
        return np.array(self.agent_pos), {}

    def step(self, action):
        if action == 0:  # 上
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # 下
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # 左
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # 右
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)

        terminated = False
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 100
            terminated = True
        elif np.array_equal(self.agent_pos, self.hazard_pos):
            reward = -100
            terminated = True
        else:
            reward = -1
            
        info = {}
        truncated = False
        return np.array(self.agent_pos), reward, terminated, truncated, info

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '_', dtype=str)
        grid[tuple(self.goal_pos)] = 'G'
        grid[tuple(self.hazard_pos)] = 'H'
        grid[tuple(self.agent_pos)] = 'A'
        
        for row in grid:
            print(' '.join(row))
        print("-" * 10)

# 创建环境实例
env = DeepSpaceProbeEnv()

# Q-learning 参数
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
episodes = 20000

# 初始化Q表格
q_table = np.zeros([env.grid_size, env.grid_size, env.action_space.n])

# 用于记录每个回合奖励的列表
rewards_per_episode = []

# 训练循环
for i in range(episodes):
    state, info = env.reset()
    state_tuple = tuple(state)
    done = False
    total_episode_reward = 0
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_tuple])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state_tuple = tuple(next_state)
        
        total_episode_reward += reward

        old_value = q_table[state_tuple + (action,)]
        next_max = np.max(q_table[next_state_tuple])

        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state_tuple + (action,)] = new_value

        state = next_state
        state_tuple = tuple(state)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        
    rewards_per_episode.append(total_episode_reward)
        
    if (i + 1) % 5000 == 0:
        print(f"Episode {i + 1}/{episodes} | Epsilon: {epsilon:.4f}")

print("\n--- 训练完成 ---")

# 验证学习到的策略
print("\n--- 验证最优策略 ---")
state, info = env.reset()
env.render()
done = False
total_reward = 0
while not done:
    action = np.argmax(q_table[tuple(state)])
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    env.render()

print(f"任务完成! 总奖励: {total_reward}")