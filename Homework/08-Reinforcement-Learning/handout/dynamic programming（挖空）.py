import gym # openAi gym
import numpy as np 

import warnings
warnings.filterwarnings('ignore')

env = gym.make("FrozenLake-v0")
env.reset()

def policy_evaluation(policy, env, gamma=1.0, theta=0.00001):
  """
  实现策略评估算法，给定策略与环境模型，计算该策略对应的价值函数。

  参数：
    policy：维度为[S, A]的矩阵，用于表示策略。
    env：gym环境，其env.P表示了环境的转移概率。
      env.P[s][a]为一个列表，其每个元素为一个表示转移概率以及奖励函数的元组(prob, next_state, reward, done)
      env.observation_space.n表示环境的状态数。
      env.action_space.n表示环境的动作数。
    gamma：折扣因子。
    theta：用于判定评估是否停止的阈值。
  
  返回值：长度为env.observation_space.n的数组，用于表示各状态的价值。
  """
  
  nS = env.observation_space.n
  nA = env.action_space.n

  # 初始化价值函数
  V = np.zeros(nS)
  while True:
    delta = 0
    for s in range(nS):
      v_new = 0
      for a in range(nA):
        for prob, next_state, reward, done in env.P[s][a]:
          # TODO: 计算用于更新 V[s] 的 v_new
          
      delta = max(delta, np.abs(V[s]-v_new))
      V[s] = v_new
    # 误差小于阈值时终止计算
    if delta < theta:
      break
    
  return np.array(V)

def policy_iteration(env, policy_eval_fn=policy_evaluation, gamma=1.0):
  """
  实现策略提升算法，迭代地评估并提升策略，直到收敛至最优策略。

  参数：
    env：gym环境。
    policy_eval_fn：策略评估函数。
    gamma：折扣因子。

  返回值：
    (policy, V)
    policy为最优策略，由维度为[S, A]的矩阵进行表示。
    V为最优策略对应的价值函数。
  """

  nS = env.observation_space.n
  nA = env.action_space.n

  def one_step_lookahead(state, V):
    """
    对于给定状态，计算各个动作对应的价值。
    
    参数：
        state：给定的状态 (int)。
        V：状态价值，长度为env.observation_space.n的数组。
    
    返回值：
        每个动作对应的期望价值，长度为env.action_space.n的数组。
    """
    A = np.zeros(nA)
    for a in range(nA):
        for prob, next_state, reward, done in env.P[state][a]:
            # TODO: 计算动作 a 的价值 A[a]

    return A

  # 初始化为随机策略
  policy = np.ones([nS, nA]) / nA
  
  num_iterations = 0

  while True:
      num_iterations += 1
      
      V = policy_eval_fn(policy, env, gamma)
      policy_stable = True
      
      for s in range(nS):
          old_action = np.argmax(policy[s])

          q_values = one_step_lookahead(s, V)
          new_action = np.argmax(q_values)

          if old_action != new_action:
              policy_stable = False
                      
          policy[s] = np.zeros([nA])
          policy[s][new_action] = 1

      if policy_stable:
          print(num_iterations)
          return policy, V
  

env.reset()
policyPI, valuePI = policy_iteration(env, gamma=0.95)
# print(policyPI)
# print(valuePI)

def value_iteration(env, theta=0.0001, gamma=1.0):
  """
  实现价值迭代算法。
  
  参数：
    env：gym环境，其env.P表示了环境的转移概率。
      env.P[s][a]为一个列表，其每个元素为一个表示转移概率以及奖励函数的元组(prob, next_state, reward, done)
      env.observation_space.n表示环境的状态数。
      env.action_space.n表示环境的动作数。
    gamma：折扣因子。
    theta：用于判定评估是否停止的阈值。
      
  返回值：
    (policy, V)
    policy为最优策略，由维度为[S, A]的矩阵进行表示。
    V为最优策略对应的价值函数。       
  """

  nS = env.observation_space.n
  nA = env.action_space.n
  
  def one_step_lookahead(state, V):
    """
    对于给定状态，计算各个动作对应的价值。
    
    参数：
        state：给定的状态 (int)。
        V：状态价值，长度为env.observation_space.n的数组。
    
    返回值：
        每个动作对应的期望价值，长度为env.action_space.n的数组。
    """
    A = np.zeros(nA)
    for a in range(nA):
        for prob, next_state, reward, done in env.P[state][a]:
            # TODO: 计算动作 a 的价值 A[a]

    return A
  
  V = np.zeros(nS)
  
  num_iterations = 0
  
  while True:
      num_iterations += 1
      delta = 0
      
      for s in range(nS):
          q_values = one_step_lookahead(s, V)
          new_value = np.max(q_values)
          
          delta = max(delta, np.abs(new_value - V[s]))
          V[s] = new_value
      
      if delta < theta:
          break
  
  policy = np.zeros([nS, nA])
  for s in range(nS): 
      q_values = one_step_lookahead(s,V)
      
      new_action = np.argmax(q_values)
      policy[s][new_action] = 1
  
  print(num_iterations)    
  return policy, V

env.reset()
policyVI,valueVI = value_iteration(env, gamma=0.95)
# print(policyVI)
# print(valueVI)

nS = env.observation_space.n
nA = env.action_space.n

samePolicy = True
for s in range(nS):
    if samePolicy == False:
        break
    for a in range(nA):
        if policyPI[s][a] != policyVI[s][a]:
            samePolicy=False
            break

if samePolicy:
    print("策略迭代算法与价值迭代算法的最终策略一致。")
else:
    print("策略迭代算法与价值迭代算法的最终策略不一致。")

