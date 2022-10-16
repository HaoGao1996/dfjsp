state: 7 [U_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, Tard_e, Tard_a]
action: 6 [rule1, rule2, rule3, rule4, rule5, rule6]
reward:

## Reference

1. [Tensorflow2.0|基于深度强化学习（DQN）实现动态柔性作业车间调度问题（DFJSP）](https://blog.csdn.net/crazy_girl_me/article/details/118694032?spm=1001.2014.3001.5501)



```mermaid
classDiagram
    class Env
    Env : self.param
    Env : self.observation_space
    Env : self.action_space
    Env : reset()
    Env : step(action)
```

