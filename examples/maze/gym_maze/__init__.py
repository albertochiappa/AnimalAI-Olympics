from gym.envs.registration import register

register(
    id='MazeEnv-v0',
    entry_point='gym_maze.envs:MazeEnv',
)

register(
    id='MazeEnv-v1',
    entry_point='gym_maze.envs:MazeEnv2',
)

register(
    id='BasicMultiAgent-v0',
    entry_point='gym_maze.envs:BasicMultiAgent',
)

register(
    id='AlternateMultiAgent-v0',
    entry_point='gym_maze.envs:AlternateMultiAgent',
)


register(
    id='Adversary-v0',
    entry_point='gym_maze.envs:Adversary',
)

register(
    id='AdvPro-v0',
    entry_point='gym_maze.envs:AdvPro',
)

register(
    id='AdvPro-v1',
    entry_point='gym_maze.envs:AdvPro2',
)

register(
    id='Paired-v0',
    entry_point='gym_maze.envs:PAIRED',
)