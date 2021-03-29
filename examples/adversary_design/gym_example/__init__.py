from gym.envs.registration import register

register(
    id="adversarial-v0",
    entry_point="gym_example.envs:Adversarial_v0",
)

register(
    id="adversarial-v1",
    entry_point="gym_example.envs:Adversarial_v1",
)

