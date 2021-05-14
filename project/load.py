from environment.env_wrapper import EnvWrapper
from environment.level_loader import load_level

if __name__ == '__main__':
    env = EnvWrapper()

    for i in range(100):
        print(i)
        env.load(i)

