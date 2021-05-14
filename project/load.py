from environment.env_wrapper import EnvWrapper

if __name__ == '__main__':
    env = EnvWrapper()

    for i in range(500):
        print(i)
        env.load(i)

