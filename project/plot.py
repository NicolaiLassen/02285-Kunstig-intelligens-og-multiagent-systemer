import pickle

if __name__ == '__main__':
    with open('ckpt/reward_level_ckpt.pickle', 'rb') as handle:
        b = pickle.load(handle)
    print(b)