from core.utils import create_image_dataset
num_demos = [10, 50, 150, 200, 250]
envs = ['MiniWorld-OneRoom-v0', 'MiniWorld-Hallway-v0', 'MiniWorld-YMaze-v0', 'MiniWorld-FourRooms-v0']
seed = 42
for env in envs:
    for demo in num_demos:
        create_image_dataset(env, 'agent', num_demos=demo, seed=seed)
        if env != "MiniWorld-Hallway-v0":
            create_image_dataset(env, 'top', num_demos=demo, seed=seed)
print('done')