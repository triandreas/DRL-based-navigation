import argparse
import gym
from NavTrackDQNAgent import NavTrackDQNAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Navigate the environment using the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes the model should play.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes

    env = gym.make('gym_navigation:NavigationTrack-v0', render_mode="human", track_id=1)
    if train_model == "random":
        agent = NavTrackDQNAgent(epsilon=1)
    else:
        agent = NavTrackDQNAgent(epsilon=0)
        agent.load(train_model)

    for e in range(play_episodes):
        current_state, _ = env.reset()

        total_reward = 0
        time_frame_counter = 1

        while True:
            action = agent.act(current_state)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated:
                print(f'Episode: {e+1}/{play_episodes}, Scores(Time Frames): {time_frame_counter}, Total Rewards: {total_reward:.1f}')
                break
            if time_frame_counter > 1000:
                print('=============================================================================')
                print('Agent could not stop being perfect... (or confused)!! On purpose interuption!')
                print('=============================================================================')
                print(f'Episode: {e+1}/{play_episodes}, Scores(Time Frames): {time_frame_counter}, Total Rewards: {total_reward:.1f}')
                break
            time_frame_counter += 1

            current_state = next_state