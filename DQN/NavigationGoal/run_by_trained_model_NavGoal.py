import argparse
import os
import gym
from NavGoalDQNAgent import NavGoalDQNAgent

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Navigate the environment using the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes the model should play.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes

    env = gym.make('gym_navigation:NavigationGoal-v0', render_mode="human", track_id=2)
    agent = NavGoalDQNAgent(epsilon=0)
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
                print(f'Episode: {e+1}/{play_episodes}, Time Frames: {time_frame_counter}, Total Rewards: {total_reward:.1f}, Goal distance: {next_state[5]:.2f}')
                break
            if time_frame_counter > 1000:
                print('=============================')
                print('On purpose interuption!')
                print('=============================')
                print(f'Episode: {e+1}/{play_episodes}, Time Frames: {time_frame_counter}, Total Rewards: {total_reward:.1f}, Goal distance: {next_state[5]:.2f}')
                break
            time_frame_counter += 1

            current_state = next_state
