import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Navigate the environment using the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes the model should play.')
    args = parser.parse_args()
    trained_model = args.model
    play_episodes = args.episodes

    env = gym.make('gym_navigation:NavigationGoal-v0', render_mode="human", track_id=2)
    observation_dimensions = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # the sizes of the neural network
    hidden_sizes = [64, 128]
    sizes = list(hidden_sizes) + [num_actions]

    # initialize the nn
    observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
    mlp = observation_input
    for size in sizes[:-1]:
        mlp = layers.Dense(units=size, activation=tf.tanh)(mlp)
    mlp = layers.Dense(units=sizes[-1], activation=None)(mlp)        
    actor = keras.Model(inputs=observation_input, outputs=mlp)
    
    # load the weights of the trained actor from file
    actor.load_weights(trained_model)
   
    for e in range(play_episodes):
        current_state, _ = env.reset()
        
        init_dist_from_goal = current_state[5]
        
        total_reward = 0
        time_frame_counter = 1

        while True:
            current_state = current_state.reshape(1, -1)
            logits = actor(current_state)
            action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
            next_state, reward, terminated, truncated, info = env.step(action[0].numpy())

            total_reward += reward

            if terminated:
                print(f'Episode: {e+1}/{play_episodes}, Time Frames: {time_frame_counter}, Total Rewards: {total_reward:.1f}, Goal distance: {next_state[5]:.2f}')
                break
            if time_frame_counter > 1000:
                # print('=========================')
                # print('On purpose interuption!')
                # print('=========================')
                print(f'Episode: {e+1}/{play_episodes}, Time Frames: {time_frame_counter}, Total Rewards: {total_reward:.1f}, Goal distance: {next_state[5]:.2f}')
                break
            time_frame_counter += 1

            current_state = next_state
