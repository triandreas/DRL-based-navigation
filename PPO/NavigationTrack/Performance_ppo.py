import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import csv

START_EPS             = 1
END_EPS               = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a session of episodes using the given trained model, and output the performance in a csv file.')
    parser.add_argument('-m', '--model', required=True, help='The `.h5` file of the trained model.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 100.')
    parser.add_argument('-r', '--render', type=bool, default=True, help='Set False if you dont want to render the environment.')
    parser.add_argument('-nf', '--newfile', type=bool, default=False, help='If the program should write to a new file.')
    args = parser.parse_args()
    trained_model = args.model    
    render = args.render
    new_file = args.newfile

    if args.start:
        START_EPS = args.start
    if args.end:
        END_EPS = args.end

    if render == True:
        env = gym.make('gym_navigation:NavigationTrack-v0', render_mode="human", track_id=1)
    else:
        env = gym.make('gym_navigation:NavigationTrack-v0', track_id=1)
    
    if new_file == True:
        headers = ['model','episode', 'timeframes', 'totalrewards', 'success']
        with open('./performance.csv', 'w', newline='') as c:
            writer = csv.writer(c)
            writer.writerow(headers)

    observation_dimensions = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # the sizes of the neural network
    hidden_sizes = [32, 64]
    sizes = list(hidden_sizes) + [num_actions]

    # initialize the nn
    observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
    mlp = observation_input
    for size in sizes[:-1]:
        mlp = layers.Dense(units=size, activation='relu')(mlp)
    mlp = layers.Dense(units=sizes[-1], activation='linear')(mlp)        
    actor = keras.Model(inputs=observation_input, outputs=mlp)
    
    # load the weights of the trained actor from file
    actor.load_weights(trained_model)

    for e in range(START_EPS, END_EPS+1):
        current_state, _ = env.reset()

        total_reward = 0
        time_frame_counter = 1

        while True:
            current_state = current_state.reshape(1, -1)
            logits = actor(current_state)
            action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
            next_state, reward, terminated, truncated, info = env.step(action[0].numpy())

            success = 0            
            total_reward += reward

            if terminated:                                              
                with open('./performance.csv', 'a', newline='') as c:
                    writer = csv.writer(c)
                    writer.writerow([trained_model.replace('.h5', '').replace('actor_',''), e, time_frame_counter, total_reward, success])
                break
            if time_frame_counter >= 1000:
                success = 1                
                with open('./performance.csv', 'a', newline='') as c:
                    writer = csv.writer(c)
                    writer.writerow([trained_model.replace('.h5', '').replace('actor_',''), e, time_frame_counter, total_reward, success])
                break
            time_frame_counter += 1

            current_state = next_state
