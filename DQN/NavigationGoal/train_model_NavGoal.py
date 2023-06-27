import argparse
import gym
from collections import deque
from NavGoalDQNAgent import NavGoalDQNAgent
import csv

STARTING_EPISODE              = 1
ENDING_EPISODE                = 40
TRAINING_BATCH_SIZE           = 4 # 
SAVE_TRAINING_FREQUENCY       = 20
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to navigate the gym-navigation:NavigationGoal')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to --.')
    parser.add_argument('-p', '--epsilon', type=float, help='The starting epsilon of the agent, default to 1.0')
    args = parser.parse_args()

    env = gym.make('gym_navigation:NavigationGoal-v0', render_mode="human", track_id=2)
    agent = NavGoalDQNAgent(epsilon=args.epsilon if args.epsilon else 1.0)

    # pass the arguments
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end
   
    # results.csv creation
    if STARTING_EPISODE == 1:
        headers = ['episode', 'time-frames', 'total-rewards', 'goal-distance', 'epsilon']
        with open('./save/results.csv', 'w', newline='') as c:
            writer = csv.writer(c)
            writer.writerow(headers) 
    
    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        current_state, _ = env.reset()        

        total_reward = 0
        time_frame_counter = 1
        terminated = False

        while True:
            action = agent.act(current_state)
            next_state, reward, terminated, truncated, info = env.step(action)

            distance_diff = next_state[5] - current_state[5]           

            # Increase reward when object is heading towards goal with forward movements (in an angle roughly facing towards goal)            
            if distance_diff < 0 and abs(next_state[6]) < 0.25 and action == 0:
                reward *= 1.5 

            total_reward += reward
            
            agent.memorize(current_state, action, reward, next_state, terminated)

            if terminated:                
                print(f'Episode: {e}/{ENDING_EPISODE}, Time Frames: {time_frame_counter}, Total Rewards: {(total_reward):.1f}, Goal distance: {next_state[5]:.2f}, Epsilon: {agent.epsilon:.3f}')
                # log values in a results csv file, pattern: episode-num, time-frames, total-rewards, goal-distance, epsilon
                with open('./save/results.csv', 'a', newline='') as c:
                    writer = csv.writer(c)
                    writer.writerow([e, time_frame_counter, total_reward, next_state[5], agent.epsilon])
                break        
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)                
            time_frame_counter += 1

            current_state = next_state            
        
        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()
        
        if e % SAVE_TRAINING_FREQUENCY == 0 or (e > 80 and e % 10 == 0):
            agent.save(f'./save/trial_{e}.h5') 
            
    env.close()
