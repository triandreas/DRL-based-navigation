import argparse
import gym
from NavGoalDQNAgent import NavGoalDQNAgent
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
    train_model = args.model    
    render = args.render
    new_file = args.newfile

    if args.start:
        START_EPS = args.start
    if args.end:
        END_EPS = args.end

    if render == True:
        env = gym.make('gym_navigation:NavigationGoal-v0', render_mode="human", track_id=2)
    else:
        env = gym.make('gym_navigation:NavigationGoal-v0', track_id=2)
    
    if train_model == "random":
        agent = NavGoalDQNAgent(epsilon=1)
    else:
        agent = NavGoalDQNAgent(epsilon=0)
        agent.load(train_model)

    if new_file == True:
        headers = ['model','episode', 'timeframes', 'totalrewards', 'success', 'goaldist', 'initdist']
        with open('./performance.csv', 'w', newline='') as c:
            writer = csv.writer(c)
            writer.writerow(headers)

    for e in range(START_EPS, END_EPS+1):
        current_state, _ = env.reset()
	
	    # initial distance from goal
        init_dist = current_state[5]

        total_reward = 0
        time_frame_counter = 1

        while True:
            action = agent.act(current_state)
            next_state, reward, terminated, truncated, info = env.step(action)

            success = 0
            goal_distance = next_state[5]
            total_reward += reward

            if terminated:
                if reward == 200: 
                    success = 1                                
                with open('./performance.csv', 'a', newline='') as c:
                    writer = csv.writer(c)
                    writer.writerow([train_model.replace('.h5', ''), e, time_frame_counter, total_reward, success, goal_distance, init_dist])
                break
            if time_frame_counter >= 1000:
                success = -1
                total_reward -= 200
                with open('./performance.csv', 'a', newline='') as c:
                    writer = csv.writer(c)
                    writer.writerow([train_model.replace('.h5', ''), e, time_frame_counter, total_reward, success, goal_distance, init_dist])
                break
            time_frame_counter += 1

            current_state = next_state
