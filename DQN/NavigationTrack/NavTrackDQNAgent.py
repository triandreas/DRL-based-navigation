import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class NavTrackDQNAgent:
    def __init__(
        self,
        action_space = [0, 1, 2],
        gamma = 0.95,
        epsilon = 1.0,
        epsilon_min = 0.1,
        epsilon_decay = 0.9995,
        learning_rate = 0.001,
        memory_size = 2500,
    ):
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        # Builds the neural net for the Deep-Q learning model                
        model = Sequential([
            Input(shape = 5),
            Dense(32, activation = "relu"),
            Dense(64, activation = "relu"),
            Dense(len(self.action_space), activation = "linear")
        ])
        model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())   

    def act(self, state):
        if np.random.rand() > self.epsilon:            
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(len(self.action_space))        
        return self.action_space[action_index]
    
    def memorize(self, state, action, reward, next_state, terminated):
        self.memory.append((state, self.action_space.index(action), reward, next_state, terminated))
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, terminated in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if terminated:
                target[action_index] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)