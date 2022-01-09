import numpy as np
import gym
from gym import Env, spaces
from gym.envs.classic_control import rendering
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Lambda, Activation, BatchNormalization
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

import numpy as np
import tensorflow as tf
from tensorflow import keras

NUM_AGENTS = 5
LEN_HISTORY = 1     # how many prev time steps in the observations
LEN_EPISODE = 150   # number of steps in one game
LEARNING_RATE = 0.01
MOMENTUM = 0.9
X_MIN = -100
X_MAX = 100
Y_MIN = X_MIN
Y_MAX = X_MAX
SAVE_MODEL = True
USE_PRETRAINED_MODEL = False
MODE = "human"

class FiveAgents(Env):
    def __init__(self):
        super(FiveAgents, self).__init__()

        # Define a 3-D observation space: 10 (for time-steps) by 5 (the number of agents) by 2 (x and y coordinates)
        self.observation_shape = (LEN_HISTORY, NUM_AGENTS, 2)

        self.observation_space = spaces.Box(low = -X_MIN, high = X_MAX, shape = self.observation_shape, dtype = np.float32)

        # Define an action space for
        self.action_space = spaces.Discrete(4,)

        # Keeps track of previous states
        self.previous_positions = np.zeros(self.observation_shape)

        # Keeps track of graphics
        self.viewer = None

    # Create a separate file to run this program
    # Step function -> control input policy, to visualize robots moving

    def step(self, action):
        current_positions = self.previous_positions[0, :, :]

        # Calculate centroid
        current_centroid = current_positions.mean(axis=0)

        # Find which direction good agents have to go to get to centroid
        directions = current_centroid - current_positions
        distances = np.sqrt(np.sum(directions * directions, axis = 1, keepdims = True))
        move = directions/distances

        # Move good agents to their position
        new_positions = current_positions + move

        # Using action, move malicious agent to its new position
        if action == 0:
            new_positions[NUM_AGENTS-1, :] = current_positions[NUM_AGENTS-1,:] + [0, 1]
        elif action == 1:
            new_positions[NUM_AGENTS-1, :] = current_positions[NUM_AGENTS-1,:] + [0, -1]
        elif action == 2:
            new_positions[NUM_AGENTS-1, :] = current_positions[NUM_AGENTS-1,:] + [1, 0]
        elif action == 3:
            new_positions[NUM_AGENTS-1, :] = current_positions[NUM_AGENTS-1,:] + [-1, 0]
        else:
            print("error")

        # new_positions[:, 0] = np.maximum(np.minimum(new_positions[:, 0], X_MAX), X_MIN)
        # new_positions[:, 1] = np.maximum(np.minimum(new_positions[:, 1], Y_MAX), Y_MIN)

        # Update previous positions
        self.previous_positions[1:, :, :] = self.previous_positions[:-1, :, :]
        self.previous_positions[0, :, :] = new_positions

        return self.previous_positions, np.all(distances[:-1] < 5)

    def reset(self):
        intial_positions = np.random.uniform(low=X_MIN/2, high=X_MAX/2, size=(NUM_AGENTS,2))
        self.previous_positions = np.full(self.observation_shape, intial_positions)

    def render(self, mode = "computer"):

        if self.viewer is None:
            self.viewer = rendering.Viewer(X_MAX-X_MIN, Y_MAX-Y_MIN)
            self.viewer.set_bounds(X_MIN, X_MAX, Y_MIN, Y_MAX)

            # create empty lists of the right size
            agent = [None]*NUM_AGENTS
            self.agent_transform = [None]*NUM_AGENTS

            # make a list of agents later
            for i in range(NUM_AGENTS):
                agent[i] = rendering.make_circle(5)
                agent[i].set_color(0,0,0)
                self.agent_transform[i] = rendering.Transform()
                agent[i].add_attr(self.agent_transform[i])
                self.viewer.add_geom(agent[i])

            agent[NUM_AGENTS-1].set_color(1, 0, 0)

            self.centroid = rendering.make_circle(5)
            self.centroid.set_color(0, 0, 1)
            self.centroid_transform = rendering.Transform()
            self.centroid.add_attr(self.centroid_transform)

        current_positions = self.previous_positions[0, :, :]
        current_centroid = current_positions.mean(axis=0)

        self.viewer.add_onetime(self.centroid)
        for i in range(NUM_AGENTS):
            self.agent_transform[i].set_translation(current_positions[i, 0], current_positions[i, 1])
        self.centroid_transform.set_translation(current_centroid[0], current_centroid[1])

        if mode == "human":
            self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()

def make_malicious_agent_network():
    network = Sequential()

    network.add(Reshape((2*NUM_AGENTS*LEN_HISTORY,), input_shape=(LEN_HISTORY, NUM_AGENTS, 2)))
    # network.add(Dense(128, activation='relu'))
    # network.add(Dense(128, activation='relu'))
    network.add(BatchNormalization())
    network.add(Dense(64, activation='LeakyReLU'))
    network.add(BatchNormalization())
    network.add(Dense(32, activation='LeakyReLU'))
    network.add(BatchNormalization())
    network.add(Dense(10, activation='LeakyReLU'))
    network.add(BatchNormalization())
    network.add(Dense(4, activation='LeakyReLU'))
    network.add(BatchNormalization())
    network.add(Dense(4, activation='softmax'))

    return network

def train_on_one_game(model, mode = "computer"):
    env = FiveAgents()
    obs = env.reset()
    opt = SGD(learning_rate = LEARNING_RATE, momentum = MOMENTUM)

    observed_positions = []
    rendezvous = False
    rendezvous_time = -1

    actions_performed = []

    for count in range(LEN_EPISODE):
        # Take a random action
        obs = env.previous_positions
        observed_positions.append(np.copy(obs))
        predicted_action_probabilities = np.squeeze(model(observed_positions[-1][np.newaxis,:]))
        # print(predicted_action_probabilities)
#        print(obs)

        # Going to pick a number from to 0 to 3 based on the network
        action = np.random.choice(4, p = predicted_action_probabilities)
        _, rendezvous = env.step(action)

        if rendezvous and rendezvous_time < 0:
            rendezvous_time = count

        actions_performed.append(action)

        # Render the game
        env.render(mode)

    print("******* RENDEZVOUS TIME ********: ", rendezvous_time)
    env.close()

    # Find out if actions from game were good or bad
    advantage = 1 if not rendezvous else - (LEN_EPISODE - rendezvous_time)/LEN_EPISODE
    #print("ADVANTAGE:", advantage)

    all_positions = tf.Variable(np.stack(observed_positions))
    with tf.GradientTape() as tape:
        p = model(all_positions)
        #print(p)

        # want to maximize the objective so gradient descent on opposite
        loss = -1 * advantage * tf.math.reduce_sum(tf.math.log(tf.math.maximum(p, tf.constant(1e-9))) * to_categorical(actions_performed, 4))

    g = tape.gradient(loss, model.trainable_weights)
    #print(g)
    opt.apply_gradients(zip(g, model.trainable_weights))

    if SAVE_MODEL:
        model.save("~/models/model_250.model")

    return model

"""
model1 = keras.models.load_model('~/models/model_2000.model')
for game_number in range(10):
    train_on_one_game(model1, mode = "human")
"""

if __name__=="__main__":

    if USE_PRETRAINED_MODEL:
        x = keras.models.load_model('~/models/model_2000_ab.model')
    else:
        x = make_malicious_agent_network()
        x.summary()

    for game_number in range(250):
        new_x = train_on_one_game(x, mode = MODE)
        if not USE_PRETRAINED_MODEL:
            x = new_x
        #print(game_number)