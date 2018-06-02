import gym
import tensorflow as tf
import keras
from keras import backend as K #Idk why backend is required.
import numpy as np
import random
import queue
from PIL import Image #for converting to grayscale. As for now, this is not used.

#Class to store memory
class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

#function that chooses the best action:
def choose_best_action(model,state):
    best_action = np.argmax(model.predict([state, np.ones(n_actions)], verbose=1))
    return best_action
	
#modified 
def one_hot_encoding(value):
    output = np.zeros(n_actions)
    output[value] = 1
    return output
	
#The keras model:
#modified 
#idk if this should be in a function.
def Q_model(n_actions):
    # I think n_actions is the one hot coded action array. Wrong. It is the number of actions.
    # Input takes the one hot coded action array as a mask.
    # We assume a theano backend here, so the "channels" are first.
    ATARI_SHAPE = (None, 4, 105, 80) #not sure about this with tensorflow.

    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    
    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.Conv2D(16, 8, 8, subsample=(4, 4), activation='relu')(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.Conv2D(32, 4, 4, subsample=(2, 2), activation='relu')(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.merge([output, actions_input], mode='mul')

    self.model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    self.model.compile(optimizer, loss='mse')
#    return model  #Idk if this return statement is needed.

#The function to get epsilon
def get_epsilon_for_iteration(iteration):
    if(iteration>1000000):
        return 0.1
    else:
        return(1 - 9*iteration/10000000)
		
def experience_replay():
    batch = 32 #number of elements in a batch.
    batch_state_mem = RingBuf(batch)
    batch_action_mem = RingBuf(batch)
    batch_q_mem = RingBuf(batch)
    rand_int_list = []
    for i in range(batch):
        index = gen_rand()
        while index in rand_int_list:
            index = gen_rand()
        rand_int_list.append(index)
        batch_state_mem.append(state_memory._getitem_(index))
        batch_action_mem.append(action_memory._getitem_(index))
        Qcalc = reward_memory._getitem_(index) + gamma*(np.max(model.predict([next_state_memory._getitem_(index), np.ones(n_actions)], verbose=1)))
        Q_calc = Qcalc * action_memory._getitem_(index)
        batch_q_mem.append(Q_calc)
    model.fit(x=[batch_state_mem, batch_action_mem], y = batch_q_mem, verbose = 1)
	
#Preprocessing the image:
def to_grayscale(img):
    #return np.mean(img, axis=2).astype(np.uint8) #this one gives some error.
    return img.convert('L') #this hasnt worked yet.
def downsample(img):
    #return img[::2, ::2]
    return img[::2]
def preprocess(img):
    return to_grayscale(downsample(img))
	
#Function to transform the reward(modify to huber loss later).
def transform_reward(reward):
    return np.sign(reward) #signum function
	
#Function to fit a batch
def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
    """Do one deep Q learning iteration.
    
    Params:
    - model: The DQN
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal
    
    """
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    model.fit([start_states, actions], actions * Q_values[:, None], nb_epoch=1, batch_size=len(start_states), verbose=0)

#Function to generate random numbers
def gen_rand():
    return random.randint(Q_calc_memory.start, Q_calc_memory.end)
	
#The initial function which does q-iteration
# Create the environment
env = gym.make('Pong-v0') #Changing the environment is possible.
no_memory_elements=1000
#element = mem_element()
n_actions = env.action_space.n
#memory = RingBuf(no_memory_elements)
state_memory = RingBuf(no_memory_elements)
next_state_memory = RingBuf(no_memory_elements)
Q_calc_memory = RingBuf(no_memory_elements)
action_memory = RingBuf(no_memory_elements)
reward_memory = RingBuf(no_memory_elements)
state = queue.Queue(maxsize = 5)
next_state = queue.Queue(maxsize = 5)
no_episodes = 10
iteration=0 #iteration counter
gamma=0.99 #discount factor
alpha = 0.01 #learning rate
Q_model(n_actions) #Idk if this is necessary.
for i in range(no_episodes):
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    #env.render()
    for i in range (4): #to get the first 4 frames.
        action=env.action_space.sample()
        new_frame, reward, is_done, _ = env.step(action)
        new_frame= preprocess(new_frame)
        #element.state.put(new_frame)
        #element.next_state.put(new_frame)
        state.put(new_frame)
        next_state.put(new_frame)
        action1=action
    is_done = False
    count=0
    while not is_done:
        if(count%4 ==0):
            count=1
            iteration=iteration+1
            epsilon = get_epsilon_for_iteration(iteration)
            # Choose the action 
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = choose_best_action(model, state)
            new_frame, reward, is_done, _ = env.step(action)
            if(is_done):
                continue
            new_frame= preprocess(new_frame)
            reward=transform_reward(reward)
            #element.reward = reward
            #element.is_terminal = not is_done
            #element.next_state.get()
            #element.next_state.put(new_frame)
            next_state.get()
            next_state.put(new_frame)
            Qcalc = reward + gamma* (np.max(model.predict([next_state, np.ones(n_actions)], verbose=1)))
            Q_calc = Qcalc * one_hot_encoding(action)
            Q_calc_memory.append(Q_calc)
            state_memory.append(state)
            next_state_memory.append(next_state)
            reward_memory.append(reward)
            action_memory.append(one_hot_encoding(action))
            #element.action = one_hot_encoding(action)
            #memory.append(element)
            #element.state.get()
            #element.state.put(new_frame)
            state.get()
            state.put(new_frame)
            action1=action
        else:
            new_frame, reward, is_done, _ = env.step(action1)
            new_frame= preprocess(new_frame)
            #element.state.get()
            #element.state.put(new_frame)
            #element.next_state.get()
            #element.next_state.put(new_frame)
            state.get()
            state.put(new_frame)
            next_state.get()
            next_state.put(new_frame)
            count=count+1
    for i in range(4):
        #element.state.get()
        state.get()
        next_state.get()
model.fit(x=[state_memory, action_memory], y = Q_calc_memory, verbose = 1)
no_replays = 2
for j in range (no_replays):
    experience_replay()
#env.close()
#fit batches now.
#also fit in each inner loop

#The final function which plays
"""action = choose_best_action(model, state)
# Perform the best action, returns the new frame, reward and whether the game is over
frame, reward, is_done, _ = env.step(action) #(env.action_space.sample())
# Render
# env.render()"""
# Reset the environment, returning the starting frame
new_frame = env.reset()
state.put(new_frame)
# Render
env.render()
for i in range (3):
        action=env.action_space.sample()
        new_frame, reward, is_done, _ = env.step(action)
        new_frame= preprocess(new_frame)
        state.put(new_frame)
count=0
is_done = False
while not is_done:
    action = choose_best_action(model, state)
    # Perform the best action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(action)
    new_frame= preprocess(new_frame)
    state.get()
    state.put(new_frame)    
    # Render
    env.render()
env.close()