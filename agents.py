import gym
import numpy as np
import time
import random
import tensorflow as tf
from queue import deque


class SnakeAgent():
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.action_space = gym.spaces.Discrete(env.action_space.high[id] + 1)
        self.action_size = self.action_space.n

    def get_action(self, state):
        action = self.action_space.sample()
        return action

    def play_game(self, episodes, step_cap, render_delay):
        total_reward = 0
        episode_rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            num_steps = 0
            while not done:
                self.env.render()
                action = self.get_action(state)
                time.sleep(render_delay)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                episode_rewards.append(reward)
                total_reward += reward
                if num_steps == step_cap:
                    break


class SnakeQAgent(SnakeAgent):
    def __init__(self, env, id, discount_rate=0.2, learning_rate=0.1, q_table=None, eps=1):
        super().__init__(env, id)
        self.state_size = 64
        self.eps = eps
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        if q_table is None:
            self.build_model()
        else:
            self.q_table = q_table

    def build_model(self):
        self.q_table = np.multiply(np.random.random([self.state_size, self.action_size]), np.array(
            [np.random.choice([-1, 1], self.action_size) for k in
             range(self.state_size)]))

    def get_action(self, state):
        if state is None:
            return -1
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if np.random.rand() < self.eps else action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = experience
        q_next = self.q_table[next_state]
        q_target = reward + self.discount_rate * np.max(q_next)
        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update
        if reward > 0:
            self.eps = max(0.01, 0.99 * self.eps)


class QNetwork():
    def __init__(self, state_dim, action_size, tf_model):
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim], name='state_in')
        self.flattened_state_in = tf.layers.flatten(self.state_in, name='flattened_state_in')
        self.action_in = tf.placeholder(tf.int32, shape=[None], name='action_in')
        self.q_target_in = tf.placeholder(tf.float32, shape=[None], name='target_in')
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        self.hidden1 = tf.layers.dense(inputs=self.flattened_state_in, units=125, activation=tf.nn.relu)
        self.q_state = tf.layers.dense(inputs=self.hidden1, units=action_size, activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.saver = tf.train.Saver()

    def update_model(self, session, state, action, q_target):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        session.run(self.optimizer, feed_dict=feed)

    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict={self.state_in: state})
        return q_state

    def save_network(self, session, file_name, steps):
        self.saver.save(session, file_name, global_step=steps)

    def restore_network(self, session, file_name):
        self.saver = tf.train.import_meta_graph(file_name + ".meta")
        self.saver.restore(session, file_name)


class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))


class SnakeDQNAgent(SnakeAgent):
    def __init__(self, env, id, eps=1, tf_model=None, training=False):
        super().__init__(env, id)
        self.state_dim = env.observation_space.shape
        self.training = training
        self.action_size = 3
        self.q_network = QNetwork(self.state_dim, self.action_size, tf_model=tf_model)
        self.replay_buffer = ReplayBuffer(maxlen=1)
        self.gamma = 0.2
        self.eps = eps
        self.train_steps = 0
        self.sess = tf.Session()
        if tf_model is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.q_network.restore_network(self.sess, tf_model)

    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state) % self.action_size
        action_random = np.random.randint(self.action_size)
        if random.random() < self.eps:
            action = action_random
        else:
            action = action_greedy
        return action

    def train(self, state, action, next_state, reward, done):
        if done:
            self.eps = max(0.01, 0.99 * self.eps)
            return
        self.replay_buffer.add((state, action, next_state, reward, done))
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(5)
        q_next_states = self.q_network.get_q_state(self.sess, next_states)
        q_targets = rewards + self.gamma * np.max(q_next_states, axis=1)
        self.q_network.update_model(self.sess, states, actions, q_targets)
        self.train_steps += 1

    def __del__(self):
        if self.training:
            self.q_network.save_network(self.sess, 'tf_models/latest', self.train_steps)
        self.sess.close()
