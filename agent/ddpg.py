from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.utils import *
from agent import Agent
from dm_control.rl.control import flatten_observation

class DDPGagent(Agent):

	def __init__(self, env, actor, critic, actor_t, critic_t, replay_buffer, noise, session, pms, saver = None):
		super(DDPGagent, self).__init__(env, session, pms, saver)
		self.actor = actor
		self.critic = critic
		self.actor_t = actor_t
		self.critic_t = critic_t
		self.replay_buffer = replay_buffer
		self.noise = noise
		self.init_vars()
		print('Building Network')

	def init_vars(self):
		with tf.name_scope(self.pms.name_scope):
			
			self.critic_q = self.critic.output
			self.critic_target_q = self.critic_t.output
			self.state_ph = self.actor.input_ph
			self.action_ph = self.critic.action_ph
			self.actor_action = self.actor.output_net
			self.actor_target_action = self.actor_t.output_net

			self.training_q = tf.placeholder(tf.float32, [None, 1], 'training_q')
			self.critic_g_ph = tf.placeholder(tf.float32, [None, self.pms.action_shape], 'critic_g_ph')

			self.actor_optimizer = tf.train.AdamOptimizer(learning_rate = self.pms.actor_learning_rate, name = 'actor_adam')
			self.critic_optimizer = tf.train.AdamOptimizer(learning_rate = self.pms.critic_learning_rate, name = 'critic_adam')

			self.actor_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.actor.net.name)]
			self.actor_t_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.actor_t.net.name)]
			self.critic_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.critic.name)]
			self.critic_t_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.critic_t.name)]

			self.critic_loss = tf.losses.mean_squared_error( self.training_q, self.critic_q)
			self.critic_tmp_g = tf.gradients(self.critic_loss, self.critic_var_list)
			self.critic_train_op = self.critic_optimizer.apply_gradients( zip(self.critic_tmp_g, self.critic_var_list))
			self.critic_g = tf.gradients(self.critic_q, self.action_ph)
			self.update_critic_paras = [target_param.assign( tf.multiply( target_param, 1. - self.pms.tau) + tf.multiply(network_param, self.pms.tau)) \
				for target_param, network_param in zip(self.critic_t_var_list, self.critic_var_list) ]

			self.actor_g = tf.gradients(self.actor_action, self.actor_var_list, grad_ys = -self.critic_g_ph)
			self.actor_train_op  = self.actor_optimizer.apply_gradients( zip( self.actor_g, self.actor_var_list) )
			self.update_actor_paras = [target_param.assign( tf.multiply( target_param, 1. - self.pms.tau) + tf.multiply(network_param, self.pms.tau)) \
				for target_param, network_param in zip(self.actor_t_var_list, self.actor_var_list) ]


	def step(self, state):
		action = self.actor.get_action(state)
		action = [action] if len(np.shape(action)) == 0 else action
		action = np.array(action) + self.noise.noise() 
		# next_state, reward, terminal, _ = self.env.step(action)
		time_step, reward, _, next_state = self.env.step(action)
		terminal = time_step.last()
		next_state = flatten_observation(next_state)
		next_state = next_state.values()[0]
		self.replay_buffer.add_sample(state, action, reward, next_state, terminal)
		if self.pms.render:
			self.env.render()
		return state, action, reward, next_state, terminal 

	def draw_minibatch(self, batchsize = None):
		mini_batch = self.replay_buffer.rand_sample(batchsize)
		return mini_batch

	def train_actor(self, s_batch):
		train_action_batch = self.sess.run(self.actor_action, feed_dict = {self.state_ph: s_batch})
		critic_grad = self.sess.run(self.critic_g, feed_dict = {self.state_ph: s_batch, self.action_ph:train_action_batch})
		feed_dict = {self.state_ph: s_batch, self.critic_g_ph: np.reshape(critic_grad, [-1, self.pms.action_shape])}
		return self.sess.run(self.actor_train_op, feed_dict = feed_dict)
	
	def train_critic(self, s_batch, a_batch, r_batch, t_batch):
		a2_batch = self.sess.run(self.actor_target_action, feed_dict = {self.state_ph: s_batch})
		predict_q = self.sess.run(self.critic_target_q, feed_dict = {self.state_ph: s_batch, self.action_ph: a2_batch})
		training_q = r_batch + self.pms.discount * predict_q * ~t_batch
		feed_dict = {self.state_ph: s_batch, self.action_ph: a_batch, self.training_q: training_q}
		return self.sess.run([self.critic_train_op, self.critic_loss], feed_dict = feed_dict)
	
	def train_minibatch(self, minibatch):
		s_batch, a_batch, r_batch, s2_batch, t_batch = minibatch
	
		_, loss = self.train_critic(s_batch, a_batch, r_batch, t_batch)
		_ = self.train_actor(s_batch)

		self.sess.run([self.update_actor_paras, self.update_critic_paras])
		return loss

	def learn(self, verbose = True):
		
		dict_keys = ['total_return', 'time_step', 'critic_loss', 'end_iter_num', 'sample_time', 'train_time']
		saving_result = dict([(v, []) for v in dict_keys])
		state = self.env.reset()
		state = flatten_observation(state[-1])
		state = state.values()[0]
		iter_num = 0
		episode_num = 0
		while iter_num < self.pms.max_iter:
			reward_list = []
			time_step = 0
			critic_loss = []
			sample_time = 0
			train_time = 0
			for _ in range(self.pms.max_time_step):
				iter_num += 1
				t = time.time()
				_, _ , reward , next_state ,terminate = self.step(state)
				sample_time += time.time() - t

				reward_list.append(reward)
				if iter_num < self.pms.warm_up_size:
					if terminate:
						state = self.env.reset()
						break
					state = next_state
					continue

				t = time.time()
				mini_batch = self.draw_minibatch(self.pms.batchsize)
				critic_loss.append( self.train_minibatch(mini_batch) )
				train_time += time.time() - t
				time_step += 1
				if terminate:
					state = self.env.reset()
					state = flatten_observation(state[-1])
					state = state.values()[0]
					break
				state = next_state
			episode_num += 1
			save_value_list = [np.sum(reward_list), time_step, critic_loss, iter_num, sample_time, train_time]
			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

			if episode_num % self.pms.save_model_iters == 0:
				print('\n******************* Episode %i *******************'%episode_num)
				print("%-20s: %15.5f"%('total_return', save_value_list[0]))
				print("%-20s: %15.5f"%('time_step', save_value_list[1]))
				print("%-20s: %15.5f"%('sample_time', sample_time))
				print("%-20s: %15.5f"%('train_time', train_time))
				if self.pms.save_model:
					self.save_model(self.pms.save_dir + self.pms.env_name + '-episode%i'%(episode_num))
		return saving_result
