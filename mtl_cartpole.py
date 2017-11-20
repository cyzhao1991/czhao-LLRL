from __future__ import print_function
import tensorflow as tf
import numpy as np
import model.knowledge_base as mykb
from env.cartpole import CartPoleEnv
import os, pdb, gym, shelve
from gym import wrappers
from utils.replay_buffer import ReplayBuffer
from utils.ounoise import OUNoise

LAYER_SHAPE = [300,300]
NUM_OF_LATENT = 10
LEARNING_RATE_ACTOR_KB = 0.0001
LEARNING_RATE_CRITIC_KB = 0.001
LEARNING_RATE_ACTOR_s = 0.0001
LEARNING_RATE_CRITIC_s = 0.001
ALPHA_L2 = 0.01
ALPHA_L1 = 0.1
TAU_ACTOR = 0.001
TAU_CRITIC = 0.001
INITIALIZE_REPLAY_BUFFER = 1000
MAX_TIME = 200
BATCH_SIZE = 64
MAX_ITER = 100000
GAMMA = 0.99
NUM_OF_TASK = 5
SEED = 1992

m_cart_list = np.random.rand(NUM_OF_TASK) * 4.5 + 0.5
m_pole_list = np.random.rand(NUM_OF_TASK) * 0.9 + 0.1
l_pole_list = np.random.rand(NUM_OF_TASK) * 1.0 + 0.5
g_bool_list = np.random.randint(2, size = NUM_OF_TASK)
g_doub_list = g_bool_list * 9.81

gen_env_list = [(CartPoleEnv(g, mc, mp, size_pole = (lp, .1)), np.array([mc, mp, lp, g/9.81]))for mc, mp, lp, g in zip(m_cart_list, m_pole_list, l_pole_list, g_doub_list)]
env_list, context_list = zip(*gen_env_list)
input_dim = env_list[0].observation_space.shape[0]
context_dim = 4
output_dim = env_list[0].action_space.shape[0]
action_low_bound = env_list[0].action_space.low
action_high_bound = env_list[0].action_space.high

num_of_task = len(env_list) 
## Building network
print('Building Network')
context_input = [tf.placeholder(tf.float32, [ None, context_dim], name = 'context_input%i'%(i)) for i in range(num_of_task)]
action_input = [tf.placeholder(tf.float32, [ None, output_dim], name = 'action_input%i'%(i)) for i in range(num_of_task)]
pre_defined_context = [tf.concat([u,v], axis = 1) for u,v in zip(context_input, action_input)]

# context_input = tf.placeholder(tf.float32, [None, context_dim], name = 'context_input')
# action_input = tf.placeholder(tf.float32, [None, output_dim], name = 'action_input')
# pre_defined_context = tf.concat([context_input, action_input], axis = 1)
# target_context_input = [tf.placeholder(tf.float32, [None, context_dim]) for _ in range(num_of_task)]
# target_action_input = [tf.placeholder(tf.float32, [None, output_dim]) for _ in range(num_of_task)]

L_actor = mykb.KnowledgeBase(LAYER_SHAPE, NUM_OF_LATENT, name = 'cartpole_KB_actor', seed = SEED)
L_critic = mykb.KnowledgeBase(LAYER_SHAPE, NUM_OF_LATENT, name = 'cartpole_KB_critic', seed = SEED)
s_actor_list = [mykb.PathPolicy(input_dim, output_dim, knowledge_base = L_actor, name = 'cartpole_s_actor_task%i'%(i), seed = SEED, \
	allow_context = True, context_dim = context_dim, context_layer = 0, final_layer_act_function = 'tanh') for i in range(num_of_task)]
## critic takes input action along with context
s_critic_list = [mykb.PathPolicy(input_dim, 1, knowledge_base = L_critic, name = 'cartpole_s_critic_task%i'%(i), seed = SEED, \
	allow_context = True, context_dim = output_dim + context_dim, context_layer = 0, pre_defined_context = pre_defined_context[i]) for i in range(num_of_task)]
rb_list = [ReplayBuffer() for i in range(num_of_task)]
on_list = [OUNoise(output_dim) for i in range(num_of_task)]
## Defining variables
print('Defining Variables')
input_placeholders_actor = [v.input for v in s_actor_list]
context_placeholders_actor = [v.context for v in s_actor_list]
input_placeholders_critic = [v.input for v in s_critic_list]
# context_placeholders_critic = [v.context for v  in s_critic_list]
q_list = [v.output for v in s_critic_list]
action_list = [tf.clip_by_value(v.output, action_low_bound, action_high_bound) for v in s_actor_list]

actor_optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE_ACTOR_KB, name = 'adam_actor')
critic_optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE_CRITIC_KB, name = 'adam_critic')

KB_actor_var_list = L_actor.variable_list()
KB_actor_target_var_list = L_actor.target_variable_list()
KB_actor_var_list_per_latent = [ [v for v in KB_actor_var_list if '_latent%i'%(i) in v.name] for i in range(NUM_OF_LATENT)]
KB_critic_var_list = L_critic.variable_list()
KB_critic_var_list_per_latent = [ [v for v in KB_critic_var_list if '_latent%i'%(i) in v.name] for i in range(NUM_OF_LATENT)]
KB_critic_target_var_list = L_critic.target_variable_list()

s_actor_var_list_per_task = [pp.variable_list() for pp in s_actor_list]
s_actor_target_var_list_per_task = [pp.target_variable_list() for pp in s_actor_list]
s_critic_var_list_per_task = [pp.variable_list() for pp in s_critic_list]
s_critic_target_var_list_per_task = [pp.target_variable_list() for pp in s_critic_list]

## Calculating the loss and gradients
print('Calculating the loss and gradients')
training_q_list = [tf.placeholder(tf.float32, [None, 1], name = 'training_q_task%i'%(i)) for i in range(num_of_task)]
mse_loss_list = [tf.losses.mean_squared_error(tq, q) for tq, q in zip(training_q_list, q_list)]
l2_loss_list = [ALPHA_L2 * tf.add_n([tf.nn.l2_loss(v) for v in v_list]) for v_list in KB_critic_var_list_per_latent]
l1_loss_list = [ALPHA_L1 * tf.add_n([tf.reduce_sum(tf.abs(v)) for v in v_list]) for v_list in s_critic_var_list_per_task]

total_loss = tf.reduce_sum(mse_loss_list) + tf.reduce_sum(l2_loss_list) + tf.reduce_sum(l1_loss_list)
KB_update_loss = tf.reduce_sum(mse_loss_list) + tf.reduce_sum(l2_loss_list)
s_update_loss_list = [mse + l1loss for mse, l1loss in zip(mse_loss_list, l1_loss_list)]

KB_critic_gradient = critic_optimizer.compute_gradients( KB_update_loss, var_list = KB_critic_var_list)
s_critic_gradient_list = [critic_optimizer.compute_gradients( tmp_loss, var_list = tmp_var ) for tmp_loss, tmp_var in zip(s_update_loss_list, s_critic_var_list_per_task)]

inter_grads = [tf.gradients(q, a)[0] for q, a in zip(q_list, action_input)]
KB_actor_gradient = [actor_optimizer.compute_gradients(a, var_list = KB_actor_var_list, grad_loss = -g) for a, g in zip(action_list, inter_grads)]
KB_actor_gradient = [( tf.add_n([KB_actor_gradient[j][i][0] for j in range(len(KB_actor_gradient))]),KB_actor_gradient[0][i][1]) for i in range(len(KB_actor_gradient[0]))]
s_actor_gradient_list = [actor_optimizer.compute_gradients(a, var_list = s_var_list, grad_loss = -g) for a,s_var_list,g in zip(action_list, s_actor_var_list_per_task,inter_grads)]
# Applying gradients and updating network
print('Applying gradients and updating network')
KB_critic_train_op = critic_optimizer.apply_gradients( KB_critic_gradient )
KB_critic_update_op = [target_param.assign( tf.multiply(target_param, 1.-TAU_CRITIC) + tf.multiply(network_param, TAU_CRITIC)) \
	for target_param, network_param in zip(KB_critic_target_var_list, KB_critic_var_list)]

s_critic_train_op_list = [critic_optimizer.apply_gradients( s_critic_graident ) for s_critic_graident in s_critic_gradient_list]
s_critic_update_op_list = [ [target_param.assign( tf.multiply(target_param, 1.-TAU_CRITIC) + tf.multiply(network_param, TAU_CRITIC)) \
	for target_param, network_param in zip(temp_var, temp_target_var)] for temp_var, temp_target_var in zip(s_critic_var_list_per_task, s_critic_target_var_list_per_task) ]

KB_actor_train_op = actor_optimizer.apply_gradients( KB_actor_gradient )
KB_actor_update_op = [target_param.assign( tf.multiply(target_param, 1.- TAU_ACTOR) + tf.multiply(network_param, TAU_ACTOR)) \
	for target_param, network_param in zip(KB_actor_target_var_list, KB_actor_var_list)]
s_actor_train_op_list = [actor_optimizer.apply_gradients( s_actor_gradient ) for s_actor_gradient in s_actor_gradient_list]
s_actor_update_op_list = [ [target_param.assign( tf.multiply(target_param, 1.-TAU_ACTOR) + tf.multiply(network_param, TAU_ACTOR))\
	for target_param, network_param in zip(temp_var, temp_target_var)] for temp_var, temp_target_var in zip(s_actor_var_list_per_task, s_actor_target_var_list_per_task) ]


task_listing = zip(env_list, context_list, s_actor_list, s_critic_list, rb_list, on_list, range(num_of_task))
# task_listing = zip(*task_listing)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

## reward_list/done_list: num_Iter x num_tasks x time_step
## s_list: num_Iter x num_tasks x time_step x s_size
reward_list = []
done_list = []
s_actor_eval_list = []
s_critic_eval_list = []
for iter_count in range(MAX_ITER/MAX_TIME):
	reward_list_tmp = []
	done_list_tmp = []
	s_actor_list_tmp = []
	s_critic_list_tmp = []

	action_tmp_list = []
	q_loss_tmp_list = []
	mse_loss_tmp_list=[]

	for env, context, actor, critic, rb, noise, i in task_listing:

		state = env.reset()
		noise.reset()
		reward_list_ttmp = []
		done_list_ttmp = []
		s_actor_list_ttmp = []
		s_critic_list_ttmp = []
		s_update_count = 0

		

		for _ in range(MAX_TIME):
			action = sess.run(actor.output ,feed_dict = {actor.input: state[np.newaxis,:], actor.context: context[np.newaxis,:]} )
			action += noise.noise()
			action = np.clip(action, action_low_bound, action_high_bound)
			next_state, reward, done, _ = env.step(action)
			rb.add_sample(state, action[0], reward, next_state, context)
			# print(action[0])
			reward_list_ttmp.append(reward)
			done_list_ttmp.append(done)
			s_actor_list_ttmp.append(sess.run(s_actor_var_list_per_task[i]))
			s_critic_list_ttmp.append(sess.run(s_critic_var_list_per_task[i]))
			if rb.count < INITIALIZE_REPLAY_BUFFER:
				continue

			mini_batch = rb.rand_sample(batch_size = BATCH_SIZE)
			s_batch, a_batch, r_batch, s2_batch, c_batch = mini_batch
			a2_batch = sess.run(actor.target_output, feed_dict = {actor.target_input: s2_batch, actor.target_context: c_batch})
			tmp_q_batch = sess.run(critic.target_output, feed_dict = {critic.target_input: s2_batch, context_input[i]: c_batch, action_input[i]: a2_batch })
			training_q_batch = r_batch + GAMMA * tmp_q_batch

			sess.run(s_actor_train_op_list[i], feed_dict = {actor.input: s_batch, actor.context: c_batch, critic.input: s_batch, context_input[i]: c_batch, action_input[i]: a_batch})
			sess.run(s_critic_train_op_list[i], feed_dict = {critic.input: s_batch, context_input[i]: c_batch, action_input[i]: a_batch, training_q_list[i]: training_q_batch})
			sess.run([s_critic_update_op_list[i],s_actor_update_op_list[i]])
			s_update_count+=1
			# print('s update_count%i'%(s_update_count))

		s_actor_list_tmp.append(s_actor_list_ttmp)
		s_critic_list_tmp.append(s_critic_list_ttmp)
		reward_list_tmp.append(reward_list_ttmp)
		done_list_tmp.append(done_list_ttmp)

	s_actor_eval_list.append(s_actor_list_tmp)
	s_critic_eval_list.append(s_critic_list_tmp)
	reward_list.append(reward_list_tmp)
	done_list.append(done_list_tmp)

	if rb_list[0].count <= INITIALIZE_REPLAY_BUFFER:
		continue
	KB_update_count = 0
	for _ in range(MAX_TIME):
		mini_batches = [rb.rand_sample(batch_size = BATCH_SIZE) for rb in rb_list]
		feeding_dict = {}
		[feeding_dict.update({actor.target_input: mini_batch[3], actor.target_context:mini_batch[4]}) for actor, mini_batch in zip(s_actor_list, mini_batches)]
		# feeding_dict = {s_actor_list[0].target_input: mini_batches[0][3], s_actor_list[0].target_context: mini_batches[0][4]}
		a2_batches = sess.run([actor.target_output for actor in s_actor_list], feed_dict = feeding_dict)

		tmp_q_batches = [sess.run(critic.target_output, feed_dict = {critic.target_input: mini_batch[3], context_input[i]: mini_batch[4], action_input[i]: a2_batch}) \
			 for critic, mini_batch, a2_batch, i in zip(s_critic_list, mini_batches, a2_batches, range(num_of_task))]
		training_q_batches = [mini_batch[2] + GAMMA * tmp_q_batch for mini_batch, tmp_q_batch in zip(mini_batches, tmp_q_batches)]
		
		feeding_dict = {}
		[feeding_dict.update({critic.input: mini_batch[0], context_input[i]: mini_batch[4], action_input[i]: mini_batch[1], training_q_list[i]: training_q_batch}) \
			 for critic, mini_batch, training_q_batch, i in zip(s_critic_list, mini_batches, training_q_batches, range(num_of_task))]
		_, q_loss_tmp,mse_loss_tmp = sess.run([KB_critic_train_op, total_loss, tf.reduce_sum(mse_loss_list)], feed_dict = feeding_dict)

		feeding_dict = {}
		[feeding_dict.update({actor.input: mini_batch[0], actor.context: mini_batch[4], critic.input:mini_batch[0], context_input[i]: mini_batch[4], action_input[i]: mini_batch[1]}) \
			 for actor, critic, mini_batch, i in zip(s_actor_list, s_critic_list, mini_batches, range(num_of_task))]
		sess.run(KB_actor_train_op, feed_dict = feeding_dict)
		sess.run([KB_critic_update_op, KB_actor_update_op])
		KB_update_count += 1
		q_loss_tmp_list.append(q_loss_tmp)
		mse_loss_tmp_list.append(mse_loss_tmp)

	print('iter_count: %i, avg_reward: %3.2f, avg_loss:%3.2f, avg_mse_loss:%3.2f'%(iter_count, 1.*np.mean(reward_list_tmp), 1.*np.mean(q_loss_tmp_list), 1.*np.mean(mse_loss_tmp_list)))

	
		# print('KB update count%i'%(KB_update_count))
	# [feeding_dict.update({critic.target_input: mini_batch[3]})]
	# a2_batches = sess.run([])
	# for i in range(num_of_task):


	# feed_dict = {s_critic_list[i].for i in range(num_of_task)}
	# for 



			# img = env.render(mode = 'rgb_array')

		# env.render(close = True)

# pdb.set_trace()
filename = '/disk/scratch/chenyang/Data/mtl_cartpole/exp_mtl'
my_shelf = shelve.open(filename,'n')
saving_var_list = ['reward_list', 'done_list', 's_actor_list', 's_critic_list']
for key in saving_var_list:
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()