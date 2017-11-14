import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import ode, math
from os import path

class CartPoleEnv(gym.Env):
	metadata = {
		'render.modes':['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}


	def create_basebox(self, body, pos, mass, size):
		body.setPosition(pos)
		M = ode.Mass()
		M.setBoxTotal(mass, size[0], size[1], 1.) #setBox(mass, lx, ly, lz)
		M.translate((size[0]/2, size[1]/2 , 0.))
		# M.setParameters(M.mass, M.c[0], M.c[1], M.c[2], M.I[0][0], M.I[1][1],M.I[2][2], M.I[0][1], M.I[0][2], M.I[1][2]) #setParameters(mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23) 
		M.setParameters(M.mass, 0 ,0 , M.c[2], M.I[0][0], M.I[1][1],M.I[2][2], M.I[0][1], M.I[0][2], M.I[1][2]) #setParameters(mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23) 

		body.setMass(M)

	def create_link(self, body, pos, mass, size):
		body.setPosition(pos)
		M = ode.Mass()
		M.setCylinderTotal(mass, 2, size[1], size[0])
		M.translate( (0., size[0]/2, 0.) )
		# M.setParameters(M.mass, M.c[0], M.c[1], M.c[2], M.I[0][0], M.I[1][1],M.I[2][2], M.I[0][1], M.I[0][2], M.I[1][2]) #setParameters(mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23)
		M.setParameters(M.mass, M.c[0], 0, M.c[2], M.I[0][0], M.I[1][1],M.I[2][2], M.I[0][1], M.I[0][2], M.I[1][2]) #setParameters(mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23)
		body.setMass(M)

	# def create_ee(self, body, pos, mass, radius):
	# 	body.setPosition(pos)
	# 	M = ode.Mass()
	# 	M.setCyliderTotal(mass, 1., size[0], size[1])
	# 	M.translate((.5, 0., 0.))
	# 	M.setParameters((M.mass, 0, M.c[1], M.c[2], M.I[0][0], M.I[1][1],M.I[2][2], M.I[0][1], M.I[0][2], M.I[1][2])) #setParameters(mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23) 
	# 	body.setMass(M)

	def __init__(self, gravity = 9.8, mass_cart = 1.0, mass_pole = 1.0, tau = 0.02, size_box = (0.5, 0.3), size_pole = (1.0, .1)):
		self.gravity = gravity
		self.mass_pole = mass_pole
		self.mass_cart = mass_cart
		self.dt = tau
		self.viewer = None
		self.viewerSize = 500
		self.spaceSize = 7.
		self.size_box = size_box
		self.size_pole = size_pole

		self.max_force = 10.
		self.theta_threshold = 3.2
		self.x_threshold = 3.0
		self.action_space = spaces.Box(low = -self.max_force, high = self.max_force, shape=(1,))
		high = np.array([self.x_threshold * 2, np.finfo(np.float32).max, self.theta_threshold *2, np.finfo(np.float32).max])
		self.observation_space = spaces.Box(-high, high)

		self.world = ode.World()
		self.world.setGravity((0, -gravity, 0))
		self.body1 = ode.Body(self.world)
		self.body2 = ode.Body(self.world)
		self.create_basebox(self.body1, (0.,0.,0.), self.mass_cart, size_box)
		self.create_link(self.body2, (0.,size_pole[0] /2 ,0.), self.mass_pole, size_pole)

		self.space = ode.Space()

		self.j1 = ode.SliderJoint(self.world)
		self.j1.attach(self.body1, ode.environment)
		self.j1.setAxis( (1, 0, 0) )
		self.j1.setFeedback(1)

		self.j2 = ode.HingeJoint(self.world)
		self.j2.attach(self.body1, self.body2)
		self.j2.setAnchor( (0., 0. ,0.) )
		self.j2.setAxis ( (0, 0, -1) )
		self.j2.setFeedback(1)

	def _seed(self, seed = None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def enable_gravity(self, on = True):
		if on:
			self.world.setGravity((0,-gravity,0))
		else:
			self.world.setGravity((0,0,0))

	def _step(self, action):
		# assert self.action_space.contains(action), "action %r (%s) invalid"%(action, type(action))
		self.j1.addForce(action)
		self.j2.addTorque(0.)
		self.world.step(self.dt)
		state = self._get_obs()
		costs = self.cost_function(state, action)
		done = self.if_done(state)
		return state, -costs, done, {}

	def _reset(self):
		# ode.CloseODE()
		del self.world
		self.__init__(gravity = self.gravity, mass_cart = self.mass_cart, mass_pole = self.mass_pole, tau = self.dt, size_box = self.size_box, size_pole = self.size_pole )
		# self.body1.setPosition((0.,0.,0.))
		# self.body2.setPosition(())
		return self._get_obs()

	def _get_obs(self):
		x = self.j1.getPosition()
		xdot = self.j1.getPositionRate()
		theta = self.j2.getAngle()
		thetadot = self.j2.getAngleRate()
		return np.array([x, xdot, theta, thetadot])


	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		x1,y1,z1 = self.body1.getPosition()
		x2,y2,z2 = self.body2.getPosition()
		state = self._get_obs()

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(self.viewerSize, self.viewerSize)
			self.viewer.set_bounds(-self.spaceSize/2.0,self.spaceSize/2.0,-self.spaceSize/2.0,self.spaceSize/2.0)

			cw, ch = self.size_box
			l, r, t, b = -cw/2, cw/2, ch/2, -ch/2
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.cart_trans = rendering.Transform()
			cart.add_attr(self.cart_trans)
			self.viewer.add_geom(cart)

			pole = rendering.make_capsule(self.size_pole[0], self.size_pole[1])
			pole.set_color(.8, .6, .4)
			self.pole_trans = rendering.Transform()
			pole.add_attr(self.pole_trans)
			pole.add_attr(self.cart_trans)
			self.viewer.add_geom(pole)

			axle = rendering.make_circle(self.size_pole[1]/2.)
			axle.set_color(.5,.5,.8)
			# self.axle_trans = rendering.Transform()
			axle.add_attr(self.cart_trans)
			self.viewer.add_geom(axle)

			self.track = rendering.Line((-self.spaceSize/2.0, 0), (self.spaceSize/2.0, 0))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

		self.cart_trans.set_translation(x1,y1)
		# self.pole_trans.set_translation(x1,y1)
		self.pole_trans.set_rotation(state[2] + math.pi/2)
		# self.axle_trans2.set_translation(x2,y2)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

			# base = rendering.make_capsule(self.size_box[0], self.size_box[1])
			# base.set_color(.8,.3,.3)

	def cost_function(self, state, action):
		x, xdot, th, thdot = state
		u = action
		costs = x **2 + .2*xdot **2 + angle_normalize(th) ** 2 + .1*thdot**2 + .001 *u**2
		return costs

	def if_done(self, state):
		return False

def angle_normalize(x):
	return (((x+np.pi)%(2*np.pi)) - np.pi)