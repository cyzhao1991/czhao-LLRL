import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import ode, math
from os import path

class PendulumEnv(gym.Env):
	metadata = {
		'render.modes':['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}


	def create_basebox(self, body, pos):
		body.setPosition(pos)
		M = ode.Mass()
		M.setSphereTotal(1., .2) #setBox(mass, lx, ly, lz)
		# M.translate((size[0]/2, size[1]/2 , 0.))
		# M.setParameters(M.mass, M.c[0], M.c[1], M.c[2], M.I[0][0], M.I[1][1],M.I[2][2], M.I[0][1], M.I[0][2], M.I[1][2]) #setParameters(mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23) 
		M.setParameters(M.mass, M.c[0] ,M.c[1] , M.c[2], M.I[0][0], M.I[1][1],M.I[2][2], M.I[0][1], M.I[0][2], M.I[1][2]) #setParameters(mass, cgx, cgy, cgz, I11, I22, I33, I12, I13, I23) 
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

	def __init__(self, gravity = 9.8, mass = 1.0, tau = 0.02, size_pole = (1.0, .1)):
		self.gravity = gravity
		self.mass = mass
		self.dt = tau
		self.viewer = None
		self.viewerSize = 500
		self.spaceSize = 7.
		self.size_pole = size_pole

		self.max_force = 3.
		self.theta_threshold = 3.2
		self.action_space = spaces.Box(low = -self.max_force, high = self.max_force, shape = (1,))
		high = np.array([self.theta_threshold *2, np.finfo(np.float32).max])
		self.observation_space = spaces.Box(-high, high)

		self.world = ode.World()
		self.world.setGravity((0, -9.81, 0))
		self.body1 = ode.Body(self.world)
		self.body2 = ode.Body(self.world)
		self.create_basebox(self.body1, (0.,0.,0.))
		self.create_link(self.body2, (0.,0.5,0.), self.mass, size_pole)

		self.space = ode.Space()

		self.j1 = ode.FixedJoint(self.world)
		self.j1.attach(self.body1, ode.environment)
		self.j1.setFixed()

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
			self.world.setGravity((0,-9.81,0))
		else:
			self.world.setGravity((0,0,0))

	def _step(self, action):
		# assert self.action_space.contains(action), "action %r (%s) invalid"%(action, type(action))
		self.j2.addTorque(action)
		self.world.step(self.dt)
		state = self._get_obs()
		costs = self.cost_function(state, action)
		done = self.if_done(state)
		return state, -costs, done, {}

	def _reset(self):
		ode.CloseODE()
		self.__init__()
		return _get_obs()

	def _get_obs(self):
		theta = self.j2.getAngle()
		thetadot = self.j2.getAngleRate()
		return np.array([theta, thetadot])


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

			base = rendering.make_circle(.2)
			base.set_color(0,0,0)
			self.viewer.add_geom(base)

			pole = rendering.make_capsule(self.size_pole[0], self.size_pole[1])
			pole.set_color(.8, .6, .4)
			self.pole_trans = rendering.Transform()
			pole.add_attr(self.pole_trans)
			self.viewer.add_geom(pole)

			axle = rendering.make_circle(self.size_pole[1]/2.)
			axle.set_color(.5,.5,.8)
			self.viewer.add_geom(axle)

		self.pole_trans.set_rotation(state[0] + math.pi/2)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def cost_function(self, state, action):
		th, thdot = state
		u = action
		costs = angle_normalize(th) ** 2 + .1*thdot**2 + .001 *u**2
		return costs

	def if_done(self, state):
		return False

def angle_normalize(x):
	return (((x+np.pi)%(2*np.pi)) - np.pi)