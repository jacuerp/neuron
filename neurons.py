import numpy as np
import pylab as pb

class Neuron(object):
	"""
	Aninstance of a basic McCulloch-Pitts neuron.
	
	INSTANCE VARIABLES:
		type [string]:
			Denotes the type of neuronal model 
		theta [float]:

		inputs [list<float>]:

		weight [float]:

		post_synapse [Neuron]:
			
	"""

	def __init__(self,name="McCulloch-Pitts",theta=1.0,inputs=None,weight=1.0,post_synaptic_neuron=None):
		self.type = name
		self.theta = theta
		if inputs is None: self.inputs = []
		self.weight = weight
		self.post_synapse = post_synaptic_neuron

	def output(self):
		if sum(self.inputs) >= self.theta:
			return 1.0 * self.weight
		else:
			return 0.0

	def connectTo(self,post):
		if self.post_synapse is not None:
			print "replacing old postsynaptic neuron with", post
		self.post_synapse = post
		

class Layer(object):
	"""
	"""
	def __init__(self,network,level,neurons=None):
		self.network=network
		self.level = level
		self.nNeurons = 0
		if neurons is None: self.neurons = []
		else: self.nNeurons = len(neurons)

	def addInput(self,weight):
		newInput = Input(weight)
		self.neurons.append(newInput)
		self.network.neurons.append(newInput)
		self.network.nNeurons+=1
		return newInput

	def addNeuron(self):
		newNeuron = Neuron()
		self.neurons.append(newNeuron)
		self.network.neurons.append(newNeuron)
		self.network.nNeurons+=1
		return newNeuron

	def addIF(self):
		newNeuron = IF()
		self.neurons.append(newNeuron)
		self.network.neurons.append(newNeuron)
		self.network.nNeurons+=1
		return newNeuron

	def addLIF(self):
		newNeuron = LIF()
		self.neurons.append(newNeuron)
		self.network.neurons.append(newNeuron)
		self.network.nNeurons+=1
		return newNeuron


class Network(object):
	"""
	"""
	def __init__(self):
		self.layers = []
		self.nLayers = 0
		self.neurons = []
		self.nNeurons = 0
		self.output = None

	def _initialize(self,steps):
		"""Initializes integrating neuron membrane potential"""
		self.output=np.zeros(len(steps))
		for i in self.neurons:
			if i.type=="IF" or i.type=="LIF":
				i.V_m=np.zeros(len(steps))
				i.t_ref=0

	def _spikePlot(self, title, time):
		pb.plot(time, self.output)
		pb.title(title)
		pb.ylabel('Membrane Potential (V)')
		pb.xlabel('Time (msec)')
		pb.ylim([0,3])
		pb.show()

	def reset(self):
		for j in self.neurons:
			if j.type=="IF":
				j.membrane_voltage=0

	def addLayer(self):
		newLayer = Layer(self,self.nLayers)
		self.layers.append(newLayer)
		self.nLayers+=1
		return newLayer

	def output(self):
		"""Still needs work. Should output an output value for a single step"""
		out = 0
		for i in xrange(self.nLayers):
			layer = self.layers[i]
			for j in xrange(len(layer.neurons)):
				pre = layer.neurons[j]
				output_pre = pre.output()
				if pre.post_synapse is None:
					out=output_pre
				else:
					post = pre.post_synapse
					post.inputs.append(output_pre)
				pre.inputs = []
		self.reset()
		return out

	def run(self,T,dt):
		"""
		Returns output array for this network over T with time step dt
		"""
		time = np.arange(0,T+dt,dt)
		self._initialize(time)
		_printHeader()
		for i,t in enumerate(time):
			for j in xrange(self.nLayers):
				layer = self.layers[j]
				for k in xrange(len(layer.neurons)):
					pre = layer.neurons[k]
					if _isIntegrated(pre):
						output_pre = pre.output(i,t,dt)
					else:
						output_pre = pre.output()
					if pre.post_synapse is None:
						self.output[i]=output_pre
					else:
						post = pre.post_synapse
						post.inputs.append(output_pre)
					pre.inputs = []
		self._spikePlot("Network Output",time)


class Input(Neuron):
	"""
	"""
	def __init__(self,w,name="input",inputs=None):
		Neuron.__init__(self,name=name,theta=0.0,inputs=inputs,weight=w,post_synaptic_neuron=None)
		if inputs is None: self.inputs = []


class IF(Neuron):
	"""
	"""
	def __init__(self,name="IF",V_m=None,C=1.0,tau_ref=0.0,theta=1.0,V_d=1.0,inputs=None,weight=1.0):
		Neuron.__init__(self,name=name,theta=theta,inputs=inputs,weight=weight,post_synaptic_neuron=None)
		if inputs is None: self.inputs = []
		self.V_m=V_m
		self.C=C
		self.tau_ref=tau_ref
		self.V_d=V_d

	def _printState(self,i):
		print i, "|", self.inputs, "\t",self.V_m[i], "\t", self.theta

	# def output(self):
	# 	self.membrane_voltage += sum(self.inputs)
	# 	self.printState()
	# 	if self.membrane_voltage >= self.theta:
	# 		self.membrane_voltage = 0.0
	# 		return 1.0 * self.weight
	# 	return 0.0

	def _updateMembrane(self,s,t,dt):
		self.V_m[s] = self.V_m[s-1] + self.I / self.C * dt
		if self.V_m[s] >= self.theta:
			self.V_m[s] += self.V_d
			self.t_ref = t + self.tau_ref

	def _spikePlot(self,title,time):
		pb.plot(time, self.V_m)
		pb.title(title)
		pb.ylabel('Membrane Potential (V)')
		pb.xlabel('Time (msec)')
		pb.ylim([0,3])
		pb.show()

	def output(self,s,t,dt):
		if self.inputs is None: return "No input to neuron"
		self.I=sum(self.inputs)
		if t > self.t_ref:
			self._updateMembrane(s,t,dt)
		self._printState(s)
		return self.V_m[s]


	def run(self,T,dt):
		"""
		Runs IF model
		returns array V(t) given constant input
		"""
		if self.inputs is None: return "No input to neuron"
		self.I=sum(self.inputs)
		self.t_ref=0
		time = np.arange(0,T+dt,dt)
		self.V_m=np.zeros(len(time))
		_printHeader()
		for i,t in enumerate(time):
			if t > self.t_ref:
				self._updateMembrane(i,t,dt)
			# self._printState(i)
		print "before"
		self._spikePlot("LIF Output",time)
		print "after"
		return self.V_m

class LIF(IF):
	"""
	Constants: 
	T: duration in msec
	dt: time step
	time = np.arange(0,T+dt,dt)
	t_ref: refractory time

	LIF internals:
	V_m: membrane potential V
	R: resistance kOhm
	C: capacitance mF
	tau: time constant R*C
	tau_ref: refractory period
	theta: spike threshold
	V_d: spike delta
	I: input A
	"""
	def __init__(self,name="LIF",V_m=None,R=1.0,C=10.0,tau_ref=0.0,theta=1.0,V_d=1.0,inputs=None):
		IF.__init__(self,name=name,V_m=V_m,C=C,tau_ref=tau_ref,theta=theta,V_d=V_d,inputs=inputs)
		if inputs is None: self.inputs = []
		self.R=R
		self.tau=R*C

	def _updateMembrane(self,s,t,dt):
		self.V_m[s] = self.V_m[s-1] + (-self.V_m[s-1] + self.I*self.R) / self.tau * dt
		if self.V_m[s] >= self.theta:
			self.V_m[s] += self.V_d
			self.t_ref = t + self.tau_ref


def _isIntegrated(neuron):
	return neuron.type=="IF" or neuron.type=="LIF"

def _printHeader():
	print "_____________________________________________"
	print "|step|input      |membrane     |threshold    |"
	print "|____|___________|_____________|_____________|"



# AND = Network()
# l0 = AND.addLayer()
# l1 = AND.addLayer()
# i0 = l0.addInput(1)
# i1 = l0.addInput(1)
# n0 = l1.addNeuron()
# n0.weight = 1.0
# n0.theta = 2.0
# i0.connectTo(n0)
# i1.connectTo(n0)

net = Network()
l0 = net.addLayer()
l1 = net.addLayer()
i0 = l0.addInput(1)
i1 = l0.addInput(1)
n0 = l1.addLIF()
n0.tau_ref=4.0
n0.theta=1.0
i0.connectTo(n0)
i1.connectTo(n0)
net.run(50,.125)

lif = LIF(tau_ref=4.0,theta=1.0)
lif.inputs=[1.0,1.0]
lif.run(50,.5)

# fig = Figure()
# canvas = FigureCanvas(fig)
# ax1 = fig.add_subplot()
# ax2 = fig.add_subplot()
# ax1.plot()
# ax2.plot()
# ax1.set_title('Network Output')
# ax1.grid(True)
# ax1.set_xlabel('time')
# ax1.set_ylabel('volts')
# canvas.print_figure('test')