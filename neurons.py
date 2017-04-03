import sys
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

	def __init__(self,name="McCulloch-Pitts",theta=1.0,inputs=None,weight=1.0,post_synaptic_neuron=None,pre_synaptic_neuron=None):
		self.type = name
		self.theta = theta
		if inputs is None: self.inputs = []
		self.weight = weight
		self.post_synapse = post_synaptic_neuron
		if pre_synaptic_neuron is None: self.pre_synapse = []

	def output(self):
		if sum(self.inputs) >= self.theta:
			return 1.0 * self.weight
		else:
			return 0.0

	def connectTo(self,post):
		if self.post_synapse is not None:
			print "replacing old postsynaptic neuron with", post
		self.post_synapse = post
		post.pre_synapse.append(self)
		

class Layer(object):
	"""
	"""
	def __init__(self,network,level,neurons=None):
		self.network=network
		self.level = level
		self.nNeurons = 0
		if neurons is None: self.neurons = []
		else: self.nNeurons = len(neurons)

	# def addInput(self,weight):
	# 	newInput = Input(weight)
	# 	self.neurons.append(newInput)
	# 	self.network.neurons.append(newInput)
	# 	self.network.nNeurons+=1
	# 	return newInput

	# def addNeuron(self):
	# 	newNeuron = Neuron()
	# 	self.neurons.append(newNeuron)
	# 	self.network.neurons.append(newNeuron)
	# 	self.network.nNeurons+=1
	# 	return newNeuron

	# def addIF(self):
	# 	newNeuron = IF()
	# 	self.neurons.append(newNeuron)
	# 	self.network.neurons.append(newNeuron)
	# 	self.network.nNeurons+=1
	# 	return newNeuron

	# def addLIF(self):
	# 	newNeuron = LIF()
	# 	self.neurons.append(newNeuron)
	# 	self.network.neurons.append(newNeuron)
	# 	self.network.nNeurons+=1
	# 	return newNeuron

	def addNeuron(self,neuron_type,args):
		if neuron_type.lower()=="input": 
			newNeuron = Input(**args) 
		elif neuron_type.lower()=="mcculloch-pitts": 
			newNeuron = Neuron(**args) 
		elif neuron_type.lower()=="if":
			newNeuron = IF(**args) 
		elif neuron_type.lower()=="lif":
			newNeuron = LIF(**args) 
		else: 
			print "Invalid argument:", neuron_type.lower() 
			return None
		self.neurons.append(newNeuron)
		self.network.neurons.append(newNeuron)
		self.network.nNeurons+=1
		return newNeuron

	def removeNeuron(self, neuron):
		if neuron.post_synapse is not None: neuron.post_synapse.pre_synapse.remove(neuron)
		if neuron.pre_synapse is not None:
			for i in neuron.pre_synapse:
				i.post_synapse=None
		self.neurons.remove(neuron)
		self.nNeurons -= 1


class Network(object):
	"""
	"""
	def __init__(self):
		self.layers = []
		self.nLayers = 0
		self.neurons = []
		self.nNeurons = 0
		self.output = None
		self.time = []

	def _initialize(self):
		"""Initializes integrating neuron membrane potential"""
		self.output=np.zeros(len(self.time))
		for i in self.neurons:
			if i.type=="IF" or i.type=="LIF":
				i.V_m=np.zeros(len(self.time))
				i.t_ref=0

	def spikePlot(self, title):
		pb.plot(self.time, self.output)
		pb.title(title)
		pb.ylabel('Membrane Potential (V)')
		pb.xlabel('Time (msec)')
		pb.ylim([0,3])
		pb.show()

	def reset(self):
		for j in self.neurons:
			if j.type in ["IF","LIF"]:
				j.V_m=[]

	def addLayer(self):
		newLayer = Layer(self,self.nLayers)
		self.layers.append(newLayer)
		self.nLayers+=1
		return newLayer

	def removeLayer(self,layer):
		for i in layer.neurons:
			layer.removeNeuron(i)
			self.nNeurons -= 1
			self.neurons.remove(i)
		l_index = self.layers.index(layer)
		for i in range(l_index,len(self.layers)):
			self.layers[i].level -= 1
		self.layers.remove(layer)
		self.nLayers -= 1

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
		self.time = np.arange(0,T+dt,dt)
		self._initialize()
		_printHeader()
		for i,t in enumerate(self.time):
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
		self.spikePlot("Network Output")


class Input(Neuron):
	"""
	"""
	def __init__(self,weight,name="input",inputs=None):
		Neuron.__init__(self,name=name,theta=0.0,inputs=inputs,weight=weight,post_synaptic_neuron=None,pre_synaptic_neuron=None)
		if inputs is None: self.inputs = []


class IF(Neuron):
	"""
	"""
	def __init__(self,name="IF",V_m=None,C=1.0,theta=1.0,V_d=1.0,inputs=None,weight=1.0):
		Neuron.__init__(self,name=name,theta=theta,inputs=inputs,weight=weight,post_synaptic_neuron=None,pre_synaptic_neuron=None)
		if inputs is None: self.inputs = []
		self.V_m=V_m
		self.C=C
		self.V_d=V_d
		self.time = []

	def _printState(self,i):
		print i, self.inputs, "\t",self.V_m[i], "\t", self.theta

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
			self.t_ref = t

	def spikePlot(self,title):
		pb.plot(self.time, self.V_m)
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
		self.time = np.arange(0,T+dt,dt)
		self.V_m=np.zeros(len(self.time))
		_printHeader()
		for i,t in enumerate(self.time):
			if t > self.t_ref:
				self._updateMembrane(i,t,dt)
			self._printState(i)
		self.spikePlot("LIF Output")
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
		IF.__init__(self,name=name,V_m=V_m,C=C,theta=theta,V_d=V_d,inputs=inputs)
		if inputs is None: self.inputs = []
		self.R=R
		self.tau=R*C
		self.tau_ref=tau_ref
		self.time = []

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

def _makeLayers(n, net):
	for i in range(n):
		net.addLayer()

def main():
	inputs = open("input.txt","r")
	adj = open("adjacency.txt","r")
	out = open("output.txt","w")
	lines = inputs.readlines()
	if "layers" not in lines[0]: print "bad input: number of layers not specified"
	for row,line in enumerate(lines):
		if 'layers' in line:
			net = Network()
			nLayers=int((line.split('='))[1])
			_makeLayers(nLayers,net)
		else:
			layer=net.layers[row-1]
			neurons=line.split(',')
			for neuron in neurons:
				params=neuron.strip().split(' ')
				f = layer.addNeuron
				neuron_type = (params[0].split('='))[1]
				kw={}
				for param in params[1:]:
					pair=param.split("=")
					kw[pair[0]]=float(pair[1])
				f(neuron_type,kw)
	lines = adj.readlines()
	for row,line in enumerate(lines):
		line = line.strip()
		for i,b in enumerate(line):
			if b=="1":
				net.neurons[row].connectTo(net.neurons[i])
	net.run(50,.125)


if __name__=="__main__":
	main()

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

# net = Network()
# l0 = net.addLayer()
# l1 = net.addLayer()
# i0 = l0.addInput(1)
# i1 = l0.addInput(1)
# n0 = l1.addLIF()
# n0.tau_ref=4.0
# n0.theta=1.0
# i0.connectTo(n0)
# i1.connectTo(n0)
# net.run(50,.125)

# lif = LIF(tau_ref=4.0,theta=1.0)
# lif.inputs=[1.0,1.0]
# lif.run(50,.5)