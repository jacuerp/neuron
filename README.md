# neuron
A Python library for building simple neuron models.

## Overview
There are 4 main classes which users can manipulate:
#### 1. Inputs
A special type of neuron which output a predetermined constant value along an axon. They cannot recieve input values and do not compute any outputs.
#### 2. Neurons
Neurons recieve input values and compute an output based on the specifics of the chosen neuron model.
#### 3. Layers
![alt text](img/layer_example.png "Example of a network model organized into layers")
Layers are a tool for graphically organizing the layout of Neurons. When running a model, the neurons in Layer 1 read their input, compute an output, and use it to populate the inputs of connected neurons in the next layer, and so on. 
#### 4. Networks
A network is a canvas you can use to add Layers, Neurons, and Inputs to create more complex networks of neurons. 
## Types of Neurons
#### 1. Input
**parameters**
- type[string]  
   Neuron model
- weight[float]  
   The weight of the outgoing axon
- post_synapse[Neuron]  
   Post-synaptic neuron
   
**functions**
- output()  
   Sums inputs and returns 1.0 multiplied by the weight of this neuron
- connectTo(Neuron)  
   Connects the output of this neuron to the input of the specified neuron
#### 2. McCulloch-Pitts
**parameters**
- type[string]  
   Neuron model
- weight[float]  
   The weight of the outgoing axon
- theta[float]  
   Threshold
- inputs[float array]  
   Array of input values at this time step
- pre_synapse[Neuron array]  
   Array of pre-synaptic neurons
- post_synapse[Neuron]  
   Post-synaptic neuron
   
**functions**
- output()  
   Sums inputs and returns 1.0 multiplied by the weight of this neuron
- connectTo(post_synapse[Neuron])  
   Connects the output of this neuron to the input of the specified neuron
- spikePlot(title[string])  
   Creates a spikeplot using the membrane voltage from the most recent simulation 
- run(T,dt)  
   Runs a simulation over the specified amount of time T, and time delta dt
#### 3. Integrate and Fire
**parameters**
- type[string]  
   Neuron model
- weight[float]  
   The weight of the outgoing axon
- theta[float]  
   Threshold
- V_m[float array]  
   Voltage membrane at each timestep in a given simulation
- C[float]  
   Capacitance
- V_d[float]  
   Spike delta added to the voltage membrane at threshold
- inputs[float array]  
   Array of input values at this timestep
- pre_synapse[Neuron array]  
   Array of pre-synaptic neurons
- post_synapse[Neuron]  
   Post-synaptic neuron
   
**functions**
- output()  
   Sums inputs and returns 1.0 multiplied by the weight of this neuron
- connectTo(post_synapse[Neuron])  
   Connects the output of this neuron to the input of the specified neuron
- spikePlot(title[string])  
   Creates a spikeplot using the membrane voltage from the most recent simulation 
- run(T,dt)  
   Runs a simulation over the specified amount of time T, and time delta dt
#### 4. Leaky Integrate and Fire
**parameters**
- type[string]  
   Neuron model
- weight[float]  
   The weight of the outgoing axon
- theta[float]  
   Threshold
- V_m[float array]  
   Voltage membrane at each timestep in a given simulation
- R[float]  
   Resistance
- tau[float]  
   Time constant R*C
- tau_ref[float]  
   refractory period
- C[float]  
   Capacitance
- V_d[float]  
   Spike delta added to the voltage membrane at threshold
- inputs[float array]  
   Array of input values at this timestep
- pre_synapse[Neuron array]  
   Array of pre-synaptic neurons
- post_synapse[Neuron]  
   Post-synaptic neuron
   
**functions**
- output()  
   Sums inputs and returns 1.0 multiplied by the weight of this neuron
- connectTo(post_synapse[Neuron])  
   Connects the output of this neuron to the input of the specified neuron
- spikePlot(title[string])  
   Creates a spikeplot using the membrane voltage from the most recent simulation 
- run(T,dt)  
   Runs a simulation over the specified amount of time T, and time delta dt
## Reference
