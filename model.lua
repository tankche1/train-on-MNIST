-- @Author: shuqin
-- @Date:   2017-04-26 18:41:29
-- @Last Modified by:   shuqin
-- @Last Modified time: 2017-04-28 14:49:25

paths.dofile('ReccurentPreProcessor.lua')

------------------------------------------------------------
-- convolutional network 
------------------------------------------------------------
function createConvnet(classes)
	local net = nn.Sequential()
	-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
	net:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
	net:add(nn.Tanh())
	net:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
	-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
	net:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
	net:add(nn.Tanh())
	net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- stage 3 : standard 2-layer MLP:
	net:add(nn.Reshape(64*3*3))
	net:add(nn.Linear(64*3*3, 200))
	net:add(nn.Tanh())
	net:add(nn.Linear(200, #classes))
	net:add(nn.LogSoftMax())

    return net
end

------------------------------------------------------------
-- regular 2-layer MLP
------------------------------------------------------------
function createMLP(classes)
	local net = nn.Sequential()
	net:add(nn.Reshape(1024))
	net:add(nn.Linear(1024, 2048))
	net:add(nn.Tanh())
	net:add(nn.Linear(2048,#classes))
	net:add(nn.LogSoftMax())

	return net
end

------------------------------------------------------------
-- simple linear model: logistic regression
------------------------------------------------------------

function createLinear(classes)
	local net = nn.Sequential()
	net:add(nn.Reshape(1024))
	net:add(nn.Linear(1024,#classes))
	net:add(nn.LogSoftMax())
	
	return net
end

------------------------------------------------------------
-- reinforcement learning agent
------------------------------------------------------------

function createAgent(classes, hiddenSize, actions, nStep, execute_fn)
	--print(hiddenSize)
	--print(#actions)

	local rnn_input = nn.Sequential()
					:add(nn.View(-1,1*32*32))
					:add(nn.Linear(1*32*32, hiddenSize))
					:add(nn.Tanh())
	local rnn_feedback = nn.Linear(hiddenSize, hiddenSize)
	--local rnn_feedback = nn.Sequential()
	--						:add(nn.Linear(hiddenSize, hiddenSize))
	--						:add(nn.Tanh())
	local rnn = nn.Recurrent(hiddenSize, rnn_input, rnn_feedback,nn.Tanh(), 99999)

	local actor = nn.Sequential()
					:add(nn.Linear(hiddenSize, #actions))
					:add(nn.SoftMax())
					:add(nn.ReinforceCategorical())--A Reinforce subclass that implements the REINFORCE algorithm (ref. A) for a Categorical (i.e. Multinomial with one sample) probability distribution.  output is one hot. same shape as input
					:add(nn.ArgMax(2))

	local preprocessor = nn.RecurrentPreProcessor(rnn, actor, nStep, {hiddenSize}, execute_fn)

	-- classifier
	local classifier = nn.Sequential()
					:add(nn.Linear(hiddenSize, #classes))
					:add(nn.LogSoftMax())

	-- Stack them into a unify model
	local agent = nn.Sequential()
				:add(preprocessor)
				:add(nn.SelectTable(-1))-- change to Table
				:add(classifier)

	-- baseline, which approximate the expected reward
	local baseline = nn.Sequential()
				:add(nn.Constant(1,1))
				:add(nn.Add(1))

	local concat = nn.ConcatTable():add(nn.Identity()):add(baseline)
	local concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

	agent:add(concat2)

	return agent

end

