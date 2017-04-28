----------------------------------------------------------------------
-- TODO
-- complete the execute_fn and actions in line 127-135
----------------------------------------------------------------------


----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, LBFGS)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'dp'
require 'dpnn'
require 'rnn'
require 'io'
require 'image'

----------------------------------------------------------------------
-- parse command-line options
--

local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear | agent
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   --blur           (default false)           use blur or not 
   --noise            (default fasle)           use noise or not        
   --nStep            (default 6)           maximum step the agent can perform   
   --hiddenSize       (default 128)         rnn hiddenSize   
]]

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

----------------------------------------------------------------------
-- define actions and execute_fn 
-- the execute_fn should works like this
-- input_processed = execute_fn(input, action) 
-- here action is a tensor indicating the number of action
-- input is a minibatch input
-- input_processed should have the same size as input
-- I change function train a little bit to initialize denoise perameters


if opt.blur=='1' then
   opt.actions = ...
   execute_fn = function () end 
-- tolerance=0.2, tau=0.125, tv_weight=100 +tolerance=0.03 +tau=0.01 +tv_weight=7
elseif opt.noise=='1' then
   opt.actions = {'+tolerance','-tolerance','+tau','-tau','+tv_weight','-tv_weight'}
   execute_fn = 
   	function (input,action) 
		input_processed= input:clone()
		for i=1,opt.batchSize do
			image.save('img.jpg', input[i])

         execute_action = opt.actions[action[i]]
		
			--execute action
			if execute_action=='+tolerance' then
				denoise_paremeters[i].tolerance=denoise_paremeters[i].tolerance+0.03 
			elseif execute_action=='-tolerance' then
				denoise_paremeters[i].tolerance=denoise_paremeters[i].tolerance-0.03
			elseif execute_action=='+tau' then
				denoise_paremeters[i].tau=denoise_paremeters[i].tau+0.01
			elseif execute_action=='-tau' then
				denoise_paremeters[i].tau=denoise_paremeters[i].tau-0.01
			elseif execute_action=='+tv_weight' then
				denoise_paremeters[i].tv_weight=denoise_paremeters[i].tv_weight+7
			elseif execute_action=='+tv_weight' then
				denoise_paremeters[i].tv_weight=denoise_paremeters[i].tv_weight-7
			end
			--print(denoise_paremeters[i].tolerance..","..denoise_paremeters[i].tau..","..denoise_paremeters[i].tv_weight..",")
			
			-- deliver paremeters to python
			file = io.open("denoise_paremeters.txt","w")
			file:write(denoise_paremeters[i].tolerance..","..denoise_paremeters[i].tau..","..denoise_paremeters[i].tv_weight..",")
			file:close()

			os.execute('python denoise.py')
			input_processed[i]=image.load('img_denoise.jpg')
		end
		return input_processed
	end

end

paths.dofile('model.lua')

if opt.network == '' then
   -- define model to train
   if opt.model == 'convnet' then
      model = createConvnet(classes)

   elseif opt.model == 'mlp' then
      model = createMLP(classes)

   elseif opt.model == 'linear' then
      model = createLinear(classes)

   elseif opt.model == 'agent' then
	  print(opt.actions)
	  print(opt.hiddenSize)
      model = createAgent(classes, opt.hiddenSize, opt.actions, opt.nStep,execute_fn)

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)



----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
if opt.model == 'agent' then
   criterion = nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
      :add(nn.ModuleCriterion(nn.VRClassReward(model, opt.rewardScale), nil, nn.Convert())) -- REINFORCE
else
   criterion = nn.ClassNLLCriterion()
end

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

--denoise paremeters
denoise_paremeters={}

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end
		
	  -- reset denoise perameters
	  if opt.noise=='1' then
			print('Initialize denoise paremeters!!!')
			for i=1,opt.batchSize do
				denoise_paremeters[i]={tolerance=0.2, tau=0.125, tv_weight=100}
		 	end
	  end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )

            -- clip gradient if use Reinforcement Learning
            if opt.m == 'agent' then
               gradParameters:clamp(-5,5)
            end
         end

         -- update confusion
         if opt.model ~= 'agent' then
            for i = 1,opt.batchSize do
               confusion:add(outputs[i], targets[i])
            end
         else
            for i = 1,opt.batchSize do
               confusion:add(outputs[1][i], targets[i])
            end
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. dataset:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, dataset:size())

      else
         error('unknown optimization method')
      end
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end


-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end
		
	  -- reset denoise perameters
	  if opt.noise=='1' then
			for i=1,opt.batchSize do
				denoise_paremeters[i]={tolerance=0.2, tau=0.125, tv_weight=100}
		 	end
	  end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   train(trainData)
   test(testData)

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end
		