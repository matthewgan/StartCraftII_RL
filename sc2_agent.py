import numpy as np
from pysc2.lib import actions

import tensorflow as tf

def compute_trajectory_loss ( y_true, y_pred ):
    combinedLoss = tf.reduce_mean(y_true) - 0 * tf.reduce_mean(y_pred[-1])
    return combinedLoss

class Agent():
    def __init__(self, envParams ):
        self.welcomeStr = 'PLACEHOLDER-AGENT'
        self.learningStrategyStr = 'none'
        self.architectureStr = 'none'
        self.envParams = envParams
        self.bringup()
        
    def bringup ( self ):
        self.hello_world()        
        self.model = self.build_model()        
        self.initialize_placeholders()
        return
    
    def initialize_placeholders ( self ):
        nEnvs = self.envParams['simultaneousEnvironments']
        nSteps = self.envParams['nTrajectorySteps']
        
        nChannels = self.envParams['screenChannelsRetained'] \
                        * self.envParams['nStackedFrames']
        nNonSpatialInputs = self.envParams['nonSpatialInputDimensions'] \
                            * self.envParams['nStackedFrames']
            
        xRes = self.envParams['screenResX']
        yRes = self.envParams['screenResX']
        
        self.rewards = np.zeros( (nEnvs, nSteps+1), dtype=np.float32)
        self.valuePredictions = np.zeros( (nEnvs, nSteps+1), dtype=np.float32)

        self.nStepReturns = np.zeros((nEnvs, nSteps), dtype=np.float32)
        self.advantages = np.zeros((nEnvs, nSteps), dtype=np.float32)
        self.logProbs = np.zeros((nEnvs, nSteps), dtype=np.float32)
        self.entropy = np.zeros((nEnvs, nSteps), dtype=np.float32)
        
        self.loss = np.zeros((nEnvs, nSteps), dtype=np.float32)        
        
        # policy mask needs to keep track of which action arguments are active 
        self.policyMask = np.zeros( ( nEnvs, self.policySize ), dtype = np.float32)

        # Initialize placeholders for spatial and non-spatial [ stacked] trajectory observations
        self.nEnvTrajectoryBatch = np.zeros( ( nEnvs, nSteps, nChannels, xRes, yRes ), dtype=np.float32 )
        self.nEnvOneStepBatch = np.zeros( ( nEnvs, 1, nChannels, xRes, yRes ), dtype=np.float32 )

        # reward, cumulative score, player supply, enemy supply, action chosen, actionArgs
        self.nEnvTrajectoryBatchNonSpatial = np.zeros( ( nEnvs, nSteps, nNonSpatialInputs, ), dtype=np.float32 )
        self.nEnvOneStepBatchNonSpatial = np.zeros( ( nEnvs, 1, nNonSpatialInputs, ), dtype=np.float32 )
    
    # say hello & share high level architecture & learning strategy
    def hello_world( self ):
        print('hi I\'m the %s\n | architecture: %s\n | learning strategy: %s' 
              % (self.welcomeStr, self.architectureStr, self.learningStrategyStr))
        
    # define model architecture
    def build_model( self ):
        return None
       
    def model_summary( self ):
        if self.model is not None:
            return self.model.summary()
        else:
            return 'i have no model, i go where the randomness takes me'
        
    def choose_action ( self, actionProb, eGreedy = .9 ):
        if np.random.random() > eGreedy:
            if self.envParams['debugFlag']:
                print('!venturing out in action selection')
            actionID = np.random.choice( np.array( self.envParams['allowedActionIDs'], dtype=np.int ),
                                        size=1, p=np.array(actionProb) )
            actionID = actionID[0]
        else:
            if self.envParams['debugFlag']:
                print('staying greedy in action selection')
            actionID = self.envParams['allowedActionIDs'][ np.argmax( self.envParams['allowedActionIDs'] ) ]       
        return actionID
    
    def normalize_array( self, arrayInput ):        
        return (arrayInput - arrayInput.min()) / (arrayInput - arrayInput.min()).sum()
    
    def mask_unusable_actions ( self, availableActions, actionProbabilityDistribution ) :
        for iAction in range( len(actionProbabilityDistribution) ):
            if self.envParams['allowedActionIDs'][iAction] not in availableActions:
               actionProbabilityDistribution[iAction] = 0

        if not np.isclose( actionProbabilityDistribution.sum(), 1.00000 ):
            actionProbabilityDistribution = self.normalize_array( actionProbabilityDistribution )
            
        return actionProbabilityDistribution
    
    def choose_coordinate ( self, coordinateArray, eGreedy = .9  ):
        if np.random.random() > eGreedy:
            if self.envParams['debugFlag']:
                print('!venturing out in coordinate selection')
            availableCoordinates = list( range( self.envParams['screenResX'] * self.envParams['screenResY'] ))             
            chosenIndex = np.random.choice( np.array( availableCoordinates, dtype=np.int ), 
                                           size=1, p = np.array(coordinateArray) )[0]
        else:
            if self.envParams['debugFlag']:
                print('staying greedy in coordinate selection')
            chosenIndex = np.argmax( coordinateArray )
            
        maxCoord = np.unravel_index( chosenIndex, (self.envParams['screenResX'], self.envParams['screenResY']))
        return maxCoord[0], maxCoord[1]

    def sample_and_mask (self, obs, batchedOutputs ):
        batchSelectedActionFunctionCalls = [] 
         
        batchSelectedActionIDs = []
        batchSelectedActionIndexes = []
        batchSelectedActionArguments = []
        batchSelectedActionModifiers = []
        batchPredictedValues = []
        
        for iEnv in range ( self.envParams['simultaneousEnvironments'] ):
            policyIStart = self.policyInds['actionDistStart']
            policyIEnd = self.policyInds['actionDistEnd']
            point1IStart = self.policyInds['actionCoord1Start']
            point1IEnd = self.policyInds['actionCoord1End']
            point2IStart = self.policyInds['actionCoord2Start']
            point2IEnd = self.policyInds['actionCoord2End']
            
            # reset policy mask
            self.policyMask[ iEnv, : ] = 0
            
            actionProbabilityDistribution = self.mask_unusable_actions ( \
                                                obs[iEnv].observation['available_actions'], \
                                                    batchedOutputs[iEnv][ policyIStart:policyIEnd ] )
                        
            actionId = self.choose_action ( actionProbabilityDistribution )

            batchSelectedActionIDs += [ actionId ] # actionID
            actionIndex = self.envParams['allowedActionIDs'].index( actionId )
            
            self.policyMask[ iEnv, policyIStart:policyIEnd ] = 1
            
            actionArguments = []
            batchActionArguments = []
            probabilisticPointMap1 = batchedOutputs[iEnv][point1IStart:point1IEnd] 
            probabilisticPointMap2 = batchedOutputs[iEnv][point2IStart:point2IEnd] 
            
            x1, y1 = self.choose_coordinate ( probabilisticPointMap1 )
            x2, y2 = self.choose_coordinate ( probabilisticPointMap2 )
                
            if self.envParams['allowedActionIDRequiresLocation'][actionIndex] == 1:
                actionArguments = [ [ x1,  y1 ]]
                self.policyMask [ iEnv, point1IStart:point1IEnd ] = 1
                
            elif self.envParams['allowedActionIDRequiresLocation'][actionIndex] == 2:
                actionArguments = [[ x1,  y1 ], [ x2,  y2 ]]
                self.policyMask [ iEnv, point1IStart:point1IEnd ] = 1
                self.policyMask [ iEnv, point2IStart:point2IEnd ] = 1
                
            # queued
            if self.envParams['allowedActionIDRequiresModifier'][actionIndex] == 1:
                queuedActionModifier =  int( round( batchedOutputs[iEnv][ self.policyInds['actionModifierQueue']] ) ) # int   
                self.policyMask[ iEnv, self.policyInds['actionModifierQueue'] ] = 1
                actionArguments.insert( 0, [ queuedActionModifier ] )
                
            # select add
            if self.envParams['allowedActionIDRequiresModifier'][actionIndex] == 2:
                selectActionModifier =  int( round( batchedOutputs[iEnv][ self.policyInds['actionModifierSelect']] ) ) # int                
                self.policyMask[ iEnv, self.policyInds['actionModifierSelect'] ] = 1
                
                actionArguments.insert( 0, [ selectActionModifier ] )
            
            batchSelectedActionFunctionCalls += [ actions.FunctionCall( actionId, actionArguments ) ]
            batchActionArguments += [ actionArguments ]
            
            if self.envParams['debugFlag']:
                print('choosing action ' + str(actionId) + ', ' + str(actionArguments) )                          
            
        return batchSelectedActionFunctionCalls, batchSelectedActionIDs, batchActionArguments
            
        
    def batch_predict ( self, nEnvOneStepBatch, nEnvOneStepBatchNonSpatial ):
        return self.model.predict( x = [ nEnvOneStepBatch, nEnvOneStepBatchNonSpatial ], 
                                  batch_size = self.envParams['simultaneousEnvironments'] )
        
    def step_in_envs ( self, obs, localPipeEnds, batchSelectedActionFunctionCalls, batchSelectedActionIDs ):
        for iEnv in range ( self.envParams['simultaneousEnvironments'] ):
            selectedActionFunctionCall = batchSelectedActionFunctionCalls[iEnv]
            selectedActionID = batchSelectedActionIDs[iEnv]

            # ensure the agent action is possible
            ''' issue call '''
            if selectedActionID in obs[iEnv].observation['available_actions']:
                localPipeEnds[iEnv].send ( ( 'step', selectedActionFunctionCall ) )
                obs[iEnv] = localPipeEnds[iEnv].recv()                
            # take no-op action and advance to game state where we can act again                       
            else:
                localPipeEnds[iEnv].send ( ('step', actions.FunctionCall( 0, [])) )
                obs[iEnv] = localPipeEnds[iEnv].recv()
                
        return obs, 0

    def parse_rewards(self, obs):
        return [ obs[iEnv].reward for iEnv in list(obs.keys()) ]

    def inplace_update_trajectory_observations ( self, iStep, obs ): #, actionID, actionArguments ):
        for iEnv in range( self.envParams['simultaneousEnvironments'] ):            
            newObs = obs[iEnv]
            # spatial data
            self.nEnvOneStepBatch[iEnv, 0, self.envParams['screenChannelsRetained']:, :, :] = \
                self.nEnvOneStepBatch[iEnv, 0, 0:-self.envParams['screenChannelsRetained'], :, :]

            self.nEnvOneStepBatch[iEnv, 0, 0:self.envParams['screenChannelsRetained'], :, :] = \
                newObs.observation['screen'][self.envParams['screenChannelsToKeep'],:,:]

            self.nEnvTrajectoryBatch[iEnv, iStep, :, :, : ] = self.nEnvOneStepBatch[iEnv, 0, :, :, :]

            # non-spatial data
            self.nEnvOneStepBatchNonSpatial[iEnv, 0, self.envParams['nonSpatialInputDimensions']:,] = \
                self.nEnvOneStepBatchNonSpatial[iEnv, 0, 0:-self.envParams['nonSpatialInputDimensions'],]

            self.nEnvOneStepBatchNonSpatial[iEnv, 0, 0:self.envParams['nonSpatialInputDimensions'],] = \
                [ newObs.observation['game_loop'][0],  # game time
                  newObs.observation['score_cumulative'][0], # cumulative score
                  newObs.reward, # prev reward
                  newObs.observation['player'][3], # used supply
                  np.sum(newObs.observation['multi_select'][:,2]), # total multi selected unit health
                  np.sum(newObs.observation['single_select'][:,2]), # total single selected unit health
                  0, # action
                  0, # action modifier
                  0, # action coordinate x1
                  0, # action coordinate y1
                  0, # action coordinate x2
                  0 ] # action coordinate y2 

            self.nEnvTrajectoryBatchNonSpatial[ iEnv, iStep, :,] = self.nEnvOneStepBatchNonSpatial[ iEnv, 0, :,]
            
    def compute_returns_advantages ( self ):
        nextRewards = self.rewards[:, 1:]
        nextValues = self.valuePredictions[:, 1:]
        
        for iEnv in range ( self.envParams['simultaneousEnvironments']):
            # compute n-Step returns
            for iStep in reversed( range ( self.envParams['nTrajectorySteps'] ) ) :
                if iStep == ( self.envParams['nTrajectorySteps'] - 1 ) :
                    self.nStepReturns[ iEnv, iStep ] = nextValues[ iEnv, -1 ] # final return bootstrap                 
                else:
                    self.nStepReturns[ iEnv, iStep ] = nextRewards[ iEnv, iStep ] + \
                                                       self.envParams['futureDiscountRate'] \
                                                       * self.nStepReturns[ iEnv, iStep + 1 ]

            # prepare for training loop
            self.advantages[iEnv, :] = self.nStepReturns[iEnv, :] - self.valuePredictions[iEnv, 0:-1]
            
    def inplace_update_logProbs_and_entropy ( self, iStep, concatModelOutputNESS ) :
        for iEnv in range ( self.envParams['simultaneousEnvironments'] ):
            activePolicy = concatModelOutputNESS[iEnv] * self.policyMask[iEnv]
            self.logProbs[iEnv, iStep] = np.sum( -1 * np.ma.log( activePolicy ).filled(0) )
            self.entropy[iEnv, iStep] = -1 * np.sum( np.ma.log( activePolicy ).filled(0) * activePolicy )

        
    def compute_loss (self):
        self.compute_returns_advantages ( )
        for iEnv in range ( self.envParams['simultaneousEnvironments'] ):
            for iStep in range ( self.envParams['nTrajectorySteps'] ):
                policyLoss = self.advantages[iEnv, iStep] * self.logProbs[iEnv, iStep]
                valueLoss = np.square( self.nStepReturns[iEnv, iStep] - self.valuePredictions[iEnv, iStep] )/2.0
                self.loss[ iEnv, iStep] = \
                    self.envParams['policyWeight'] * policyLoss \
                    + self.envParams['valueWeight'] * valueLoss \
                    + self.envParams['entropyWeight'] * self.entropy[iEnv, iStep]
                if self.envParams['debugFlag']:
                    print( 'iEnv: ' + str(iEnv) + ' ; iStep: ' + str(iStep) )
                    print( '\t policyLossTerm: ' + str( policyLoss ))
                    print( '\t valueLossTerm: ' + str( valueLoss ))
                    print( '\t entropyLossTerm: ' + str( self.entropy[iEnv, iStep] ))
                    print( '\t totalLoss: ' + str(self.loss[ iEnv, iStep]))
                    
                if not np.isfinite( self.loss[ iEnv, iStep] ):
                    print( 'policyLossTerm: ' + str( policyLoss ))
                    print( 'valueLossTerm: ' + str( valueLoss ))
                    print( 'entropyLossTerm: ' + str( self.entropy[iEnv, iStep] ))
                    
                    raise ValueError('non-finite loss encountered')                 
    def flatten_first_dimensions ( self, inputData ):
        inputDataShape = inputData.shape
        outputShape = tuple( [inputDataShape[0]*inputDataShape[1] ] + [ i for i in inputDataShape[2:] ] )
        output = np.reshape( inputData, outputShape )
        return output 
    
    def train ( self ):
        spatialInputs = self.flatten_first_dimensions( self.nEnvTrajectoryBatch )
        nonSpatialInputs = self.flatten_first_dimensions( self.nEnvTrajectoryBatchNonSpatial )
        loss = self.flatten_first_dimensions( self.loss )
        
        verbosityLevel = 0
        if self.envParams['debugFlag']:
            verbosityLevel = 1
            
        self.model.fit( x = [ spatialInputs, nonSpatialInputs ], y = loss, verbose = verbosityLevel)
        
    def model_checkpoint( self ):
        # serialize model to JSON
        model_json = self.model.to_json()
        filePath = self.envParams['experimentDirectory'] + self.welcomeStr
        
        with open(filePath + '_model.json', 'w') as json_file:
            json_file.write(model_json)
        
        # serialize weights to HDF5
        self.model.save_weights(filePath + '_model.h5')
        print(' saved model to disk ')
        