import numpy as np
from pysc2.lib import actions

from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, concatenate, LSTM, Softmax
from keras.models import Model, model_from_json
import sc2_agent as sc2Agent

class ConvAgent ( sc2Agent.Agent ):
    def __init__(self, envParams ):
        self.welcomeStr = 'CONV-AGENT'
        self.learningStrategyStr = 'backprop'
        self.architectureStr = 'im a simple conv agent'
        self.envParams = envParams
        self.policyInds = {}
        self.bringup()                
        return
    
    def build_model ( self ):        
        # screen/visual features
        spatialInput = Input( shape = ( self.envParams['screenChannelsRetained'] \
                                       * self.envParams['nStackedFrames'], 
                                       self.envParams['screenResX'], 
                                       self.envParams['screenResY'] ), dtype=np.float32 )

        # non-visual features (e.g., supply, current reward, cumulative score )
        nonSpatialInput = Input ( shape = ( self.envParams['nonSpatialInputDimensions'] \
                                           * self.envParams['nStackedFrames'],) 
                                 , dtype=np.float32)
        
        
        ''' feature building convolutional layers 
            these will hold a shared representation used for both action and value selection'''
        firstConv = Conv2D ( filters = 32,
                             kernel_size = 4,
                             activation = 'relu',
                             padding = 'same',
                             data_format = 'channels_first',
                             dilation_rate = 1, strides = 1 ) ( spatialInput )

        secondConv = Conv2D ( filters = 32,
                             kernel_size = 4,
                             activation = 'relu',
                             padding = 'same',
                             data_format = 'channels_first',
                             dilation_rate = 1, strides = 1 ) ( firstConv )
        
        ''' spatial action judgments will be made with two convolutional layers 
            this pair of filters (1 per coordinate) will be softmax transformed so as to contain...
                ...a probabilistic representation of the choice for the coordinate arguments 
            the first point in the action argument ( x1, y1 ) will be sampled from firstCoordinateConv 
            the second point in the action argument ( x2, y2 ) will be sampled from secondCoordinateConv 
        '''        
        firstCoordinateConv = Conv2D( 1, 3, activation = 'relu', 
                                     padding = 'same', 
                                     data_format = 'channels_first') (secondConv)
        secondCoordinateConv = Conv2D(1, 3, activation = 'relu', 
                                      padding = 'same',
                                      data_format = 'channels_first') (secondConv) 
        
        # flatten and softmax
        flattenedFirstCoordinateConv = Flatten() (firstCoordinateConv)
        softmaxFlattenedFirstCoordinateConv = Softmax()(flattenedFirstCoordinateConv)
        flattenedSecondCoordinateConv = Flatten() (secondCoordinateConv)
        softmaxFlattenedSecondCoordinateConv = Softmax()(flattenedSecondCoordinateConv)

        
        ''' linear and non-linear controllers -- used to make value and non-spatial action judgements '''
        flattenedVisuaFeatures = Flatten() ( secondConv )
        mergeSpatialNonSpatial = concatenate( [ nonSpatialInput, flattenedVisuaFeatures ], axis = 1)

        linearControllerLayer1 = Dense ( 512, activation = 'linear' ) ( mergeSpatialNonSpatial )
        linearControllerLayer2 = Dense ( 512, activation = 'linear' ) ( linearControllerLayer1 )
        linearControllerLayer3 = Dense ( 512, activation = 'linear' ) ( linearControllerLayer2 )
        
        nonlinearControllerLayer1 = Dense ( 512, activation = 'tanh' ) ( mergeSpatialNonSpatial )
        nonlinearControllerLayer2 = Dense ( 512, activation = 'tanh' ) ( nonlinearControllerLayer1 )
        nonlinearControllerLayer3 = Dense ( 512, activation = 'tanh' ) ( nonlinearControllerLayer2 )
        
        linearNonlinearConcat = concatenate( [ linearControllerLayer3, nonlinearControllerLayer3 ] )
        
        # outputs
        value = Dense ( 1, activation = 'linear' ) ( linearNonlinearConcat )
        
        actionID = Dense ( self.envParams['prunedActionSpaceSize'], 
                          activation = 'softmax' ) ( linearNonlinearConcat )
        
        actionModifierQueue = Dense ( 1, activation = 'sigmoid' ) ( linearNonlinearConcat )
        actionModifierSelect = Dense ( 1, activation = 'sigmoid' ) ( linearNonlinearConcat )
        
        finalLayerConcat = concatenate( [ value, 
                                          actionID, 
                                          actionModifierQueue,
                                          actionModifierSelect,
                                          softmaxFlattenedFirstCoordinateConv, 
                                          softmaxFlattenedSecondCoordinateConv], axis = 1)
        
        self.model = Model ( inputs = [ spatialInput, nonSpatialInput ], outputs = [ finalLayerConcat ] ) 
        self.policySize = self.model.layers[-1].output_shape[1]
        
        self.policyInds = {}
        # book-keeping for subsequent parsing of model output
        self.policyInds['value'] = 0
        self.policyInds['actionDistStart'] = 1
        self.policyInds['actionDistEnd'] = self.policyInds['actionDistStart'] \
                                                + len(self.envParams['allowedActionIDs'])
            
        self.policyInds['actionModifierQueue'] = self.policyInds['actionDistEnd']
        self.policyInds['actionModifierSelect'] = self.policyInds['actionModifierQueue'] + 1
        
        self.policyInds['actionCoord1Start'] = self.policyInds['actionModifierSelect'] + 1 
        self.policyInds['actionCoord1End'] = self.policyInds['actionCoord1Start'] \
                                                    + self.envParams['screenResX'] * self.envParams['screenResY']
            
        self.policyInds['actionCoord2Start'] = self.policyInds['actionCoord1End']
        self.policyInds['actionCoord2End'] = self.policyInds['actionCoord2Start'] \
                                                    + self.envParams['screenResX'] * self.envParams['screenResY'] 
        
        self.model.compile ( optimizer = 'adam', loss = sc2Agent.compute_trajectory_loss)
        
        return self.model