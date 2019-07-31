from model.model import get_red30_model
from model.util  import log2lin, lin2log

class predictor:
    
    def __init__(self, weights, logspace, label="", verbose=False):
        """
        Object instantiation
        @param self    : this instance of the class
        @param weights : string of the path to the hdf5 file containing the model weights
        @param logspace: boolean indicating whether the loaded weights require input in logspace
        @param label   : a string label for the model
        @param verbose : boolean indicating whether to print verbose output
        """
        self._logspace = logspace
        self.label     = label
        self._model    = get_red30_model()
        self.verbose   = verbose
        
        self._model.load_weights(weights)
    
    def getLogspace(self):
        """
        Method for checking if the model requires logspace input
        """
        return self._logspace
    
    
    def predict(self, image, image_logspace):
        """
        Generate a prediction using the model
        @param image         : the input image as np.array from which to make the prediction
        @param image_logspace: boolean indicating if the provided image has been converted to logspace
        @returns prediction  : the model prediction as np.array
        """
        assert(image.shape == (1, 256, 256, 2), "Expected shape (1, 256, 256, 2), received %s" % str(image.shape))
        
        # TODO check that image is normalized
        if self._logspace != image_logspace:
            if self._logspace:
                if self.verbose: print('Converting input lin2log')
                image = lin2log(image)
            else:
                if self.verbose: print('Converting input log2lin')
                image = log2lin(image)        
        
        prediction = self._model.predict(image)
        
        # always return in linspace
        if self._logspace:
            if self.verbose: print('Converting output log2lin')
            prediction = log2lin(prediction)
            
        return prediction
