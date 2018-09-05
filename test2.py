import sys
import numpy as np
import numpy.matlib as matlib
import os
import scipy.misc
import scipy.io
import theano
import theano.tensor as T
import lasagne
import time
import pickle
import matplotlib.pyplot as plt

batch_size = 16
CONTROLDIM=128

points = scipy.io.loadmat('allPoints48_48.mat')
points = points['allPoints']
pointsMax = np.max(points,axis=0)
pointsMin = np.min(points,axis=0)
#pointsMAX = matlib.repmat(pointsMax,batch_size,1)
#pointsMIN = matlib.repmat(pointsMin,batch_size,1)

points = points[3,:]#+np.random.rand(98)*1.3
class Unpool2DLayer(lasagne.layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(ds[0], axis=2).repeat(ds[1], axis=3)






POINTS = np.zeros([batch_size,98])
POINTSreal = np.zeros([batch_size,98])
for ii in range(batch_size):
    Uncertainty = np.float32(np.random.rand(98) * 2 - 1)*.5
    POINTSreal[ii, :] = points + Uncertainty
    POINTS[ii,:] = np.float32(np.divide((POINTSreal[ii,:] - pointsMin),(pointsMax-pointsMin))*2.0-1.0)

seeds = np.float32(np.random.rand(batch_size, CONTROLDIM)*2-1)
for ii in range(batch_size):
    #seeds[ii,0:98] = np.float32(np.divide((points - pointsMin),(pointsMax-pointsMin))*2.0-1.0)
    seeds[ii,0:98] = POINTS[ii,:]


with open('netBestGeneratorCELEBA_BEGAN3singleGenWithRegAttempt3Epoch80.pickle','rb') as handle:
    nno = pickle.load(handle)

OUTPUT_image = lasagne.layers.get_output(nno,(seeds))
imageOut = OUTPUT_image.eval()
imageOut = (imageOut - np.min(imageOut))/(np.max(imageOut)-np.min(imageOut))

f2, ((axx1, axx2 , axx3), (axx4, axx5, axx6), (axx7, axx8, axx9)) = plt.subplots(3, 3, sharey=True)
axx1.imshow(np.transpose(np.squeeze(imageOut[0 , : , : , :]),axes=[1,2,0]))
#PointsOutputFlat = lasagne.layers.get_output(nno,(seeds[0,:]))
#PointsOutputFlat = PointsOutputFlat.eval() * 128
PointsOutput = np.reshape(POINTSreal[1,:],[49,2])
axx1.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
axx2.imshow(np.transpose(np.squeeze(imageOut[1 , : , : , :]),axes=[1,2,0]))
#PointsOutputFlat = lasagne.layers.get_output(nno,(seeds[1,:]))
#PointsOutputFlat = PointsOutputFlat.eval() * 128
#PointsOutput = np.reshape(PointsOutputFlat,[49,2])
PointsOutput = np.reshape(POINTSreal[2,:],[49,2])
axx2.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
axx3.imshow(np.transpose(np.squeeze(imageOut[2 , : , : , :]),axes=[1,2,0]))
#PointsOutputFlat = lasagne.layers.get_output(nno,(seeds[2,:]))
#PointsOutputFlat = PointsOutputFlat.eval() * 128
#PointsOutput = np.reshape(PointsOutputFlat,[49,2])
PointsOutput = np.reshape(POINTSreal[3,:],[49,2])
axx3.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
axx4.imshow(np.transpose(np.squeeze(imageOut[3 , : , : , :]),axes=[1,2,0]))
#PointsOutputFlat = lasagne.layers.get_output(nno,(seeds[3,:]))
#PointsOutputFlat = PointsOutputFlat.eval() * 128
#PointsOutput = np.reshape(PointsOutputFlat,[49,2])
PointsOutput = np.reshape(POINTSreal[4,:],[49,2])
axx4.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
axx5.imshow(np.transpose(np.squeeze(imageOut[4 , : , : , :]),axes=[1,2,0]))
#PointsOutputFlat = lasagne.layers.get_output(nno,(seeds[4,:]))
#PointsOutputFlat = PointsOutputFlat.eval() * 128
#PointsOutput = np.reshape(PointsOutputFlat,[49,2])
PointsOutput = np.reshape(POINTSreal[5,:],[49,2])
axx5.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
axx6.imshow(np.transpose(np.squeeze(imageOut[5 , : , : , :]),axes=[1,2,0]))
#PointsOutputFlat = lasagne.layers.get_output(nno,(seeds[5,:]))
#PointsOutputFlat = PointsOutputFlat.eval() * 128
#PointsOutput = np.reshape(PointsOutputFlat,[49,2])
PointsOutput = np.reshape(POINTSreal[6,:],[49,2])
axx6.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
axx7.imshow(np.transpose(np.squeeze(imageOut[6 , : , : , :]),axes=[1,2,0]))
#PointsOutputFlat = lasagne.layers.get_output(nno,(seeds[6,:]))
#PointsOutputFlat = PointsOutputFlat.eval() * 128
#PointsOutput = np.reshape(PointsOutputFlat,[49,2])
PointsOutput = np.reshape(POINTSreal[7,:],[49,2])
axx7.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
axx8.imshow(np.transpose(np.squeeze(imageOut[7 , : , : , :]),axes=[1,2,0]))
#PointsOutputFlat = lasagne.layers.get_output(nno,(seeds[7,:]))
#PointsOutputFlat = PointsOutputFlat.eval() * 128
#PointsOutput = np.reshape(PointsOutputFlat,[49,2])
PointsOutput = np.reshape(POINTSreal[8,:],[49,2])
axx8.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
axx9.imshow(np.transpose(np.squeeze(imageOut[8 , : , : , :]),axes=[1,2,0]))
#PointsOutputFlat = lasagne.layers.get_output(nno,(seeds[8,:]))
#PointsOutputFlat = PointsOutputFlat.eval() * 128
#PointsOutput = np.reshape(PointsOutputFlat,[49,2])
PointsOutput = np.reshape(POINTSreal[9,:],[49,2])
axx9.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
plt.show()
scipy.io.savemat('out6VARGAN.mat',{'images':imageOut,'points':POINTSreal})
print 'yup'