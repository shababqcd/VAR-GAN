import scipy.io
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import glob
import scipy.misc
import lasagne
import theano
import theano.tensor as T
import pickle
import sys
import time
from MODELSrgb import GeneratorNetwork48x48 , DiscriminatorNetwork48x48
sys.setrecursionlimit(50000)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
'''filesPoints = glob.glob("/home/gwy-dnn/Documents/Aspect2DataDATA/points/*.mat")
filesPoints.sort()'''

'''File1 = scipy.io.loadmat('DB4.mat')
images = File1['images']
images_48x48 = np.uint8(np.zeros([len(images),48,48,3]))

for ii in range(len(images)):
    images_48x48[ii,:,:,:] = scipy.misc.imresize(images[ii,:,:,:],[48,48])
    #print ii

scipy.io.savemat('DB4_48x48.mat',{'images_48x48':images_48x48})'''

File1 = scipy.io.loadmat('DB1_48x48.mat')
images1 = File1['images_48x48']
File2 = scipy.io.loadmat('DB2_48x48.mat')
images2 = File2['images_48x48']
File3 = scipy.io.loadmat('DB3_48x48.mat')
images3 = File3['images_48x48']
File4 = scipy.io.loadmat('DB4_48x48.mat')
images4 = File4['images_48x48']
images = np.concatenate((images1,images2,images3,images4),axis=0)
images = np.transpose(images,axes=[0,3,1,2])

'''allPoints = np.zeros([len(images),98])
for ii in range(len(images)):
    pointsii = scipy.io.loadmat(filesPoints[ii])
    pointsii = pointsii['test_points'] * 48.0/128.0
    pointsii = np.reshape(pointsii,[1,98])
    allPoints[ii,:] = pointsii
    print ii
scipy.io.savemat('allPoints48_48.mat',{'allPoints':allPoints})
plt.imshow(np.squeeze(images[ii , : , : , :]))
PointsOutput = np.reshape(pointsii,[49,2])
plt.scatter(x=np.squeeze(PointsOutput[:,0]),y=np.squeeze(PointsOutput[:,1]))
plt.show()'''


points = scipy.io.loadmat('allPoints48_48.mat')
points = points['allPoints']
pointsMax = np.max(points,axis=0)
pointsMin = np.min(points,axis=0)

num_epochs = 10000
#my_loss = 100000
k_t=np.float32(0)
lambda_k = 0.001
gamma = 0.5
alpha = 0.97
beta = 0.03

CONTROLDIM = 128
batch_size = 16
NN = 64
pointsMAX = matlib.repmat(pointsMax,batch_size,1)
pointsMIN = matlib.repmat(pointsMin,batch_size,1)


Gen_out_layer = GeneratorNetwork48x48(CONTROLDIM=CONTROLDIM,mini_batch_size=batch_size,NN=NN,name='Gen')
Disc_out_layer = DiscriminatorNetwork48x48(CONTROLDIM=CONTROLDIM,mini_batch_size=batch_size,NN=NN,name='Disc')

name = 'reg'
networkBi = lasagne.layers.InputLayer(shape=(None, 3, 48, 48), name=name+'Linput')
netBL1 = lasagne.layers.Conv2DLayer(networkBi, num_filters=64, filter_size=(3, 3), name=name+'L1')
netBL1MP = lasagne.layers.MaxPool2DLayer(netBL1, pool_size=(2, 2), name=name+'L1MP')
netBL2 = lasagne.layers.Conv2DLayer(netBL1MP, num_filters=64, filter_size=(3, 3), name=name+'L2')
netBL2MP = lasagne.layers.MaxPool2DLayer(netBL2, pool_size=(2, 2), name=name+'L2MP')
netBL3 = lasagne.layers.Conv2DLayer(netBL2MP, num_filters=64, filter_size=(3, 3), name=name+'L3')
netBL4 = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL3), num_units=1024, name=name+'L3')
networkBOut = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL4), num_units=98,
                                        nonlinearity=lasagne.nonlinearities.tanh, name=name+'_out_layer')

print 'BUILDING THE MODEL....'
noise_var = T.matrix(name='noise')
input_var = T.tensor4(name='inputs')
#input_varB = T.tensor4(name='input_B')
target_varB = T.fmatrix(name='target_B')
K_T = T.scalar(name='K_T')

# BEGAN variables
Disc_output_real = lasagne.layers.get_output(Disc_out_layer,inputs=input_var)
Disc_output_fake = lasagne.layers.get_output(Disc_out_layer,
                                              inputs=lasagne.layers.get_output(Gen_out_layer,
                                                                               inputs=noise_var))
Gen_output = lasagne.layers.get_output(Gen_out_layer,inputs=noise_var)

# Classifier variables
NetB_all_input = T.concatenate([lasagne.layers.get_output(Gen_out_layer,inputs=noise_var),
                                input_var], axis=0)

NetB_output_all = T.add(lasagne.layers.get_output(networkBOut,inputs=NetB_all_input),np.finfo(np.float32).eps)

# BEGAN losses
Disc_loss_real = T.abs_(Disc_output_real - input_var)
Disc_loss_real = Disc_loss_real.mean()

Disc_loss_fake = T.abs_(Disc_output_fake - Gen_output)
Disc_loss_fake = Disc_loss_fake.mean()

Gen_loss = T.abs_(Disc_output_fake - Gen_output)
Gen_loss = Gen_loss.mean()

# Classifier losses
#NetB_loss_fake = lasagne.objectives.categorical_crossentropy(NetB_output_fake,targets_fake)
#NetB_loss_fake = NetB_loss_fake.mean()

#NetB_loss_all = lasagne.objectives.squared_error(NetB_output_all,target_varB)
NetB_loss_all = T.log(T.add(T.abs_(target_varB-NetB_output_all)+1,np.finfo(np.float32).eps))
NetB_loss_all = NetB_loss_all.mean()

# Total Losses

Disc_loss_T = Disc_loss_real - K_T * Disc_loss_fake

Gen_loss_T = alpha * Gen_loss + beta * NetB_loss_all

NetB_loss_T = NetB_loss_all

# Parameters

Disc_params = lasagne.layers.get_all_params(Disc_out_layer , trainable=True)

Gen_params = lasagne.layers.get_all_params(Gen_out_layer , trainable=True)

NetB_params = lasagne.layers.get_all_params(networkBOut , trainable=True)

# updates
updates = lasagne.updates.adam(Gen_loss_T,Gen_params,learning_rate=0.0001,beta1=.5,beta2=0.999)
updates.update(lasagne.updates.adam(Disc_loss_T,Disc_params,learning_rate=0.0001,beta1=.5,beta2=0.999))
updates.update(lasagne.updates.nesterov_momentum(NetB_loss_T,NetB_params,learning_rate=0.01,momentum=0.9))

print 'COMPILING THE MODEL... PLEASE WAIT....'
TrainFunction = theano.function([noise_var,input_var,K_T,target_varB],
                                [Disc_loss_T,
                                 Disc_loss_real,
                                 Gen_loss,
                                 NetB_loss_T,
                                 Disc_loss_real + T.abs_(gamma * Disc_loss_real - Gen_loss)],updates=updates)

GEN_LOSS = np.array([])
DISC_LOSS = np.array([])
NETB_LOSS = np.array([])
M_GLOBAL = np.array([])

m_global_value_best = 999999999999999
print 'TRAINING STARTED...'
for epoch in range(num_epochs):
    train_error = 0
    gen_loss_value = 0
    disc_loss_value = 0
    m_global_value = 0
    net_b_loss = 0
    start_time = time.time()
    dummy = 0
    #carry = np.float32(1)
    for batch in iterate_minibatches(images,points,batch_size,shuffle=True):
        dummy = dummy + 1
        inputsNum, targetsNum = batch
        seeds = np.float32(np.random.rand(batch_size,CONTROLDIM)*2-1)

        #targetsNum = np.float32((targetsNum / 48.0)*2.0-1.0)
        targetsNum = np.float32(np.divide((targetsNum - pointsMIN),(pointsMAX-pointsMIN))*2.0-1.0)
        seeds[:,0:98] = targetsNum
        targetsNum = np.concatenate((targetsNum,targetsNum),axis=0)
        inputsNum = np.float32(inputsNum/np.float32(255) * 2.0 - 1.0)
        disc_error, Disc_Loss_real_Num, gen_error, NetB_Loss_Num, M_Global_Num = TrainFunction(seeds, inputsNum, k_t,
                                                                                               targetsNum)
        k_t = np.float32(np.clip(k_t + lambda_k * (gamma * Disc_Loss_real_Num - gen_error), 0, 1))
        gen_loss_value += gen_error
        disc_loss_value += disc_error
        m_global_value += M_Global_Num
        net_b_loss += NetB_Loss_Num
        if dummy % 100 == 0:
            print 'Iteration ' + str(dummy + 1) + ' finished successfully.'
        # if m_global_value<m_global_value_best:
        # m_global_value_best=m_global_value
    with open('netBestDiscriminatorCELEBA_BEGAN3singleGenWithRegAttempt3Epoch' + str(
            epoch) + '.pickle', 'wb') as handle:
        print('saving the model Discriminator....')
        pickle.dump(Disc_out_layer, handle)

    with open('netBestGeneratorCELEBA_BEGAN3singleGenWithRegAttempt3Epoch' + str(
            epoch) + '.pickle', 'wb') as handle:
        print('saving the model Generator....')
        pickle.dump(Gen_out_layer, handle)


    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    # print("  training loss acc:\t\t{}".format(train_error/10))
    print("  training loss generator:\t\t{:.6f}".format(gen_loss_value / 10))
    print("  training loss discriminator:\t\t{:.6f}".format(disc_loss_value / 10))
    print("  training m_global:\t\t{:.6f}".format(m_global_value))
    GEN_LOSS = np.append(GEN_LOSS, gen_loss_value)
    DISC_LOSS = np.append(DISC_LOSS, disc_loss_value)
    M_GLOBAL = np.append(M_GLOBAL, m_global_value)
    NETB_LOSS = np.append(NETB_LOSS, net_b_loss)
    scipy.io.savemat('LossesCELEBA_BEGAN3singleGenWithRegAttempt3.mat',
        mdict={'genLoss': GEN_LOSS, 'discLoss': DISC_LOSS, 'Mglobal': M_GLOBAL, 'netBloss': NETB_LOSS})


print 'yup'