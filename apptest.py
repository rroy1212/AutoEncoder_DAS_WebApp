from __future__ import division, print_function
from random import randint
import os
from time import strftime
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

import time, datetime
import os
import obspy
from obspy.io.segy.core import _read_segy
from obspy.signal.tf_misfit import cwt
from obspy.imaging.cm import obspy_sequential,obspy_divergent
from obspy.signal.trigger import z_detect,plot_trigger
import argparse
import obspy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, load, dump
import tempfile
from scipy import signal
from subprocess import call



# coding=utf-8
import sys
import os
import glob
import re
import matplotlib.pyplot as plt
#from keras import backend as K
from tensorflow.keras import backend as K

import numpy as np

# Keras
import tensorflow as tf
from tensorflow.python.keras import datasets
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer





def getChannelSpecgram(datatype, traceList, sampleRate, outfile, channelStart, channelStep=10):

    """Generate spectrogram for a single channel
    one picture for each channel
    @param datatype, 'mat' or 'segy'
    @param a list of traces for segy or numpy array for mat
    @param sampleRate, rate of sampling
    @param outfile, name of the output
    """
    assert(datatype in ['mat', 'segy'])
    if datatype=='segy':
        st = obspy.Stream(traceList)            
        nTraces = len(traceList)
    else:
        raise Exception('not implemented')
    
    nfft = np.min([1024/4, len(st[0].data)])
    window = nfft
    path=[]
    imagelist=[]

    frac_overlap = 0.1
    for itr in range(0,nTraces,channelStep):
        F,T,SXX = signal.spectrogram(st[itr].data, fs=sampleRate)
        S1 = np.log10(np.abs(SXX/np.max(SXX)))*10.0
        plt.figure()
        plt.pcolormesh(T, F, S1)
        print (channelStart+itr) 
        #path.append('spectrogram_plots/test/tracespectrogram' + str(channelStart) +'_'+ str(itr) + '.png')
        #print(path)
        imgname = 'tracespectrogram_{0}_ch{1}.png'.format(outfile, channelStart+itr)
        imagelist.append(imgname)
        #imgname=tracespectrogram_{0}_ch{1}.png'.format(outfile, channelStart+itr)
        filename='static/Obspy_Plots/test/tracespectrogram_{0}_ch{1}.png'.format(outfile, channelStart+itr)
        path.append(filename)
        plt.savefig('static/Obspy_Plots/test/tracespectrogram_{0}_ch{1}.png'.format(outfile, channelStart+itr))
        plt.close()
        #datafile = 'spec_npy/tracespectrogram_{0}_ch{1}.npy'.format(outfile,channelStart+itr)
        #np.save(datafile,S1)
    return imagelist
    """
    #this uses obspy spectogram
    st[50].spectrogram(samp_rate=sample_rate, 
                              per_lap = frac_overlap,
                              wlen = window,
                              dbscale=True, 
                              log=True)
    for itr in range(0,100,10):
        st[itr].spectrogram(log=True, cmap='copper')
        plt.savefig('tracespectrogram_{0}_{1}.png'.format(self.filename, itr))
    """


def filterSingleTrace(tr, *args):

    ioption = args[0]
    if ioption=='downsample':
        #do downsampling only
        downsampleFactor=args[1]
        if downsampleFactor>1:
            #tr.resample(sampling_rate=downsampleFactor)
            tr.decimate(factor=downsampleFactor,no_filter = True)            
    elif ioption=='bandpass':
        #do bandpass filtering and downsampling
        fmin = args[1]
        fmax = args[2]
        downsampleFactor=args[3]

        tr.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)
        if downsampleFactor>1:
            #tr.resample(sampling_rate=downsampleFactor)
            tr.decimate(factor=downsampleFactor,no_filter = True)

    elif ioption=='lowpass':
        #apply lowpass filter
        fmax = args[1]
        downsampleFactor=args[2]
        tr.filter('lowpass', freq=fmax, corners=2, zerophase=True) 
        if downsampleFactor>1:
            #tr.resample(sampling_rate=downsampleFactor)        
            tr.decimate(factor=downsampleFactor,no_filter = True)

def getSingleTrace(tr, downsampleFactor=10, isIntegrate=False):
    """
    this is used to process a single trace
    """
    if isIntegrate:
        tr.integrate()
        
    #hardcoding bandpass filter here
    filterSingleTrace(tr, 'bandpass', 1.0, 100.0, downsampleFactor)
                    
    return tr

class PoroTomo():

    def __init__(self, segyfile, channelRange, frameWidth, 
                 samplingRate = 1000,
                 downsampleFactor=100,
                 stackInterval = 1,
                 isIntegrate=False):    
  
        iloc = segyfile.find('iDAS')
        self.filename = segyfile[iloc:-4]
        self.load(segyfile)
        
        self.gather = None
        self.frameWidth = frameWidth
        self.dsfactor = downsampleFactor
        self.stackInterval = stackInterval
        self.channelRange = np.arange(channelRange[0],channelRange[1])

        self.isIntegrate = isIntegrate
        
        self.sampRate = samplingRate/self.dsfactor
        
        self._getGather()
        
        #interval of sampling
        self.dt = 1.0/self.sampRate
        
    def _getGather(self):
        """
        This is the main function that gathers all traces and form a station
        """
        if self.gather is None:
            print ('loading traces')
            if DEBUG:
                start_time = time.time()

            nChannels = len(self.channelRange)
            traceList = [None]*nChannels
            
            #detrend all traces
            self.st.detrend('constant')
            #taper all traces on both sides
            self.st.taper(max_percentage=0.01)
            #process traces in parallel
            with Parallel(n_jobs=12) as parallelPool:
                traceList = parallelPool(delayed(getSingleTrace)
                                        (self.st[channelNo].copy(),                                                     
                                        self.dsfactor,
                                        self.isIntegrate)
                             for channelNo in self.channelRange)

            #Turn tracelist into numpy array
            tempmat = np.zeros((nChannels,len(traceList[0].data)),dtype=np.float64)
            for ic in range(len(traceList)):
                tempmat[ic,:] = traceList[ic].data
            print ('temp data shape', tempmat.shape)
            #stacking is a way for boosting the signal strength
            if self.stackInterval>1:
                nStacks = int(nChannels/self.stackInterval)            
                stackedArr = np.zeros((nStacks, len(traceList[0].data)),dtype=np.float64)
                for iStack in range(nStacks):
                    stackedArr[iStack,:] = np.sum(tempmat[iStack:(iStack+1)*self.stackInterval, :], axis=0)            
            else:
                stackedArr = tempmat
                
            self.traceList = traceList
            self.stackedArr = stackedArr
            if DEBUG:
                print ('number of traces = {0}, number of samples each trace={1}'.format(nChannels, len(traceList[0].data)))
                print ('processing data took ', time.time()-start_time)
                plt.figure()
                #plt.imshow(self.stackedArr,cmap = 'bwr', origin='lower', aspect=10.0)   
                plt.savefig('gather_plot{0}.png'.format(self.filename))
                plt.close()




    def gen_tfplot(self,channelNo):
        tr = self.st[channelNo].copy()


        # Filtering with a lowpass on a copy of the original Trace
        tr_filt = tr.copy()
        tr_filt.filter('lowpass', freq=1.0, corners=2, zerophase=True)

        # Now let's plot the raw and filtered data...
        t = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)
        plt.subplot(211)
        plt.plot(t, tr.data, 'k')
        plt.ylabel('Raw Data')
        plt.subplot(212)
        plt.plot(t, tr_filt.data, 'k')
        plt.ylabel('Lowpassed Data')
        plt.xlabel('Time [s]')
        #plt.suptitle(tr.stats.starttime)
        img_name='TF_plot_'+str(channelNo) +'.png'
        path='static/Obspy_Plots/test/' + img_name
        plt.savefig(path)
        #plt.show()
    
        return img_name


    def spectrogramPlot(self, channelNo):
        """
        Test spectrogram plot for a single channel
        """
        tr=self.st[channelNo].copy()
        #tr.filter('lowpass', freq=10.0, corners=2, zerophase=True)
        #tr.decimate(factor=10, strict_length=False)
        filterSingleTrace(tr, 'decimate', 10)
        plt.figure()
        tr.spectrogram(show=False)
        image_name='spectrogram_{0}.png'.format(channelNo)
        path='static/Obspy_Plots/test/' + image_name
        plt.savefig('spectrogram_{0}.png'.format(channelNo))

        return image_name

    def detectEvent(self,channelNo, methodName='z_detect'):  
        tr = self.st[channelNo].copy()        
        if methodName=='z_detect':
            df = tr.stats.sampling_rate
            cft = z_detect(tr.data, int(10 * df))
            plot_trigger(tr, cft, -0.4, -0.3, show=False)
            plt.title('Channel No.' + str(channelNo))
            plot_name='trigger_detection_'+str(channelNo) + '.png'
            path='static/Obspy_Plots/test/' +plot_name
            plt.savefig(path)
            return plot_name

    def cwtPlot(self, channelNo):
        """
        Test cwt for a single trace
        """
        tr = self.st[channelNo].copy()
        npts = tr.stats.npts
        dt = tr.stats.delta
        t = np.linspace(0, dt * npts, npts)
        f_min = 1
        f_max = 50
        
        scalogram = cwt(tr.data, dt, 8, f_min, f_max)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        x, y = np.meshgrid(
            t,
            np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
        
        ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
        ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_yscale('log')
        ax.set_ylim(f_min, f_max)
        img_name='cwt{0}_.png'.format(channelNo)
        plt.savefig('static/Obspy_Plots/test/cwt{0}_.png'.format(channelNo))  
        
        return img_name
                
    def load(self, segyfile):
        """
        Load seg-y file
        """
        print ('loading ', segyfile)
        if DEBUG:
            start_time = time.time()
        self.st = _read_segy(segyfile)
        if DEBUG:
            print ('loading data took ', time.time()-start_time)
        self.nTrace = len(self.st)
        if DEBUG:
            print ('number of traces is ', self.nTrace)
            print ('data in each trace is', len(self.st[0].data))

def get_obspy_plot(minchannelrange,maxchannelrange,framelen,samplingrate,dsfactor,filename):
    channelRange=[minchannelrange,maxchannelrange]
    stackInterval=1
    tom = PoroTomo(filename, 
                   channelRange, 
                   framelen, 
                   samplingRate=samplingrate,
                   downsampleFactor=dsfactor, 
                   stackInterval=stackInterval,
                   isIntegrate=True)
    plot_arr=[]
    imagelist=getChannelSpecgram('segy', tom.traceList, tom.sampRate, tom.filename, channelRange[0])
    plot_arr.append(imagelist[0])
    cwt_name=tom.cwtPlot(channelRange[0])
    plot_arr.append(cwt_name)
    event_name=tom.detectEvent(channelRange[0])
    plot_arr.append(event_name)
    tf_name=tom.gen_tfplot(channelRange[0])
    plot_arr.append(tf_name)
    print(plot_arr)
    plot_dict={'spect':imagelist[0],'cwt_name':cwt_name,'event_name':event_name,'tf_name':tf_name}
    return plot_dict
    
# Model saved with Keras model.save()
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
#MODEL_PATH = 'models/model.h5'

json_file = open('models/model_gpu.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load woeights into new model
loaded_model.load_weights("models/model_gpu_weights.h5")
graph = tf.compat.v1.get_default_graph()
loaded_model.compile(optimizer='adadelta', loss='mean_squared_error')
# Necessary
print('Model loaded. Start serving...')

def model_predict(img_path, loaded_model):
    
    #image preprocessing
    print("model_predict",img_path)
    
    batch_size = 1
    test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
    anomaly_generatortest = test_datagen.flow_from_directory(
    img_path,
    target_size=(96, 96),
    batch_size=batch_size,
    class_mode='input'
    )
    
    anomaly_data_listA = []
    batch_index = 0
    while batch_index <= anomaly_generatortest.batch_index:
        dataA = anomaly_generatortest.next()
        anomaly_data_listA.append(dataA[0])
        batch_index = batch_index + 1
    
    ###### Anomaly Detection #####  
    img_data= anomaly_data_listA[0][0]   
    reconstruction_error_threshold = 0.06 # This threshold was chosen based on looking at the distribution of reconstruction errors of the normal class
    reconstruction = loaded_model.predict([[img_data]])
    reconstruction_error = loaded_model.evaluate([reconstruction],[[img_data]], batch_size = 1)
    
    print(f'reconstruction_error: {reconstruction_error}')
    if reconstruction_error > reconstruction_error_threshold:
        preds="This is an anomaly"
        #plt.imshow(img_data)
        #plt.show()
    else:
        preds="This is not an anomaly"
        #plt.imshow(img_data)
        #plt.show()

 
    return preds

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SjdnUends821Jsdlkvxh391ksdODnejdDw'
SPEC_FOLDER = os.path.join('static', 'Obspy_Plots','test')
# TRIGGER_FOLDER = os.path.join('static', 'Trigger_Events')
# TF_FOLDER = os.path.join('static', 'TF_Plots')
# CWT_FOLDER = os.path.join('static', 'CWT_Plots')

print(SPEC_FOLDER)
app.config['UPLOAD_FOLDER'] = SPEC_FOLDER


test=[]
filename=""
img_path=[]
image_name=""
model_name=""



@app.route("/", methods=['GET', 'POST'])
def segydata():
    form = request.form
    segy_files = ['PoroTomo_iDAS16043_160321000521.sgy', 'PoroTomo_iDAS16043_160321000721.sgy', 'PoroTomo_iDAS16043_160321000921.sgy']
    filter_type=['Low Pass','High Pass','Band Pass']
    return render_template('index.html',form=form,files=segy_files,filters=filter_type)

@app.route("/success", methods=['GET', 'POST'])
def processdata():
    global image_name
    print("----------process")
    form = request.form

        #print(form.errors)
        #print(request.method)
    if request.method == 'POST':
            #f = request.files['file']  
            #f.save(f.filename)
        print("----------process")
        input_file = request.form['files']
        minchannelrange=request.form['minchannelrange']
        maxchannelrange=request.form['maxchannelrange']
        framelen=request.form['framelen']
        samplingrate=request.form['samplingrate']
        dsfactor=request.form['dsfactor']
            
        filename='PoroTomo_iDAS16043_160321000721.sgy'
        plot_dict = get_obspy_plot(int(minchannelrange),int(maxchannelrange),int(framelen),int(samplingrate),int(dsfactor),filename)

        spect_name = plot_dict['spect']
        cwt_name = plot_dict['cwt_name']
        event_name = plot_dict['event_name']
        tf_name = plot_dict['tf_name']
        image_name=spect_name
        print(image_name)

 
           
    spec_full_filename = os.path.join(app.config['UPLOAD_FOLDER'], spect_name)
    CWT_full_filename = os.path.join(app.config['UPLOAD_FOLDER'], cwt_name)
    TF_full_filename = os.path.join(app.config['UPLOAD_FOLDER'], event_name)
    event_full_filename = os.path.join(app.config['UPLOAD_FOLDER'], tf_name)
    print("full_filename is " ,spec_full_filename)

    test_data={'spec':spec_full_filename,'cwt':CWT_full_filename,'tf':TF_full_filename,'event':event_full_filename}
    return render_template('obspy_plot.html', data = test_data) 

@app.route("/upload",methods=['POST'])
def upload_model():
    return render_template('model_upload.html') 

@app.route("/predict",methods=['GET', 'POST'])
def anomaly_detection():
    if request.method == 'POST':
        
        global image_name
        global model_name
        f = request.files['file']
        filename = f.filename
        option = request.form['options']
        model_name = option
        path='model_uploads/' + str(filename)
        f.save(path)
       
        spec_full_filename = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        print(spec_full_filename)
        img_path='static/Obspy_Plots/'
        # Make prediction       
        preds = model_predict(img_path, loaded_model)
        test_data={'path':spec_full_filename,'model':model_name,'result':preds}
        return render_template('anomaly_detection.html', data = test_data) 

if __name__ == "__main__":
    app.run(use_reloader=False)
