import time, os, sys
import librosa
import librosa.display
from keras.models import load_model
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from PIL import Image


def loading_bar(start_time, progress, barLen=40):
    d = barLen * progress
    s=(d+0.000001)/(time.time()-start_time+0.000001)
    if d == barLen:
        sys.stdout.write('\r')
        sys.stdout.write("Progress: |{:<{}}| {:.2f}% | ETA: {:.2f}s | Elapsed Time: ~{:.2f}s".format("░" * int(d), 
                                                                                   barLen, 
                                                                                   progress * 100,
                                                                                   (barLen-d)/s, time.time()-start_time))
        sys.stdout.flush()
    else:
        sys.stdout.write('\r')
        sys.stdout.write("Progress: |{:<{}}| {:.2f}% | ETA: {:.2f}s ".format("░" * int(d), 
                                                                                   barLen, 
                                                                                   progress * 100,
                                                                                   (barLen-d)/s))
        sys.stdout.flush()
    del start_time, progress, barLen, d, s



def load_file(filename):
    try:
        x, sample_rate=librosa.load(filename, res_type='kaiser_fast')
        del filename
    except:
        return None, None
    return x, sample_rate


def load_data(root_dir, folders, file_names, class_id, err_file='UNSUCCESSFUL_LOAD.txt', err_enabled=None):
    data=[]
    sample_rates=[]
    lbl=[]
    err_write=None
    num=len(file_names)
    if err_enabled:
        err_write=open(err_file, 'w')
    if len(folders)==len(file_names):
        start_time=time.time()
        for i in range(num):
            file_path = os.path.join(root_dir,
                                     'fold{}'.format(folders[i]),
                                     file_names[i])
            x, sr=load_file(file_path)
            if x is not None and sr is not None:
                data.append(x)
                sample_rates.append(sr)
                lbl.append(class_id[i])
            elif err_enabled:
                err_write.write('Unabled to load file -> '+file_path+'\n')
                
            loading_bar(start_time, progress=(i+1)/num)
    else:
        print('Error: Unequal Length')
        
    if err_enabled and err_write:
        err_write.close()
    print('\n\nTotal Files:',len(file_names))
    print('Successful:',len(data))
    print('Unsuccessful:', len(file_names)-len(data),'\n')
    del root_dir, folders, file_names, class_id, err_file, err_enabled, err_write, num, start_time, file_path, x, sr
    
    return data, sample_rates, lbl


def _features(data, sample_rate, n_features):
    try:
        feature = np.mean(librosa.feature.mfcc(y=data,
                                               sr=sample_rate,
                                               n_mfcc=n_features).T,axis=0) 
        
        del data, sample_rate, n_features
        return feature
    except Exception as e:
        print(e)
    

    
def extract_features(data, sample_rates, n_features=100):
    num=len(sample_rates)
    features = np.zeros((num, n_features))
    start_time=time.time()
    for i in range(num):
        features[i, :] = _features(data[i], sample_rates[i], n_features)
        loading_bar(start_time, progress=(i+1)/num)
    del data, sample_rates, n_features, start_time, num
    return features


def one_hot_encoder(labels):
    num=len(labels)
    classes=set(labels)
    one_hot=np.zeros((num, len(classes)))
    for i in range(num):
        one_hot[i, labels[i]]=1
    del labels, classes
    return one_hot


def load_or_create_model(parent_dir, old_model=None, new_model=None, args={'out_classes': 10}):
    model=None
    if old_model:
        model_path=os.path.join('trained_models', parent_dir, old_model)
        if os.path.isfile(model_path):
                model=load_model(model_path)
        else:
            print('Error Model Not Found:','"', model_path,'"')
        del model_path
    elif new_model:
        model=new_model(**args)
    else:
        print('Error: Unknown Model')
    del parent_dir, old_model, new_model, args
    return model

def save_model(model, parent_dir, model_name):
    save_path=os.path.join('trained_models', parent_dir, model_name)
    model.save(save_path)
    del model, parent_dir, model_name, save_path

def create_spectrogram(data, sample_rate, save_as, figsize, dpi):
    plt.interactive(False)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(save_as, dpi=dpi, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del save_as, data, sample_rate, fig, ax, S
    
def create_spectrogram_batch(data, sample_rates, labels, class_ord, parent_dir='images', figsize=[0.8, 0.8], dpi=400):
    start_time=time.time()
    
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    class_count=[0]*len(class_ord)
    i=0
    num=len(labels)
    dataframe = pd.DataFrame(columns=['file_id', 'class'])
    
    for d, sr, ll, in zip(data, sample_rates, labels):
        class_count[ll]+=1
        _id=class_ord[ll]+str(class_count[ll])+'.jpg'
        dataframe.loc[i] = [_id, ll]
        save_as=os.path.join(parent_dir, _id)
        create_spectrogram(d, sr, save_as, figsize, dpi)
        i+=1
        loading_bar(start_time, progress=i/num)
    
    meta_path=os.path.join(parent_dir, 'meta_data')
    
    if not os.path.exists(meta_path):
        os.makedirs(meta_path)
        
    dataframe.to_csv(os.path.join(meta_path, 'spectrogram_images.csv'), index=False)
    del class_count,start_time, parent_dir, save_as, data, sample_rates, labels, class_ord, num, i, _id, dataframe, meta_path, figsize, dpi
    

def load_image(filepath, size):
    p = Image.open(filepath)
    p1= p.resize(size)
    return np.asarray(p1)
    
  
    
def load_spectrogram_images(parent_dir='images', meta_file='spectrogram_images.csv', size=(120, 120)):
    start_time=time.time()
    dataframe = pd.read_csv(os.path.join(parent_dir, 'meta_data', meta_file))
    files = dataframe['file_id']
    lbl = dataframe['class']
    num = len(files)
    img = []
    i=0
    for file in files:
        filepath=os.path.join(parent_dir, file)
        img.append(load_image(filepath, size))
        i+=1
        loading_bar(start_time, progress=i/num)
        
    del parent_dir, meta_file, start_time, dataframe, files, num, filepath, i
    return np.array(img), np.array(lbl)
