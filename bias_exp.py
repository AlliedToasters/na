import numpy as np
import pandas as pd
from sklearn import preprocessing
from annoy import AnnoyIndex
from PIL import Image
from PIL.ImageEnhance import Brightness, Color, Sharpness, Contrast
from PIL.ImageOps import mirror
import matplotlib.pyplot as plt
from IPython.display import clear_output

import os
import random
from random import shuffle
import string
import math
import pickle

from keras.callbacks import Callback, ModelCheckpoint
from keras_vggface import VGGFace
from keras.layers import Flatten, Input, Dense
from keras.models import Model, load_model


class UniqueIDMaker(object):
    """Returns unique ids; never produces the same
    id.
    """
    def __init__(self, n_chars=5):
        self.n_chars = n_chars
        self.issued = set()
        self.choices = string.ascii_lowercase + string.ascii_uppercase + '0123456789'

    def make_id(self):
        result = ''
        for i in range(self.n_chars):
            result += random.choice(self.choices)
        return result
    
    def __call__(self):
        new_id = self.make_id()
        while new_id in self.issued:
            new_id = self.make_id()
        self.issued.add(new_id)
        return new_id
        

def get_image(filepath):
    """Opens image and converts to numpy array."""
    img = Image.open(filepath)
    return np.array(img)

class BatchMaker(object):
    """Makes batches and keeps track of the mean per-channel pixel
    values for convenience.
    """
    def __init__(self, ch0_mean, ch1_mean, ch2_mean):
        self.c0m = ch0_mean
        self.c1m = ch1_mean
        self.c2m = ch2_mean
        
    def make_batch(self, images):
        """Takes a list of images and converts it into a batch."""
        for img in images:
            img[:, :, 0] = img[:, :, 0] - self.c0m
            img[:, :, 1] = img[:, :, 1] - self.c1m
            img[:, :, 2] = img[:, :, 2] - self.c2m
        images = [x/255 for x in images] 
        images = [np.expand_dims(x, axis=0) for x in images]
        batch = np.concatenate(images, axis=0)
        return batch

def build_df(base_dir='./ids/'):
    dirs = os.listdir(base_dir)
    ids = [x.split('_')[-1] for x in dirs]
    df = pd.DataFrame()
    df['id'] = ids
    df['dir'] = [base_dir + x + '/' for x in dirs]
    df['sex'] = [x.split('_')[1] for x in dirs]
    df['race'] = [x.split('_')[0] for x in dirs]
    df['age'] = [x.split('_')[2] for x in dirs]
    df['n_images'] = df.dir.apply(lambda x: len(get_imgfiles(x)))
    return df

def get_imgfiles(directory):
    """Returns a list of all image files in a directory."""
    img_extensions = [
        '.jpg',
        '.png',
        '.bmp'
    ]
    all_files = [directory + x for x in os.listdir(directory)]
    files = []
    for fl in all_files:
        if any([(x in fl) for x in img_extensions]):
            files.append(fl)
    return files

def build_batchmaker(df):
    """Goes through all images and computes mean color channel values. Returns
    a BatchMaker object, with these values stored.
    """
    ch0 = []
    ch1 = []
    ch2 = []
    for i, row in df.iterrows():
        files = get_imgfiles(row.dir)
        images = [get_image(fl) for fl in files]
        for img in images:
            ch0.append(img[:, :, 0].mean())
            ch1.append(img[:, :, 1].mean())
            ch2.append(img[:, :, 2].mean())
    ch0_mean = np.array(ch0).mean()
    ch1_mean = np.array(ch1).mean()
    ch2_mean = np.array(ch2).mean()
    batch_maker = BatchMaker(ch0_mean, ch1_mean, ch2_mean)
    print('got color channel values:')
    print(ch0_mean, ch1_mean, ch2_mean)
    return batch_maker

def make_features(df, batch_maker, save=True):
    """Extracts feature vectors for each id. Stores them to a
    dictionary and to disk. Returns dictionary, which stores 
    the templates for each id. Template is the mean of the normalized
    feature vectors, as specified in the VGGFace paper.
    """
    templates = dict()
    if save:
        vgf = VGGFace(include_top=False, model='resnet50', input_shape=(200, 200, 3))
    for i, row in df.iterrows():
        temp_path = row.dir + 'template.npy'
        if save:
            files = get_imgfiles(row.dir)
            images = [get_image(fl) for fl in files]
            as_batch = batch_maker.make_batch(images)
            id_ = row.id
            fts = vgf.predict(as_batch)[:, 0, 0, :]
            fts = preprocessing.normalize(fts, norm='l2')
            template = fts.mean(axis=0)
        elif not save:
            template = np.load(temp_path)
        templates[row.id] = template
        if save:
            np.save(temp_path, template)
            to_save = [fl[:-4] + '.npy' for fl in files]
            for i, savepath in enumerate(to_save):
                slc = fts[i, :]
                if save:
                    np.save(savepath, slc)
    return templates

def build_annoy_trees(df, treename='annoy_tree.ann', n_trees=100):
    """Builds annoy tree for nn search, saves tree to treename.
    The annoy tree shares indices with the dataframe rows.
    """
    f = 2048
    t = AnnoyIndex(f, metric='angular') 
    print('indexing...')
    for i, row in df.iterrows():
        emb = row.dir + 'template.npy'
        em = np.load(emb)
        t.add_item(i, em)
    print('done!')

    print('building trees...')
    t.build(n_trees)
    print('done!')
    print('saving to: ', treename)
    t.save(treename)
    return

def get_nns(df, treename='annoy_tree.ann'):
    """Builds annoy trees and does search to find nearest neighbors.
    Adds nearest neighbors to df.
    """
    build_annoy_trees(df, treename)
    f = 2048
    t = AnnoyIndex(f, metric='angular')
    t.load(treename)
    
    #add nn and distance columns to df
    nn_cols = ['nn{}'.format(x) for x in range(10)]
    for col in nn_cols:
        df[col] = int(0)
    d_cols = ['d{}'.format(x) for x in range(10)]
    for col in d_cols:
        df[col] = float(0)
        
    print('populating nearest neighbors values...')
    for i, row in df.iterrows():
        top10 = t.get_nns_by_item(i, 10)
        d10 = [t.get_distance(i, x) for x in top10]
        for ind, nn in enumerate(top10):
            d = d10[ind]
            df.at[i, 'nn{}'.format(ind)] = nn
            df.at[i, 'd{}'.format(ind)] = d
    df['min_d'] = np.where(df[d_cols].values==0, 1, df[d_cols].values).min(axis=1)
    print('done!')
    return df

def get_faces(df, idx):
    """Given df and idx, returns a list of all images
    belonging to that id.
    """
    dirname = df.loc[idx].dir
    files = get_imgfiles(dirname)
    faces = []
    for fl in files:
        face = Image.open(fl)
        faces.append(face)
    return faces

def view_nearest_neighbors(df, idx):
    """Takes a df and index and plots the nearest neighbors."""
    faces = []
    for i in range(10):
        neighbor_idx = df.loc[idx]['nn{}'.format(i)]
        face = get_faces(df, neighbor_idx)[0]
        faces.append(face)
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle('Nearest Neighbors for {}'.format(idx))
    fcnum = 0
    for i, axi in enumerate(ax):
        for j, axj in enumerate(axi):
            ax[i, j].imshow(faces[fcnum])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_title(round(df.at[idx, 'd{}'.format(fcnum)], 4))
            fcnum += 1
    for img in faces:
        img.close()
    plt.show();
    
def get_contents(df, idx):
    """Returns a list of images and a list of feature vectors
    that correspond to the files.
    """
    dirname = df.loc[idx].dir
    files = get_imgfiles(dirname)
    faces = []
    fts = []
    for fl in files:
        face = Image.open(fl)
        ft = np.load(fl[:-4]+'.npy')
        faces.append(face)
        fts.append(ft)
    return faces, fts
        
def recalc_template(df, idx):
    """Recalculates the etc/template.npy file."""
    dirname = df.loc[idx].dir
    files = get_imgfiles(dirname)
    vecs = [np.load(x[:-4] + '.npy') for x in files]
    vecs = [np.expand_dims(x, axis=0) for x in vecs]
    vecs = np.concatenate(vecs, axis=0)
    template = vecs.mean(axis=0)
    savename = dirname + 'template.npy'
    np.save(savename, template)

def merge_identities(df, id1, id2):
    """Merges similar identities. id2 is deleted, and
    its photo is added to id1. ids are indices.
    All feature descriptors in resulting identities are re-averaged, 
    and template files overwritten.
    """
    try:
        newdir = df.loc[id1].dir
    except KeyError:
        print('first identity not found.')
        raise KeyError
    files1 = get_imgfiles(newdir)
    try:
        faces, fts = get_contents(df, id2)
    except KeyError:
        print('second identity not found.')
        raise KeyError
    for i, face in enumerate(faces):
        newnum = str(len(files1)+1)
        newname = newdir + newnum + '.png'
        newvec = newdir + newnum + '.npy'
        face.save(newname)
        np.save(newvec, fts[i])
    recalc_template(df, id1)
    os.system('rm -R {}'.format(df.loc[id2].dir))
    df = df.drop(id2)
    return df

def view_all_imgs(df, idx):
    """Displays all of the images for a given identity."""
    faces = get_faces(df, idx)
    dirname = df.loc[idx].dir
    file_iter = iter(get_imgfiles(dirname))
    n_faces = len(faces)
    nrows = n_faces//5
    ncols = min(5, len(faces))
    if n_faces % 5 != 0:
        nrows += 1
    face_iter = iter(faces)
    fig, axes = plt.subplots(1, ncols)
    fig.suptitle('identity {}'.format(df.loc[idx].id))
    skip = False
    try:
        no_axes = len(axes)
    except:
        axes.set_title(next(file_iter).split('/')[-1])
        axes.set_xticks([])
        axes.set_yticks([])
        axes.imshow(faces[0])
        skip = True
    if not skip:
        for ax in axes:
            if type(ax) == type(list()):
                for ax_ in ax:
                    ax.set_title(next(file_iter).split('/')[-1])
                    ax_.set_xticks([])
                    ax_.set_yticks([])
                    try:
                        ax_.imshow(next(face_iter))
                    except:
                        pass
            else:
                ax.set_title(next(file_iter).split('/')[-1])
                ax.set_xticks([])
                ax.set_yticks([])
                try:
                    ax.imshow(next(face_iter))
                except:
                    pass
    plt.show();
    return

def filter_by_nn(df, min_d=.004):
    """Removes any rows where min_d is less than argument min_d"""
    df = df[df.min_d > min_d].copy()
    return df

def sample_by_subgroups(df, age_groups=[(20, 39),(40,)], n_class=2622, seed=None):
    """Returns a dataframe with even splits between race, gender, 
    and age groups. Age groups must be specified and passed in
    as as list of tuples."""
    df = df[~(df.age=='None')]
    groups = []
    under = np.where((df.age.astype(int) < age_groups[0][0]), True, False)
    print(len(df[under]), ' under {}'.format(age_groups[0][0]))
    tk = under
    for group in age_groups:
        if len(group) == 1:
            grp = np.where(~tk, True, False)
            print(len(df[grp]), ' above {}'.format(group[0]))
        else:
            grp = np.where(((df.age.astype(int)<group[1])&(~tk)), True, False)
            print(len(df[grp]), ' {}-{}'.format(group[0], group[1]))
            tk = np.where(grp, True, tk)
        groups.append(grp)
    sexes = ['male', 'female']
    races = ['asian', 'black', 'caucasian', 'indian']
    classes = []
    sample_no = int(n_class/16)
    for sex in sexes:
        for race in races:
            for i, age in enumerate(groups):
                try:
                    classes += list(df[age&(df.race==race)&(df.sex==sex)].sample(sample_no, random_state=seed).index)
                except:
                    classes += list(df[age&(df.race==race)&(df.sex==sex)].sample(frac=1, random_state=seed).index)
                    print('short: ', race, sex, age_groups[i])
                    print('we have:', len(df[age&(df.race==race)&(df.sex==sex)]))
                    print('we need: ', sample_no)
    print(len(classes))
    return df.loc[classes].copy()

def flip(img, plot=False):
    """flip image. Creates a mirror image of face."""
    rev = mirror(img)
    if plot:
        fig, ax = plt.subplots(1, 2)
        for ax_ in ax:
            ax_.set_xticks([])
            ax_.set_yticks([])
        ax[0].set_title('train')
        ax[1].set_title('test')
        ax[0].imshow(img)
        ax[1].imshow(rev)
        plt.show();
    return rev

def one_hot_encode(df, id_, class_coder):
    """One-hot encodes an input class."""
    cls = class_coder[id_]
    result = np.zeros(len(df))
    result[cls] = 1
    return result

def get_face(df, idx):
    """Given df and idx, returns a list of all images
    belonging to that id.
    """
    dirname = df.loc[idx].dir
    file = dirname + '1.jpg'
    face = Image.open(file)
    return face

from PIL.ImageEnhance import Brightness, Color, Sharpness, Contrast

def augment(img):
    """Performs augmentation on input image."""
    bright = Brightness(img)
    factor = np.random.random()/2 + .75
    img = bright.enhance(factor)
    color = Color(img)
    factor = np.random.random()/2 + .75
    img = color.enhance(factor)
    sharp = Sharpness(img)
    factor = np.random.random()/2 + .75
    img = sharp.enhance(factor)
    cont = Contrast(img)
    factor = np.random.random()/2 + .75
    img = cont.enhance(factor)
    rot = np.random.randint(-5, 5)
    img = img.rotate(rot)
    return img

def make_image_batch(df, idxs, class_coder, batch_maker, train=False):
    """Returns a batch of given indices"""
    slc = df.loc[idxs]
    lbls = list(slc.id)
    filenames = [x+'1.jpg' for x in slc.dir]
    X = []
    Y = []
    for i, fn in enumerate(filenames):
        img = Image.open(fn)
        if train:
            img = augment(img)
        elif not train:
            img = flip(img)
        X.append(np.array(img))
        Y.append(np.expand_dims(one_hot_encode(df, lbls[i], class_coder), axis=0))
    Y = np.concatenate(Y, axis=0)
    X = batch_maker.make_batch(X)
    return X, Y

def datagen(df, batch_maker, batch_size=64, train=False, do_shuffle=False):
    """Generates data."""
    class_coder = dict()
    cnt = -1
    for i, row in df.iterrows():
        cnt += 1
        class_coder[row.id] = cnt
        class_coder[cnt] = row.id
    yield class_coder
    batch = []
    while True:
        if do_shuffle:
            df = df.sample(frac=1)
        for i, row in df.iterrows():
            batch.append(i)
            if len(batch) == batch_size:
                X, Y= make_image_batch(df, batch, class_coder, batch_maker, train=train)
                yield X, Y
                batch = []

def plot_history(hst, ticker='test'):
    """Makes plot from training history"""
    val_loss = np.array([])
    loss = np.array([])
    val_acc = np.array([])
    acc = np.array([])
    for history in hst:
        val_loss = np.concatenate([val_loss, history.history['val_loss']])
        loss = np.concatenate([loss, history.history['loss']])
        val_acc = np.concatenate([val_acc, history.history['val_acc']])
        acc = np.concatenate([acc, history.history['acc']])
    plt.plot(val_loss, label='val loss')
    plt.plot(loss, label='train loss')
    plt.title('loss optimization')
    plt.legend()
    plt.savefig('./plots/loss{}.png'.format(ticker))
    plt.clf()
    plt.plot(val_acc, label='val accuracy')
    plt.plot(acc, label='train accuracy')
    plt.title('accuracy optimization')
    plt.legend()
    plt.savefig('./plots/acc{}.png'.format(ticker))
    plt.clf()

class IpynbPlotter(Callback):
    """Plots optimization progress at each epoch end."""
    
    def __init__(self):
        self.loss = []
        self.vloss = []
        self.acc = []
        self.vacc = []
        self.history = dict()
        self.history['val_loss'] = self.vloss
        self.history['loss'] = self.loss
        self.history['val_acc'] = self.vacc
        self.history['acc'] = self.acc
        super().__init__()
    
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs['loss'])
        self.vloss.append(logs['val_loss'])
        self.acc.append(logs['acc'])
        self.vacc.append(logs['val_acc'])
        #clear_output()
        plot_history([self])
        return

def perform_experiment(df, batch_maker, ticker=0):
    """Performs experiment on input dataset. Outputs dataframe with
    results.
    """
    n_class = len(df)

    
    inputs = Input((200, 200, 3))
    x = VGGFace(include_top=False, model='resnet50', input_shape=(200, 200, 3))(inputs)
    x = Flatten()(x)
    out = Dense(
        n_class, 
        activation='softmax'
    )(x)
    model = Model(inputs=inputs, outputs=out)

    for layer in model.layers[1].layers:
        layer.trainable=False

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )


    batch_size = 64
    steps = n_class//batch_size

    train_gen = datagen(df, batch_maker, batch_size=batch_size, train=True, do_shuffle=True)
    class_coder = next(train_gen)
    val_gen = datagen(df, batch_maker, batch_size=batch_size, train=False)
    next(val_gen)
    
    callbacks = []
    #callbacks.append(IpynbPlotter())
    modelpath = 'model.h5'
    callbacks.append(ModelCheckpoint(filepath=modelpath, save_best_only=True, monitor='val_acc'))

    history = []

    history.append(
        model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=25,
            verbose=0,
            validation_data=val_gen,
            validation_steps=steps,
            callbacks=callbacks
        )
    )
    _ = record_results(df, modelpath, val_gen, class_coder, steps, ticker)
    return history

def record_results(df, modelpath, test_gen, class_coder, steps, ticker='test', results_path='./results.csv'):
    """Takes the modelpath, loads the model, runs tests,
    and records results.
    """
    best_model = load_model(modelpath)
    try:
        results = pd.read_csv('./results.csv')
    except FileNotFoundError:
        results = pd.DataFrame(columns=['gt_id', 'pred_id'])
    test_data = []
    for i in range(steps+1):
        test_data.append(next(test_gen))

    new_results = pd.DataFrame(columns=['class', 'pred'])
    for dt in test_data:
        pred = best_model.predict(dt[0])
        Y = dt[1]
        chunk = pd.DataFrame(columns=results.columns)
        chunk['class'] = np.argmax(Y, axis=1)
        chunk['pred'] = np.argmax(pred, axis=1)
        new_results = pd.concat([new_results, chunk], axis=0, sort=True)

    gt_ids = []
    pred_ids = []
    for i, row in new_results.iterrows():
        gt = row['class']
        gt_label = class_coder[gt]
        gt_ids.append(gt_label)
        prediction = row['pred']
        pred_label = class_coder[prediction]
        pred_ids.append(pred_label)
        
    record = pd.DataFrame(columns=['gt_id', 'pred_id'])
    record['gt_id'] = gt_ids
    record['pred_id'] = pred_ids

    record = record.drop_duplicates()
    
    results = pd.concat([results, record], axis=0, sort=True)
    results.to_csv('./results.csv', index=False)
    return results

if __name__ in '__main__':
    
    df = build_df('./ids/')
    batch_maker = build_batchmaker(df)
    templates = make_features(df, batch_maker, save=False)
    df = get_nns(df)
    
    df1 = filter_by_nn(df)
    
    df1.to_csv('first_experiment_labels.csv', index=False)
    
    history = []
    
    for i in range(100):
        sampled = sample_by_subgroups(df1, seed=i)
        history += perform_experiment(sampled, batch_maker, ticker=i)
        plot_history([history[-1]], ticker=i)