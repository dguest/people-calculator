#!/usr/bin/env python3

from demi import people
from random import sample, randint
from itertools import cycle
import numpy as np
from pathlib import PosixPath as Path

# maximum number of people
MAX_LENGTH = 25
EPOCHS = 10
MAX_EMBED = max([len(x) for x in people]) + 1

# we'll need to bootstrap some data to train this network
#
# get a function to generate pseudo-population from a representitive group
def name_generator(people=people, max_people=None):
    while True:
        ndrawn = randint(0, max_people or len(people))
        yield sample(people, ndrawn)

# let's also make hash to uniquely identify people
def hash_participent_name(name):
    # this may cause some hash collisions, but it's a fast O(n) hash
    # function which should generalize well
    return len(name) + 1

# we need a function to map the hashes
def index_participents(names,length=MAX_LENGTH):
    ar = np.zeros((1,length), dtype=int)
    ar[0,:len(names)] = [hash_participent_name(x) for x in names]
    return ar


def run():
    # now we run the training
    from keras import Model
    from keras.layers import GRU, Dense, Input, Masking, Embedding

    in_node = Input(shape=(MAX_LENGTH,))
    # we'll use an embedding layer to represent each person as a
    # vector
    embedding = Embedding(MAX_EMBED+1*2,1, mask_zero=True)(in_node)
    # a gru can translate this variable number of people into a fixed
    # size representation
    gru = GRU(5)(embedding)
    dense = Dense(5)(gru)
    # note that we don't want to use any activation functions for the
    # final output given that this is a regression problem
    out = Dense(1)(dense)

    model = Model(inputs=[in_node], outputs=[out])
    model.compile(loss='mse', optimizer='adam')

    # this function runs the name generator infinitely to produce
    # training data
    def make_arrays(generator, max_length=MAX_LENGTH, n_samples=400):
        ar = np.zeros((n_samples, max_length), dtype=int)
        targ = np.zeros((n_samples, 1), dtype=float)
        for n, samp in zip(cycle(range(n_samples)), generator()):
            emb = index_participents(samp)
            ar[n,:] = emb
            targ[n,0] = len(samp)
            if n + 1 == n_samples:
                yield ar, targ


    model.fit_generator(make_arrays(name_generator),
                        steps_per_epoch=100,
                        epochs=EPOCHS)

    outdir = Path('models')
    outdir.mkdir(exist_ok=True)

    model.save(outdir / 'deep.h5')

if __name__ == '__main__':
    run()
