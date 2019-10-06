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

# get a function to generate lists of people
def name_generator(people=people, max_people=None):
    while True:
        ndrawn = randint(0, max_people or len(people))
        yield sample(people, ndrawn)

# let's also make a lookup table of people to embedding
def people_embedding(name):
    return len(name) + 1


def embed(names,length=MAX_LENGTH):
    ar = np.zeros((1,length), dtype=int)
    ar[0,:len(names)] = [people_embedding(x) for x in names]
    return ar


def run():
    from keras import Model
    from keras.layers import GRU, Dense, Input, Masking, Embedding

    in_node = Input(shape=(MAX_LENGTH,))
    embedding = Embedding(MAX_EMBED+1*2,1, mask_zero=True)(in_node)
    gru = GRU(5)(embedding)
    dense = Dense(5)(gru)
    out = Dense(1)(dense)

    model = Model(inputs=[in_node], outputs=[out])
    model.compile(loss='mse', optimizer='adam')


    def make_arrays(generator, max_length=MAX_LENGTH, n_samples=400):
        ar = np.zeros((n_samples, max_length), dtype=int)
        targ = np.zeros((n_samples, 1), dtype=float)
        for n, samp in zip(cycle(range(n_samples)), generator()):
            emb = embed(samp)
            ar[n,:] = emb
            targ[n,0] = len(samp)
            if n + 1 == n_samples:
                yield ar.copy(), targ.copy()


    model.fit_generator(make_arrays(name_generator),
                        steps_per_epoch=100,
                        epochs=EPOCHS)

    outdir = Path('models')
    outdir.mkdir(exist_ok=True)

    model.save(outdir / 'deep.h5')

if __name__ == '__main__':
    run()
