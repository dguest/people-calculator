#!/usr/bin/env python3

from keras.models import load_model
from sys import argv
from deep_learn import embed, people
from random import sample
import numpy as np
from pathlib import PosixPath as Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvasPdf as Canvas

model = load_model(argv[1])

x = []
y = []

for n in range(len(people)*2):
    randos = sample(people, min(n,len(people)))
    if n > len(people):
        randos += sample(people, n - len(people))
    test_arr = embed(randos)
    predicted = model.predict(test_arr)
    x.append(n)
    y.append(predicted.flatten())

fig = Figure((4,3))
Canvas(fig)
plot = fig.add_subplot(111)
plot.set_xlabel('ground truth')
plot.set_ylabel('model')
plot.plot(x,y, '.')
plot.axvspan(10,20, color='red', alpha=0.5)

out_path = Path('figures')
out_path.mkdir(exist_ok=True)

fig.savefig(out_path / 'test.pdf',bbox_inches='tight')

