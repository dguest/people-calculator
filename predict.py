#!/usr/bin/env python3

from keras.models import load_model
from sys import argv
from deep_learn import index_participents, people

model = load_model(argv[1])

test_arr = index_participents(people)
pretty_test_arr = test_arr[0,:]
print(f'testing with {people} -> {pretty_test_arr}')

predicted = model.predict(test_arr)
print(f'we have {predicted} people')
