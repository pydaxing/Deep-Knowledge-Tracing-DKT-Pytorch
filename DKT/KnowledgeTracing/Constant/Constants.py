Dpath = '../../KTDataset'

datasets = {
    'assist2009' : 'assist2009',
    'assist2015' : 'assist2015',
    'assist2017' : 'assist2017',
    'static2011' : 'static2011',
    'kddcup2010' : 'kddcup2010',
    'synthetic' : 'synthetic'
}

numbers = {
    'assist2009' : 124,  ###
    'assist2015' : 100,
    'assist2017' : 102,
    'static2011' : 1224, ###
    'kddcup2010' : 661,  ###
    'synthetic' : 50
}


DATASET = datasets['static2011']
NUM_OF_QUESTIONS = numbers['static2011']
MAX_STEP = 50
BATCH_SIZE = 64
LR = 0.002
EPOCH = 1000
INPUT = NUM_OF_QUESTIONS * 2
EMBED = NUM_OF_QUESTIONS
HIDDEN = 200
LAYERS = 1
OUTPUT = NUM_OF_QUESTIONS
