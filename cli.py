import argparse
import hashlib
import numpy
import pickle
import os.path
import math
import random
import time
import sys
from model.elman import model
from preprocessing.index import WordIndex, LabeledData
from preprocessing.simplesrl import SimpleSrl
from util.util import context_window, minibatch, shuffle

parser = argparse.ArgumentParser(description='Create, train or access an elman network for simple semantic role labelling.')
parser.add_argument('--train', '-t', nargs='*', help='Specifies none/one or multiple paths of simplesrl files for training')
parser.add_argument('--net', '-n', help='Load an existing SRL elman network by specifying its path.')
parser.add_argument('-P', type=bool, help='Whether or not to print parameters of the loaded network or specified settings file.')
parser.add_argument('-c', nargs='*', help='A sentence to classify with the created/given network')
parser.add_argument('--classify', '-C', nargs='*', help='None, one or multiple files of sentences to classify with the created/given network')
parser.add_argument('--output', '-o', help='Path to output the network to (must be an folder)')
parser.add_argument('--settings', '-s', type=open, help='A file to load the settings from (pickle)')
parser.add_argument('--save', '-S', help='Whether or not to overwrite the existing network (if specified)')
parser.add_argument('--index', '-i', help='The index database file for words and labels. It will be extended or created if specified.', default='index.db')
parser.add_argument('--dumpIndex', '-D', help='An optional path to dump the (prob. merged) used index to.')
parser.add_argument('--epochs', '-e', type=int, help='Number of epochs for training', default=20)
parser.add_argument('--decay', type=bool, help='Whether or not the learning rate should decay', default=True)
parser.add_argument('--decayAfter', type=int, help='Number of passed epochs with no change the learning rate should decay', default=5)
parser.add_argument('--options', '-O', type=open, help='Extra file to load options from (pickle).')
args = parser.parse_args()


if args.net is not None:
    # Load existing network
    folder = args.net
    if not os.path.isdir(folder):
        raise Exception('Specified path "%s" is no directory.' % folder)

    # Load settings
    settingsFile = os.path.join(folder, 'settings.pickle')
    if not os.path.isfile(settingsFile):
        raise Exception('Specified net contains no settings file "%s"' % settingsFile)
    settings = pickle.load(open(settingsFile, 'rb'))

    # Load indices
    indicesFile = os.path.join(folder, 'indices.pickle')
    if not os.path.isfile(indicesFile):
        raise Exception('Specified net contains no indices file "%s"' % indicesFile)
    indices = pickle.load(open(indicesFile, 'rb'))

    print('Loading SRL elman network from "%s"' % folder)
    rnn = model.load(folder)
    print('Network loaded.')
else:
    # Create new network
    rnn = None

    # Build up new indices
    print('Building new indices.')
    indices = {
        'wordIndex': WordIndex(),
        'labelIndex': WordIndex()
    }

    # Load or build up settings array
    if args.settings is None:
        print('Using default settings.')
        settings = {
            'partialTraining': 0.95,
            'partialTesting': 0.05,
            'fold': 3, # 5 folds 0,1,2,3,4
            'lr': 0.063, # learning rate
            'verbose': 1,
            'win': 7, # number of words in the context window
            'bs': 9, # number of backprop through time steps
            'nhidden': 40, # number of hidden units
            'seed': 345,
            'emb_dimension': 100, # dimension of word embedding
        }
    else:
        settings = pickle.load(args.settings)

print('Current configuration is %s' % settings)


# Optionally load existing word & label index
indicesPath = args.index
if os.path.isfile(indicesPath):
    print('Loading index from "%s" and merging it into existing one.' % indicesPath)
    base = pickle.load(open(indicesPath, 'rb'))
    indices['wordIndex'].merge(base['wordIndex'])
    indices['labelIndex'].merge(base['labelIndex'])
else:
    print(indicesPath)

wordIndex = indices['wordIndex']
labelIndex = indices['labelIndex']


# Classification: load files and modify index
classify = []
if args.classify is not None:
    for file in args.classify:
        if not os.path.exists(file):
            sen = file.split(' ')
            if len(sen) > 1:
                # Use the argument as sentence for classification if there is more than one space in it
                classify.append(sen)
            else:
                print('File "%s" for classification does not exist. Ignoring.' % file)
                continue

        # This is not efficient for large files, but I want to extend it optionally with sentences from other
        # sources, so lets just read in 'readable' files ;)
        sentences = [line.split(' ') for line in file.readlines()]
        classify.extend(sentences)

# Classification: add implicitly given sentences
if args.c is not None:
    for sen in args.c:
        sen = sen.split(' ')
        if len(sen) > 1:
            classify.append(sen)
        else:
            print('Sentence for classification given with option -c does not contain any spaces. Ignoring.')

# Extend word index with classification sentences
if len(classify) > 0:
    if rnn is None:
        print('Extending word index with words from classification sentences.')
        wordIndex.addSentences(classify)
    else:
        print('Watch out: classification might only work with sentences, that were already given into your network before.')

    print('Collected %s sentences for classification.' % len(classify))


# Training: load files and modify index
if args.train is None:
    print('No training will be performed.')
else:
    # Load files for training
    data = LabeledData()

    hashAlgo = hashlib.md5
    for file in args.train:
        if not os.path.exists(file):
            print('File "%s" not found. Ignoring.' % file)
            continue

        sourcePath = file
        srlPath = file + '.index'
        hashPath = file + '.md5'

        currentDigest = hashAlgo(open(sourcePath, 'rb').read()).hexdigest()
        oldDigest = None

        if os.path.isfile(hashPath):
            oldDigest = pickle.load(open(hashPath, 'rb'))

        if currentDigest == oldDigest and os.path.isfile(srlPath):
            # Use existing index
            print('Using existing database for "{0}"'.format(sourcePath))
            srlData = pickle.load(open(srlPath, 'rb'))
        else:
            # Create new index
            print('Reading and creating database for "{0}"'.format(sourcePath))
            srlData = SimpleSrl(sourcePath)
            srlData.setIgnoreErrors(True)
            srlData.read()
            pickle.dump(srlData, open(srlPath, 'wb'))
            pickle.dump(currentDigest, open(hashPath, 'wb'))

        # Display errors if there are some
        if len(srlData.getErrors()) > 0:
            print('')
            print('Errors in "{0}"'.format(sourcePath))
            print('----------------------------------------------')
            for err in srlData.getErrors():
                print(err)
            print("\n\n")

        data.addData(srlData.getLabeledData())

    # Get all sentences and their labels
    sentencesList, labelList = data.getData()
    numberLabeledSentences = len(sentencesList)

    # Shuffle it once before splitting it up
    shuffle([sentencesList, labelList], settings['seed'])

    sizeTraining = int(math.floor(settings['partialTraining']*numberLabeledSentences))
    sizeTesting = int(math.floor(settings['partialTesting']*numberLabeledSentences))
    print('Size training: [0:{0}] = {0}'.format(sizeTraining))
    train_sentences = sentencesList[0:sizeTraining]
    train_labels = labelList[0:sizeTraining]
    print('Size testing: [{0}:{1}] = {2}'.format(sizeTraining, sizeTraining+sizeTesting, sizeTesting))
    test_sentences = sentencesList[sizeTraining:sizeTraining+sizeTesting]
    test_labels = labelList[sizeTraining:sizeTraining+sizeTesting]

    if rnn is None:
        # Insert words from sentences into index
        wordIndex.addSplittedSentences(sentencesList)

        # Insert label names from labeled sentences into index
        labelIndex.addSplittedSentences(labelList)
    else:
        print('Watch out: training an existing network only works with words, that were previously added in index.')


# Persist indices outside of the network?
if args.dumpIndex is not None:
    print('Persisting index in "%s".' % args.dumpIndex)
    base = { 'wordIndex': wordIndex, 'labelIndex': labelIndex }
    pickle.dump(base, open(args.dumpIndex, 'wb'))


label2index = labelIndex.getCurrentIndex()
index2label = labelIndex.getIndex2Word()
word2index = wordIndex.getCurrentIndex()
index2word = wordIndex.getIndex2Word()


vocsize = len(word2index)
nclasses = len(label2index)


if rnn is None:
    # Create new model
    print('Building model.')
    numpy.random.seed(settings['seed'])
    random.seed(settings['seed'])
    rnn = model(nh = settings['nhidden'],
                nc = nclasses,
                ne = vocsize,
                de = settings['emb_dimension'],
                cs = settings['win'])
    rnn.setup()


# Perform training
if args.train is not None:
    numberTrainSentences = len(train_sentences)
    numberOfTrainLabelsToGuess = sum([len(x) for x in test_labels])
    numberOfTrainLabelsToGuessListWeighted = [[3 if y == 'SUB' else 1 if y == 'REST' else 2 for y in x] for x in test_labels]
    numberOfTrainLabelsToGuessWeighted = (sum([sum(x) for x in numberOfTrainLabelsToGuessListWeighted]))

    print('Starting training with {0} labeled sentences in total for {1} epochs.'.format(numberTrainSentences, args.epochs))
    currentLearningRate = settings['lr']
    bestMeasure = -numpy.inf
    bestEpoch = 0
    for epoch in xrange(args.epochs):
        # Start epoch
        print('')
        print('Epoch {0}'.format(epoch))
        print('----------------------------------------------')
        shuffle([train_sentences, train_labels], settings['seed'])
        tic = time.time()
        for i in xrange(len(train_sentences)):
            indexedSentence = [word2index[w] for w in train_sentences[i]]
            indexedLabeledSentence = [label2index[l] for l in train_labels[i]]
            contextWindow = context_window(indexedSentence, settings['win'])
            words  = map(lambda x: numpy.asarray(x).astype('int32'),
                         minibatch(contextWindow, settings['bs']))
            for word, label in zip(words, indexedLabeledSentence):
                #print "[TRAIN]: %s (%s)" % (word, label)
                rnn.train(word, label, currentLearningRate)
                rnn.normalize()

            print '[learning] epoch %i >> %2.2f%%'%(epoch,(i+1)*100./numberTrainSentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
            sys.stdout.flush()

        print("\n")

        # evaluation // back into the real world : idx -> words
        predictions_test = [ map(lambda x: index2label[x],
                             rnn.classify(numpy.asarray(context_window(x, settings['win'])).astype('int32')))\
                             for x in [[word2index[word] for word in sentence] for sentence in test_sentences]]

        for sen, pred, lab in zip(test_sentences, predictions_test, test_labels):
            print(sen)
            print('Pred: %s' % pred)
            print('Orig: %s' % lab)
            print('')

        correctGuessesList = [[1 if pred_val == exp_val else 0 for pred_val, exp_val in zip(pred, exp)] for pred, exp in zip(predictions_test, test_labels)]
        correctGuessesListWeighted = [[3 if exp_val == 'SUB' and pred_val == exp_val else 1 if exp_val == 'REST' and pred_val == exp_val else 2 if exp_val == pred_val else 0 for pred_val, exp_val in zip(pred, exp)] for pred, exp in zip(predictions_test, test_labels)]

        correctGuesses = (sum([sum(x) for x in correctGuessesList]))
        correctGuessesWeighted = (sum([sum(x) for x in correctGuessesListWeighted]))
        accurracy = correctGuesses*100./numberOfTrainLabelsToGuess
        print('Accurracy (number of correct guessed labels) at %2.2f%%.'% accurracy)
        print('Accurracy (weighted; SUB=3, REST=1, other=2) at %2.2f%%.'%((correctGuessesWeighted*100./numberOfTrainLabelsToGuessWeighted)))

        # Now check if the current epoch is the best one measured so far
        if accurracy > bestMeasure:
            bestEpoch = epoch
            bestMeasure = accurracy

        if args.decay and abs(bestEpoch-epoch) >= args.decayAfter:
            currentLearningRate *= 0.5
            print('Decaying learning rate to %.5f' % currentLearningRate)
        if currentLearningRate < 1e-5: break

    print 'BEST RESULT: epoch', bestEpoch, 'with best measure', bestMeasure, '.'


# Perform classification
if len(classify) > 0:
    print('')
    print('Classification')
    print('----------------------------------------------')
    print('')
    #print(classify)
    #print([[word2index[word] for word in sentence] for sentence in classify])
    #print(rnn.classify([numpy.asarray(context_window(x, settings['win'])).astype('int32') for x in [[word2index[word] for word in sentence] for sentence in classify]][0]))
    classification = [ map(lambda x: index2label[x],
                         rnn.classify(numpy.asarray(context_window(x, settings['win'])).astype('int32')))\
                         for x in [[word2index[word] for word in sentence] for sentence in classify]]
    for sen, clas in zip(classify, classification):
        print('Sentence: %s' % sen)
        print('Classify: %s' % clas)
        print('')


# Persist network?
if args.output is not None:
    folder = args.output
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print('Persisting SRL elman network to "%s"' % folder)
    else:
        print('Overwriting files in "%s"' % folder)

    pickle.dump(settings, open(os.path.join(folder, 'settings.pickle'), 'wb'))
    pickle.dump({ 'wordIndex': wordIndex, 'labelIndex': labelIndex }, open(os.path.join(folder, 'indices.pickle'), 'wb'))
    rnn.save(folder)