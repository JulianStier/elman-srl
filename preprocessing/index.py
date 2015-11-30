import sys
import pickle
from preprocessing.simplesrl import read_simplesrl


class LabeledData(object):
    def __init__(self):
        self.data = [[], []]

    def getData(self):
        return self.data

    def getSentences(self):
        return self.data[0]

    def getLabels(self):
        return self.data[1]

    def addPair(self, sentence, labels):
        if len(sentence) is not len(labels):
            raise Exception('Expecting length of given pair to be equal, but sentence contains {0} words and labels {1} entries.'.format(len(sentence), len(labels)))
        self.data[0].append(sentence)
        self.data[1].append(labels)

    def addData(self, labeledData):
        if len(labeledData) is not 2:
            raise Exception('Expecting labeled data to contain exactly two lists (with corresponding pairs)')
        if len(labeledData[0]) is not len(labeledData[1]):
            raise Exception('Given lists of sentences and associated labels do not match in length.')
        self.data[0].extend(labeledData[0])
        self.data[1].extend(labeledData[1])



class WordIndex(object):
    def __init__(self):
        self.words = []
        self.words2index = {}
        self.index2words = {}
        self.changed = False

    def getSize(self):
        return len(self.words)

    def getCurrentIndex(self):
        return self.words2index

    def getIndex2Word(self):
        if self.changed or self.index2words is None:
            # Recalculate inverse index
            self.index2words = {}
            for w in self.words2index:
                self.index2words[self.words2index[w]] = w
        return self.index2words

    def loadIndex(self, indexPath):
        otherIndex = pickle.load(open(indexPath, 'r'))
        self.merge(otherIndex)

    def setChanged(self):
        self.changed = True

    def loadIndexReplace(self, indexPath):
        self.words2index = pickle.load(open(indexPath, 'r'))
        self.setChanged()

    def addWords(self, words):
        currentMax = len(self.words2index)
        for w in words:
            if not w in self.words2index:
                self.words2index[w] = currentMax
                currentMax += 1
        self.setChanged()

    def addSentences(self, sentences):
        for s in sentences:
            words = s.split()
            self.addWords(words)
        self.setChanged()

    def addSplittedSentences(self, sentences):
        for s in sentences:
            self.addWords(s)
        self.setChanged()

    def addWordsFromFile(self, filePath):
        raise Exception('not implemented yet')
        self.setChanged()

    def merge(self, otherIndex):
        currentMax = len(self.words2index)
        for word in otherIndex.words2index:
            if not word in self.words2index:
                self.words2index[word] = currentMax
                currentMax += 1
        self.setChanged()

    def storeCurrentIndex(self, indexPath):
        pickle.dump(self.words2index, open(indexPath, 'wb'))



'''
 Each entry in the list denotes one sentence (string)
'''
def create_word_index(sentences):
    index = 0
    words2index = {}

    for s in sentences:
        s = s[0]
        words = s.split()
        for w in words:
            if w not in words2index:
                words2index[w] = index
                index += 1

    return words2index



if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception('Pass file path as argument: python create_word_index.py source')

    filePath = sys.argv[1]
    errors, sentences = read_simplesrl(filePath)

    #print "Got {0} sentences parsed".format(len(sentences))
    #print ""

    words2index = create_word_index(sentences)

    #print "Created {0} words in index".format(len(words2index))
    #print words2index


    print pickle.dumps(words2index) #, open(destinationPath, 'wb'))