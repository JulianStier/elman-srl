import random

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling
    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def context_window(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out


if __name__ == "__main__":
    sen = ['Barack', 'Hussein', 'Obama', 'II', 'is', 'the', '44th', 'and', 'current', 'President', 'of', 'the', 'United', 'States,', 'and', 'the', 'first', 'African', 'American', 'to', 'hold', 'the', 'office.']
    #nums = range(10, 10+len(sen))
    nums = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 15, 21, 22, 17, 15, 23, 24, 25, 26, 27, 17, 28]
    print(len(sen))
    print(len(nums))
    conWin = context_window(nums, 7)
    for row in conWin:
        print(row)

    batch = minibatch(conWin, 9)
    for row in batch:
        print(row)

