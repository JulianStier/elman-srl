import sys

class SimpleSrl:
    def __init__(self, filePath):
        self.filePath = filePath
        self.ignoreErrors = False
        self.reset()

    def reset(self):
        # Reset object properties
        # Errors contains a list of error messages (usually containing info about line and type of error)
        self.errors = []
        # Structure is a list of lists which contains the sentence as first element and the parts with labels as ongoing lists:
        # [
        #  [0]: [
        #    [0]: This is a whole sentence,
        #    [1]: [ This, SUB]
        #    [2]: [ is, PRED]
        #    [3]: [ a whole sentence, OBJ]
        #   ]
        # ]
        self.structure = []
        # Labels contains a list of all used labels
        self.labels = []
        # Sentences contains the list of all occurring VALID and full sentences (only sentences with valid labels)
        self.sentences = []
        # Words contains a list of all occurring words over all sentences
        self.words = []
        # Labeled data contains two lists with the same size. In the first list there is a splitted sentence and in the
        # second list follow all its associated labels (for each word).
        # [
        #  [0]: [
        #   ['This', 'is', 'a', 'whole', 'sentence'],
        #   ['There', 'is', 'another', 'important', 'sentence']
        #  ],
        #  [1]: [
        #   ['SUB', 'PRED', 'OBJ', 'OBJ', 'OBJ'],
        #   ['REST', 'PRED', 'OBJ', 'OBJ', 'OBJ']
        #  ]
        # ]
        self.labeledData = [ [], [] ]

    def getStructure(self):
        return self.structure

    def getSentences(self):
        return self.sentences

    def getWords(self):
        return self.words

    def getLabels(self):
        return self.labels

    def getLabeledData(self):
        return self.labeledData

    def setIgnoreErrors(self, flag):
        self.ignoreErrors = flag

    def getErrors(self):
        return self.errors

    def read(self):
        '''
         Reads a text file containing a simple semantic role labelling format.
         The format is defined as following:
         Each sentence is preceeded with an empty line.
         The first line contains the full sentence without any breaks (including all special characters, ..).
         After the first line break at least one subset of the sentence has to follow.
         The subset might be the full sentence or a part of it.
         All subsets concatenated have to form the full sentence.
         Each subset of the sentence has to be placed in-order in an own line.
         After each subset of the sentence a tab follows denoting the type of the role (e.g. 'SUB' for subject or 'PRED' for predicate).
         A sentence and all its ordered subsets with labels is closed by a new line.
        '''

        self.reset()

        lineCount = 0

        # The status denotes whether
        #  0 - a new sentence has to follow
        #  1 - a sentence was just given and a subset has to follow
        #  2 - one or multiple subsets were given and another subset or a newline have to follow
        #  3 - error, waiting for next new line to keep going
        status = 0
        with open(self.filePath) as f:
            currentStructure = []
            for line in f:
                # Increase line count (initially 1)
                lineCount += 1
                #print "Working through line {0}".format(lineCount)

                # Strip line from whitespaces and breaks at start and end
                line = line.strip(' \n')
                stripped = line.strip()

                # Stop if we previously got some error
                if status is 3:
                    if len(stripped) is 0:
                        # Ok, we're back in the game
                        #print "Resuming in line {0} after error".format(lineCount)
                        status = 0
                    continue

                if status is 0:
                    if len(stripped) is 0:
                        # Arbitrary number of empty lines allowed
                        continue
                    else:
                        # We've got a new line
                        currentStructure = [line]
                        status = 1
                else:
                    if len(stripped) is 0:
                        # We need at status 1 at least one subset with its label
                        if status is 1:
                            status = 3
                            msg = "In line {0}: Expected at least one labelled sentence-part"
                            self.errors = handle_error(self.errors, msg.format(lineCount))
                            if not self.ignoreErrors:
                                return (self.errors, [])
                            continue

                        # With an empty line, a sentence and its assignments get closed
                        if status is 2:
                            check = check_assignment(currentStructure)
                            if check != True:
                                status = 3
                                msg = "In line {0}: "+check
                                self.errors = handle_error(self.errors, msg.format(lineCount))
                                if not self.ignoreErrors:
                                    return (self.errors, [])
                                continue

                            # Append current structure to list of structures
                            self.structure.append(currentStructure)

                            # Append whole sentence to list of valid sentences
                            self.sentences.append(currentStructure[0])

                            words = currentStructure[0].split()
                            self.labeledData[0].append(words)
                            labels = []
                            for labelData in currentStructure[1:]:
                                sentenceSubset = labelData[0]
                                role = labelData[1]
                                subsetWords = sentenceSubset.split()
                                labels.extend([role for w in subsetWords])
                            if len(labels) is not len(words):
                                msg = "In line {0}: number of labels does not match number of words. Error in code. NoL: {1}, NoW: {2}"
                                self.errors = handle_error(self.errors, msg.format(lineCount, len(labels), len(words)))
                                return (self.errors, [])
                            self.labeledData[1].append(labels)

                            status = 0
                    else:
                        parts = stripped.split('\t')
                        if parts is None or len(parts) is not 2:
                            status = 3
                            self.errors = handle_error(self.errors, "In line {0}: assignment splits wrong. Split by tab into {1} parts.".format(lineCount, len(parts)))
                            if not self.ignoreErrors:
                                return (self.errors, [])
                            continue
                        else:
                            subset = parts[0].strip()
                            role = parts[1].strip()

                            # Add [subset, role] to structure
                            currentStructure.append([subset, role])

                            # Add role as label to list of labels
                            if not role in self.labels:
                                self.labels.append(role)

                            # Add all new words to list of words
                            words = subset.split()
                            for w in words:
                                if not w in self.words:
                                    self.words.append(w)

                            if status is 1:
                                status = 2

        # Return errors and resulting dictionary
        return (self.errors, self.sentences)


"""
 Reads a text file containing a simple semantic role labelling format.
 The format is defined as following:
 Each sentence is preceeded with an empty line.
 The first line contains the full sentence without any breaks (including all special characters, ..).
 After the first line break at least one subset of the sentence has to follow.
 The subset might be the full sentence or a part of it.
 All subsets concatenated have to form the full sentence.
 Each subset of the sentence has to be placed in-order in an own line.
 After each subset of the sentence a tab follows denoting the type of the role (e.g. 'SUB' for subject or 'PRED' for predicate).
 A sentence and all its ordered subsets with labels is closed by a new line.
"""
def read_simplesrl(filePath, ignoreErrors = True):
    errors = []
    sentences = []
    lineCount = 0

    # The status denotes whether
    #  0 - a new sentence has to follow
    #  1 - a sentence was just given and a subset has to follow
    #  2 - one or multiple subsets were given and another subset or a newline have to follow
    #  3 - error, waiting for next new line to keep going
    status = 0
    with open(filePath) as f:
        current = []
        for line in f:
            # Increase line count (initially 1)
            lineCount += 1
            #print "Working through line {0}".format(lineCount)

            # Strip line from whitespaces and breaks at start and end
            line = line.strip(' \n')
            stripped = line.strip()

            # Stop if we previously got some error
            if status is 3:
                if len(stripped) is 0:
                    # Ok, we're back in the game
                    #print "Resuming in line {0} after error".format(lineCount)
                    status = 0
                continue

            if status is 0:
                if len(stripped) is 0:
                    # Arbitrary number of empty lines allowed
                    continue
                else:
                    # We've got a new line
                    current = [line]
                    status = 1
            else:
                if len(stripped) is 0:
                    # We need at status 1 at least one subset with its label
                    if status is 1:
                        status = 3
                        errors = handle_error(errors, "Expected at least one labelled sentence-part")
                        if not ignoreErrors:
                            return errors
                        continue

                    # With an empty line, a sentence and its assignments get closed
                    if status is 2:
                        check = check_assignment(current)
                        if check != True:
                            status = 3
                            msg = "In line {0}: "+check
                            errors = handle_error(errors, msg.format(lineCount))
                            if not ignoreErrors:
                                return errors
                            continue
                        sentences.append(current)
                        status = 0

                else:
                    parts = stripped.split('\t')
                    if parts is None or len(parts) is not 2:
                        status = 3
                        errors = handle_error(errors, "In line {0}: assignment splits wrong. Split by tab into {1} parts.".format(lineCount, len(parts)))
                        if not ignoreErrors:
                            return errors
                        continue
                    else:
                        subset = parts[0].strip()
                        role = parts[1].strip()
                        current.append([subset, role])
                        if status is 1:
                            status = 2

    # Return errors and resulting dictionary
    return (errors, sentences)

def check_assignment(parts):
    sentence = parts[0] # Original sentence
    builtSentence = parts[1][0] # Append first subset
    if len(parts) > 2:
        builtSentence += " "
    for subset, role in parts[2:-1]: # Append all following subsets with whitespace except last one
        builtSentence += subset+" "
    if len(parts) > 2: # Append last subset if there is one
        builtSentence += parts[-1][0]

    if sentence != builtSentence:
        return "Subsets and original sentence do not match:\n'"+sentence+"'\n'"+builtSentence+"'"
    else:
        return True


def handle_error(errors, error):
    if errors is None:
        errors = []
    errors.append(error)
    return errors




if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Error('Pass file path as argument')

    filePath = sys.argv[1]
    errors, sentences = read_simplesrl(filePath)

    print "Got {0} sentences parsed".format(len(sentences))
    print ""

    for sentence in sentences:
        print sentence[0]
        for subset, label in sentence[1:]:
            print "\t[" + label + "]\t" + subset
        print ""

    print ""

    if len(errors) is not 0:
        print "Amount errors: {0}".format(len(errors))
        if len(sys.argv) > 2:
            for error in errors:
                print error+"\n"
        else:
            print "Pass any third parameter to display errors."


    #print sentences
    #for s in sentences:
    #    print s[0]