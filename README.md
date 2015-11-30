# Semantic role labeling
This small project uses an elman network (*Elman, Jeffrey L. "Finding structure in time." Cognitive science 14.2 (1990): 179-211.*) to classify semantic role labels within sentences.
It is essentially based on the [DeepLearning tutorial about spoken language understanding](http://www.deeplearning.net/tutorial/rnnslu.html#rnnslu).



## Create a new model with training data
```
python cli.py --train ../srl-data/wikipedia/jupiter.simplesrl \
	../srl-data/wikipedia/paris.simplesrl \
	../srl-data/wikipedia/obama.simplesrl \
	-o elman/2015-08-Aug-06-2005
```

Persist built index to a file:
```
python cli.py --train data/wikipedia/jupiter.simplesrl \
	data/wikipedia/paris.simplesrl \
	data/wikipedia/obama.simplesrl \
	-o data/2015-08-Aug-06-2005 \
	-D index.db
```

Use specific (additional) index file (for a larger amount of words):
```
python cli.py --train data/wikipedia/jupiter.simplesrl \
	data/wikipedia/paris.simplesrl \
	data/wikipedia/obama.simplesrl \
	-o data/2015-08-Aug-06-2005 \
	-i someIndex.db
```


## Use an existing model to classify sentences
``python cli.py -n data/db/2015-11-Nov-30-1445/ -c 'Jupiter is probably a large planet.'``


## Export an index from an existing model
``python cli.py -n data/db/2015-11-Nov-30-1445/ -D newIndex.db``


## Creating labeled data
We used Wikipedia as source for training sentences.
Sentences are split into components (one or multiple words) and annotated with the desired label.
For that each training sentence has to be defined as its whole.
In the following lines chunks of the sentence follow.
Each line contains the next part of the sentence and the desired label **separated with a tab**.
All components combined have to result in the whole sentence defined in the first line.
The annotation statements are closed if a new line follows.
After such a block a new sentence might follow.
Spaces before and after each chunk have to be removed.
Punctuation has to be kept and results in another word index (``house`` and ``house,`` differ).
```
Jupiter is the fifth planet from the Sun and the largest planet in the Solar System.  
Jupiter SUB
is      PRED
the fifth planet        OBJ
from the Sun    LOC
and     REST
the largest planet      OBJ
in the Solar System.    LOC

It is a giant planet with a mass one-thousandth of that of the Sun, but is two and a half times that of all the other planets in the Solar System combined.
It      SUB
is      PRED
a giant planet  OBJ
with a mass one-thousandth of that of the Sun, but      REST
is      PRED
two and a half times that of all the other planets      REST
in the Solar System     LOC
combined.       PRED
```
It is essential to define a gold-standard how to label different parts of a sentence.
We used *SUB*, *PRED*, *OBJ*, *TIME*, *LOC*, *CON* and *REST* as labels.
You might define your own labels.
Our own classification differed heavily and in a simple manual test we realized that we labeled our own sentences between 60 and 80% accurracy depending on who defined the gold-standard.



## Worth knowing about the network
* indices contain word-index-associations for words and labels
* trained models can not extend their index, as it is built upon their word size
* existing indices can be used for new models
* indices can be exported from existing models
* existing models can be trained with sentences consisting of words, that are already indexed by the models index
* classification can only be performed with words, that exist in the models index
* the cli script shuffles the passed training data and uses a part of it for testing, so it might decay the learning rate
* the number of epochs can be specified, so the model won't be trained too many times with the same sentences
* separated training and test sets are currently not possible (only combined files, which are shuffled and then split)


## CLI examples
```
./cli.py -n db/2015-08-06.srlen -c 'This is some sentence.'
./cli.py -n db/2015-08-06.srlen -c data/someSentences.txt
./cli.py -n db/2015-08-06.srlen -t data/someLabeledSentences.simplesrl
./cli.py -t data/someLabeledSentences.simplesrl
./cli.py -t data/someLabeledSentences.simplesrl -s db/2015-08-06.settings
./cli.py -t data/someLabeledSentences.simplesrl -s db/2015-08-06.settings -o db/2015-08-07.srlen
./cli.py -n db/2015-08-06.srlen -t data/someLabeledSentences.simplesrl -o db/2015-08-07.srlen
./cli.py -n db/2015-08-06.srlen -t data/someLabeledSentences.simplesrl -c 'Please classify this sentence'
./cli.py -n db/2015-08-06.srlen -t data/someLabeledSentences.simplesrl -c data/someSentences.txt
```
