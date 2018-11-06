
# Phrasal Chunking

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
	
## Running option for jupyter notebook

We documented our path of developing the approach in the jupyter notebook `chunk.ipynb`. In that notebook, we commented out all the code for developing the approach, but not used for final submission. If the TA wants to check if those code actually work, please kindly uncomment the code you want to run. The output should be exactly the same as the output left in the jupyter notebook. In addition, training the model for final submission takes more than one hour on my machine (i7 7700k). To save more time for TA, we pushed the model file `baseline_bigram_avg.model` onto the repo. Running the whole jupyter notebook will NOT re-train the model. Instead it will read the `baseline_bigram_avg.model` and make predictions with it. If you want to retrain the model with our implementation, please kindly uncomment all the commented code in the third last cell. Besides, we put the implementation of `perc_avg_train` in `chunk.py`. So you can score the hidden test set with the command line option.

## Training phase

    python3 chunk.py > baseline_bigram_avg.model

## Testing and Evaluation phase

    python3 perc.py -m baseline_bigram_avg.model > output
    python3 score_chunks.py < output

OR

    python3 perc.py -m baseline_bigram_avg.model | python3 score_chunks.py

## Options

    python3 chunk.py -h

This shows the different options you can use in your training
algorithm implementation.  In particular the -n option will let you
run your algorithm for less or more iterations to let your code run
faster with less accuracy or slower with more accuracy. You must
implement the -n option in your code so that we are able to run
your code with different number of iterations.

## Final submission

Our final submission achieved a score of 93.51 on the devset.

    $ python3 perc.py -m baseline_bigram_avg.model | python3 score_chunks.py 
    reading data ... 
    done.
    processed 500 sentences with 10375 tokens and 5783 phrases; found phrases: 5801; correct phrases: 5416
				 ADJP: precision:  71.00%; recall:  71.72%; F1:  71.36; found:    100; correct:     99
				 ADVP: precision:  77.03%; recall:  79.70%; F1:  78.35; found:    209; correct:    202
				CONJP: precision: 100.00%; recall:  60.00%; F1:  75.00; found:      3; correct:      5
				 INTJ: precision:   0.00%; recall:   0.00%; F1:   0.00; found:      0; correct:      1
				   NP: precision:  94.42%; recall:  94.42%; F1:  94.42; found:   3026; correct:   3026
				   PP: precision:  96.77%; recall:  98.03%; F1:  97.40; found:   1237; correct:   1221
				  PRT: precision:  80.00%; recall:  72.73%; F1:  76.19; found:     20; correct:     22
				 SBAR: precision:  84.47%; recall:  81.31%; F1:  82.86; found:    103; correct:    107
				   VP: precision:  92.84%; recall:  93.09%; F1:  92.96; found:   1103; correct:   1100
	accuracy:  95.52%; precision:  93.36%; recall:  93.65%; F1:  93.51
	Score: 93.51