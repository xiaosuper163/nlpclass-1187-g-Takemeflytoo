"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

import perc
import sys, optparse, os
from collections import defaultdict

def perc_avg_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    avg_feat_vec = defaultdict(float)
    default_tag = tagset[0]

    for epoch in range(numepochs):
        count_mistake = 0
        print(f"Running on epoch {epoch+1}......")
        tic = time.time()
        for _, (labeled_list, feat_list) in enumerate(train_data):
            pred_output = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)
            true_output = [x.split()[2] for x in labeled_list]

            if pred_output != true_output:
                count_mistake += 1
                feat_index = 0
                
                for w_index in range(len(pred_output)):
                    pred_tag = pred_output[w_index]
                    true_tag = true_output[w_index]
                    (feat_index, feats) = perc.feats_for_word(feat_index, feat_list)
                    for feat in feats:
                        if feat == 'B' and w_index > 0:
                            if true_output[w_index-1] != pred_output[w_index-1] or pred_tag != true_tag:
                                feat_vec['B:' + true_output[w_index-1], true_tag] += 1
                                feat_vec['B:' + pred_output[w_index-1], pred_tag] -= 1
                        elif pred_tag != true_tag:
                            feat_vec[feat, true_tag] += 1
                            feat_vec[feat, pred_tag] -= 1


            for key in feat_vec.keys():
                # γ = σ/(mT)
                avg_feat_vec[key] += feat_vec[key]

        toc = time.time()
        print(f'Epoch {epoch+1} finished. Time cost on this epoch: {toc-tic}. Number of mistakes: {count_mistake}.')

    for key in avg_feat_vec.keys():
        avg_feat_vec[key] /= (numepochs * len(train_data))
    return avg_feat_vec

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print("reading data ...", file=sys.stderr)
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile, verbose=False)
    print("done.", file=sys.stderr)
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)

