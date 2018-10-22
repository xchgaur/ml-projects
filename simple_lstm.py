import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import collections
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import random
import pickle


start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

    
training_file = 'abstract.txt'
training_data = read_data(training_file)
#print(training_data)
print("Length of training data is {}".format(len(training_data)))

dictionary, reverse_dictionary = build_dataset(training_data)

#pickle the dicts
with open('simple_lstm_dictionary.pickle','wb') as f:
    pickle.dump(dictionary,f)
with open('simple_lstm_reverse_dictionary.pickle','wb') as f:
    pickle.dump(reverse_dictionary,f)

print("number of keys in dict are : {}".format(len(dictionary.keys())))
print(dictionary)


# Parameters
learning_rate = 0.001
training_iters = 30000
display_step = 1000
n_input = 5

# number of units in RNN cell
n_hidden = 512


#initial variables
tf.reset_default_graph()

vocab_size = len(dictionary)
# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])


def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

saver = tf.train.Saver()
pred = RNN(x, weights, biases)


# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    print(offset, end_offset)
    writer.add_graph(session.graph)

    while step < training_iters:
        #print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)
        #print(training_data)
        #print(offset)

        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        #print(symbols_in_keys)
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        #print(symbols_in_keys)
        
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        #print(symbols_out_onehot)
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
        #print(symbols_out_onehot)

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            epoch = int(step/display_step)
            print(epoch)
            ckpt_path = './models/test-model.ckpt'
            saver.save(session, ckpt_path, global_step=epoch)
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        saver.save(session, "./models/test-model-final.ckpt")
        step += 1
        offset += (n_input+1)

    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")

    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            print(symbols_in_keys)
            for i in range(5):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                print(keys)
                onehot_pred = session.run(pred, feed_dict={x: keys})
                print(onehot_pred)
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                print(onehot_pred_index)
                print(reverse_dictionary[onehot_pred_index])
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")
"""
#below lines will show teh freq distribution of words

with open("abstract.txt", "r") as a:
    lines = a.readlines()
    lexicon = []
    for l in lines:
        all_words = word_tokenize(l.lower())
        #print(lines)
        lexicon.extend(all_words)
    print(len(lexicon))
    w_counts = collections.Counter(lexicon)
    print(w_counts) 
    print(len(w_counts.keys()))
"""
