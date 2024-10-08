import csv
import emoji
import numpy as np
import tensorflow as tf

### UTILS ###
def read_csv(filename):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        for w in sorted(words):
            words_to_index[w] = i
            i = i + 1

    return words_to_index, word_to_vec_map

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

### UTILS ###

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices
    corresponding to words in the sentences.

    Arguments:
    X -- array of sentences(strings), shape (m,)
    word_to_index -- a dictionary containing the corespondence between a word and idx
    max_len -- maximum number of words in a sentence.

    Returns:
    X_indices -- same shape as X, but every word is now translated to it's idx
    """

    # Number of trainig samples
    m = X.shape[0]

    # Set initial data (with zeros for padding).
    X_indices = np.zeros((m, max_len))

    for i in range(m):
        sentence_words = X[i].lower().split()

        j = 0
        for word in sentence_words:
            X_indices[i, j] = word_to_index[word]
            j = j + 1

    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_size = len(word_to_index) + 1              # adding 1 to fit Keras embedding (requirement)
    any_word = next(iter(word_to_vec_map.keys()))
    emb_dim = word_to_vec_map[any_word].shape[0]    # define dimensionality of your GloVe word vectors (= 50)
      
    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros((vocab_size, emb_dim))
    
    # Step 2
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Step 3
    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = tf.keras.layers.Embedding(vocab_size, emb_dim)

    # Step 4 (already done for you; please do not modify)
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def emojify(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Create input layer.
    input = tf.keras.layers.Input(input_shape, dtype='int32')

    # Create embedded layer.
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(input)

    # Apply many-to-one architecture.
    X = tf.keras.layers.LSTM(units=128, return_sequences=True)(embeddings)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.LSTM(units=128)(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(units=5)(X)
    X = tf.keras.layers.Activation("softmax")(X)

    # Create model.
    model = tf.keras.Model(inputs=input, outputs=X)

    return model

# Emoji used
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)])

def main():
    X_train, Y_train = read_csv('data/emojify_data.csv')
    X_test, Y_test = read_csv('data/tesss.csv')

    """
    word_to_index -- the index of the word in the sorted words array
    index_to_word -- the coresponding word of the index in the sorted array
    word_to_vec_map -- mapping between a word and it's vector representation
    """
    word_to_index, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

    # Find maximum len of a sentence in the input.
    max_len = len(max(X_train, key=lambda x: len(x.split())).split())
    
    model = emojify((max_len,), word_to_vec_map, word_to_index)
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train_indices = sentences_to_indices(X_train, word_to_index, max_len)
    Y_train_oh = convert_to_one_hot(Y_train, C = 5)
    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len)
    Y_test_oh = convert_to_one_hot(Y_test, C = 5)

    # Train the model and use it on validation test set
    model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True, validation_data=[X_test_indices, Y_test_oh])

    # Visualize what emoji is model generating.
    while (True):
        sentence = input("Type your message:\n")
        sentence = np.array([sentence])

        sentence_indices = sentences_to_indices(sentence, word_to_index, max_len)
        print(sentence[0] + ' ' + label_to_emoji(np.argmax(model.predict(sentence_indices))))

if __name__ == "__main__":
    main()
