#chatGPT
import tensorflow as tf
#import tensorflow_io as tfio  # Import the tensorflow-io package


# Define the model class as it was originally
class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

# Define the OneStep class
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :] / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        predicted_chars = self.chars_from_ids(predicted_ids)
        return predicted_chars, states

def load_model_and_vocab(gcs_checkpoint_dir):
    vocab = ['\n', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    # Load the latest checkpoint from GCS
    latest_checkpoint = tf.train.latest_checkpoint(gcs_checkpoint_dir)

    # Define the model parameters
    vocab_size = len(ids_from_chars.get_vocabulary())
    embedding_dim = 256
    rnn_units = 1024

    # Create a new instance of the model and load the weights
    model = MyModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
    model.load_weights(latest_checkpoint)

    # Create an instance of OneStep with the loaded model
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    
    return one_step_model

def generate_suggestions(seeds, num_steps=10):
    gcs_checkpoint_dir = 'gs://mlopsfileprojectbucket/one_step'
    one_step_model = load_model_and_vocab(gcs_checkpoint_dir)

    states = None
    next_char = tf.constant(seeds)
    result = [next_char]

    for n in range(num_steps):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    suggestions = [r.numpy().decode('utf-8') for r in result]
    
    return suggestions

res_list = []
#semantically_related_words = semantic_search(seed)
semantically_related_words = ['sweaters', 'jacket', 'hoodies']

suggestions = generate_suggestions(semantically_related_words, num_steps=50)
for i in suggestions: 
    res_list = i.splitlines()
    #print (res_list[0])
print (res_list)
