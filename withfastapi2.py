import tensorflow as tf
import tensorflow_io as tfio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

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

def load_model_and_vocab(model_dir):
    logging.info(f"Loading model from {model_dir}")
    
    vocab = ['\n', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    # Ensure tensorflow-io GCS filesystem is initialized
    #tf.io.gfile.exists('gs://')

    # Load the saved model
    model = tf.keras.models.load_model(model_dir, custom_objects={'MyModel': MyModel})
    logging.info(f"Model loaded successfully from {model_dir}")

    # Create an instance of OneStep with the loaded model
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    
    return one_step_model

def semantic_search(seed):
    logging.info(f"Performing semantic search for seed: {seed}")
    
    empty_lists = []
    return_list = []
    final_list = []
    semantically_similar_categories_path = 'https://storage.googleapis.com/mlopsfileprojectbucket/similar_categories_grouped.json'
    response = requests.get(semantically_similar_categories_path)
    if response.status_code == 200:
        semantically_similar_categories = json.loads(response.text)
        all_keys = semantically_similar_categories.keys()
        for item in all_keys:
            empty_lists.append(semantically_similar_categories[item])
            
    for i in empty_lists:
        if (len(return_list) <= 5) and (seed in i):
            length = len(i)
            if length >= 5:
                random_choice = random.sample(i, 5)
                return_list.extend(random_choice)
            if length < 5:
                return_list.extend(i)
            break

    for i in return_list:
        if "(" in i:
            continue
        else:
            final_list.append(i.lower())
    
    logging.info(f"Semantic search results: {final_list}")
    return final_list

def generate_suggestions(seeds, num_steps=10):
    model_dir = 'gs://mlopsfileprojectbucket/trained_model'
    logging.info(f"Generating suggestions for seeds: {seeds} with num_steps: {num_steps}")
    one_step_model = load_model_and_vocab(model_dir)

    states = None
    next_char = tf.constant(seeds)
    result = [next_char]

    for n in range(num_steps):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    suggestions = [r.numpy().decode('utf-8') for r in result]
    
    logging.info(f"Generated suggestions: {suggestions}")
    return suggestions

# Define request and response models for FastAPI
class InputData(BaseModel):
    seeds: list[str]
    num_steps: int = 10

class OutputData(BaseModel):
    suggestions: list[str]

# Define FastAPI endpoint
@app.post("/generate-suggestions/", response_model=OutputData)
# def generate_suggestions_endpoint(input_data: InputData):
#     try:
#         list_of_words = semantic_search(input_data.seeds[0])
#         suggestions = generate_suggestions(list_of_words[0], num_steps=input_data.num_steps)
#         return {"suggestions": suggestions}
#     except Exception as e:
#         logging.error(f"Error generating suggestions: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

def generate_suggestions_endpoint(input_data: InputData):
    try:
        suggestions = generate_suggestions(input_data.seeds, num_steps=input_data.num_steps)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# def generate_suggestions_endpoint(input_data: InputData):
#     try:
#         list_of_words = semantic_search(input_data.seeds[0])
#         if not list_of_words:
#             raise ValueError("Semantic search returned no results.")
#         suggestions = generate_suggestions(list_of_words, num_steps=input_data.num_steps)
#         return {"suggestions": suggestions}
#     except Exception as e:
#         logging.error(f"Error generating suggestions: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
