import tensorflow as tf
import tensorflow_io as tfio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import random
import logging
import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri("http://34.93.45.146:5000")
mlflow.set_experiment("rnn_autosuggest_3")

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load the model and vocabulary once at startup
model_dir = 'gs://mlopsfileprojectbucket/trained_model'
vocab = ['\n', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
model = tf.keras.models.load_model(model_dir, custom_objects={'MyModel': MyModel})
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
logging.info("Model loaded successfully at startup")

class InputData(BaseModel):
    seeds: list[str]
    num_steps: int = 100

class OutputData(BaseModel):
    suggestions: list[str]

@app.post("/generate-suggestions-3/", response_model=OutputData)
def generate_suggestions_endpoint(input_data: InputData):
    try:
        list_of_words = semantic_search(input_data.seeds[0])
        if not list_of_words:
            raise ValueError("Semantic search returned no results.")
        lower_list = [i.lower() for i in list_of_words]
        suggestions = generate_suggestions(lower_list, input_data.num_steps)
        return {"suggestions": suggestions}
    except Exception as e:
        logging.error(f"Error generating suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def semantic_search(seed):
    logging.info(f"Performing semantic search for seed: {seed}")
    empty_lists = []
    return_list = []
    final_list = []
    semantically_similar_categories_path = 'https://storage.googleapis.com/mlopsfileprojectbucket/similar_categories_grouped_v2.json'
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

def generate_suggestions(seeds, num_steps=100):
    logging.info(f"Generating suggestions for seeds: {seeds} with num_steps: {num_steps}")
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run() as run:
        mlflow.log_param("seeds", seeds)
        mlflow.log_param("num_steps", num_steps)
        states = None
        next_char = tf.constant(seeds)
        result = [next_char]
        for n in range(num_steps):
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)
        result = tf.strings.join(result)
        suggestions = [r.numpy().decode('utf-8') for r in result]
        with open("/tmp/suggestions.txt", "w") as f:
            for suggestion in suggestions:
                f.write("%s\n" % suggestion)
        mlflow.log_artifact("/tmp/suggestions.txt")
        mlflow.log_metric("num_suggestions", len(suggestions))
        logging.info(f"Generated suggestions: {suggestions}")
    return suggestions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
