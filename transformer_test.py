# Using Tensorflow
from dotenv import load_dotenv
load_dotenv()

# https://huggingface.co/learn/nlp-course/chapter2/2?fw=tf
# Behind the pipeline

# V1 Test Transformer with Sentiment analysis ---------------------------------

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

print(classifier.model.config.id2label)  # Got the labels, but I CANT FIND THE PROBABILITIES!!!

# Can't see the output
# V2 Step 1 of 2 -------------------------------------------------------------------
print("Create Input")
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
print(inputs)
pc_inputs = inputs.data['input_ids']
pc_attention = inputs.data['attention_mask']

# V2 Step 2a of 2 -------------------------------------------------------------------
print("Use Input to Create Output")
from transformers import TFAutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModel.from_pretrained(checkpoint)

outputs = model(inputs)
print(outputs.last_hidden_state.shape)
#print(outputs.logits.shape)

# V2 Step 2b of 2 -------------------------------------------------------------------
print("Use Input to create new Output")
from transformers import TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(inputs)

print(outputs.logits.shape)
import tensorflow as tf

predictions = tf.math.softmax(outputs.logits, axis=-1)
print(predictions)

print(model.config.id2label)
# -----------------------------------------------------------------------------

# https://huggingface.co/learn/nlp-course/chapter2/3?fw=tf
# V1 Models --------------------------------------------------------
from transformers import BertConfig, TFBertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = TFBertModel(config)
print(config)


# V2 Models --------------------------------------------------------
from transformers import TFBertModel

model = TFBertModel.from_pretrained("bert-base-cased")

# Saving a Model
model.save_pretrained("directory_on_my_computer")

# -------------------------------------------------------------------
# https://huggingface.co/learn/nlp-course/chapter2/4?fw=tf
# Tokenisers

# V1 ------- works fine -------------------------------------------------
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")
output = model(**tokens)

tokenizer.save_pretrained("directory_on_my_computer")

# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)

id2 = tokens.data['input_ids']
id3 = id2.numpy()
id5 = id3[0].tolist()
#id4 = tf.get_static_value(id2)

decoded_string = tokenizer.decode(id5)

print(decoded_string)

# V2 --- works fine -------------------------------------------------------
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)

# -------------------------------------------------------------------
# https://huggingface.co/learn/nlp-course/chapter2/5?fw=tf
# Handling multiple sequences

# V1 ---------------------------------------------------------------
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
#sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = tf.constant([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

# -------------------------------------------------------------------
# https://huggingface.co/learn/nlp-course/chapter2/6?fw=tf
# Putting it all together

# V1 ---------------------------------------------------------------
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")

output = model(**model_inputs)

print("Fin")
