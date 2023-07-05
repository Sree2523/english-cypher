# import json
# import tensorflow as tf
# import keras
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Input, LSTM, Dense
# import numpy as np
# from keras.preprocessing import sequence

import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence

import numpy as np

# Load the dataset
with open('traindoc.json', 'r') as file:
    dataset = json.load(file)

# Define the context mapping
context_mapping = {
    "Patient": ["Persons", "Individual", "candidates"],
    "Procedure": ["test", "treatment", "examination", "screening", "analysis", "method", "service", "evaluation", "diagnosis", "diagnostic service"],
    "Hospitals": ["Medical Centers", "Clinics", "Health Facilities", "Health care organization", "Hospital service"],
    "Doctor": ["Physician", "Practitioner", "Healthcare Professional", "attending doctor", "referring doctor", "consulting doctor", "admitting doctor", "primary care provider", "responsible observer", "assistant result interpreter", "principle result interpreter", "surgeon", "anesthesiologist", "order provider", "diagnosing clinician", "technician"],
    "Location": ["place", "address", "residence", "venue", "postcode", "street address"],
    "PI": ["personal identification"],
    "Sexual Orientation": ["gender", "Biological Sex", "gender identity"],
    "NH": ["national health identification number"],
    "Health Care Activity": ["Medical Procedure", "Healthcare service", "Healthcare practice", "clinical activity", "healthcare treatment", "Healthcare procedure"],
    "Allergies": ["reaction", "condition", "disorder", "disease", "sensitivity", "symptom", "sign"],
    "Point of care": ["room", "bed", "facility", "ward", "unit", "chamber"],
    "Disease": ["syndrome", "active problem", "condition", "issue", "disorder"],
    "state": ["Province", "Territory", "Region", "Area"],
    "Finding": ["finding", "result"],
    "Order": ["order"],
    "phonenumberhome": ["Phone number", "phonenumberhome"],
    "fillerrefference": ["filler reference", "FillerOrderNumber"],
    "placerreference": ["placer reference", "PlacerOrderNumber"],
}

# Prepare the input and output data
input_sentences = []
target_queries = []

for item in dataset:
    prompt = item['prompt']
    completion = item['completion']
    for context_key, context_values in context_mapping.items():
        for context_value in context_values:
            prompt = prompt.replace(context_value, context_key + " " + context_value)
            completion = completion.replace(context_value, context_key + " " + context_value)
    input_sentences.append(prompt)
    target_queries.append(completion)

# Create vocabulary mappings
input_vocab = set()
output_vocab = set()
for input_sentence, target_query in zip(input_sentences, target_queries):
    input_vocab.update(input_sentence.split())
    output_vocab.update(target_query.split())
input_vocab = sorted(input_vocab)
output_vocab = sorted(output_vocab)
num_input_tokens = len(input_vocab)
num_output_tokens = len(output_vocab)

# Convert output_vocab to a set
output_vocab = set(output_vocab)

# Add the <end> token to the output_vocab and update num_output_tokens
output_vocab.add('<end>')
num_output_tokens += 1

# Create token index mappings
input_token_index = dict([(token, i) for i, token in enumerate(input_vocab)])
output_token_index = dict([(token, i) for i, token in enumerate(output_vocab)])

# Define model parameters
latent_dim = 256
batch_size = 64
epochs = 25

# Prepare the training data
encoder_input_data = []
decoder_input_data = []
decoder_target_data = []
for input_sentence, target_query in zip(input_sentences, target_queries):
    encoder_input = [input_token_index[token] for token in input_sentence.split()]
    decoder_input = [output_token_index[token] for token in target_query.split()]
    decoder_target = decoder_input[1:] + [output_token_index['<end>']]
    encoder_input_data.append(encoder_input)
    decoder_input_data.append(decoder_input)
    decoder_target_data.append(decoder_target)

encoder_input_data = sequence.pad_sequences(encoder_input_data, padding='post')
decoder_input_data = sequence.pad_sequences(decoder_input_data, padding='post')
decoder_target_data = sequence.pad_sequences(decoder_target_data, padding='post')

# Reshape the input data
max_encoder_seq_length = max(len(seq) for seq in encoder_input_data)
max_decoder_seq_length = max(len(seq) for seq in decoder_input_data)

encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_encoder_seq_length, dtype='int32', padding='post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_decoder_seq_length, dtype='int32', padding='post')

encoder_input_data = np.reshape(encoder_input_data, (encoder_input_data.shape[0], max_encoder_seq_length, 1))
decoder_input_data = np.reshape(decoder_input_data, (decoder_input_data.shape[0], max_decoder_seq_length, 1))

# Define the model architecture
encoder_inputs = Input(shape=(None, 1))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

decoder_inputs = Input(shape=(None, 1))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
decoder_dense = Dense(num_output_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])


# Compile and train the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Save the trained model
model.save('trained_model.h5')
