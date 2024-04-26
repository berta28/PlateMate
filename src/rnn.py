import numpy as np
from models.dataset import Dataset
# Packages for training the model and working with the dataset.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json

# Utility/helper packages.
import platform
import time
import pathlib
import os
import urllib.request

def load_dataset(silent=False):
    # List of dataset files we want to merge.
    dataset_file_names = [
        'recipes_with_nutritional_info.json'
    ]
    
    dataset = []
    dataset_file_path = 'C:/Users/smber/code/CS534/PlateMate/data/recipes/recipes_with_nutritional_info.json'


    # Download the JSON file
    url = "https://data.csail.mit.edu/im2recipe/recipes_with_nutritional_info.json"
    p = urllib.request.urlretrieve(url, dataset_file_path)
    
    # Wait until the file is downloaded and saved
    while not os.path.exists(dataset_file_path):
        time.sleep(1)
    
    with open(dataset_file_path) as dataset_file:
        json_data_list = json.load(dataset_file)
        dict_keys = [key for key in json_data_list[0]]
        dict_keys.sort()
        dataset += json_data_list
        # This code block outputs the summary for each dataset.
        if silent == False:
            print(dataset_file_path)
            print('===========================================')
            print('Number of examples: ', len(json_data_list), '\n')
            print('Example object keys:\n', dict_keys, '\n')
            print('Example object:\n', json_data_list[0], '\n')
            print('Required keys:\n')
            print('  title: ', json_data_list[0]['title'], '\n')
            print('  ingredients: ', json_data_list[0]['ingredients'], '\n')
            print('  instructions: ', json_data_list[0]['instructions'])
            print('\n\n')
    return dataset

def recipe_validate_required_fields(recipe):
    required_keys = ['title', 'ingredients', 'instructions']
    
    if not recipe:
        return False
    
    for required_key in required_keys:
        if not recipe[required_key]:
            return False
        
        if type(recipe[required_key]) == list and len(recipe[required_key]) == 0:
            return False
    
    return True
def filter_recipes_by_length(recipe_test):
    return len(recipe_test) <= MAX_RECIPE_LENGTH

def recipe_to_string(recipe):
    # This string is presented as a part of recipes so we need to clean it up.
    # noize_string = 'ADVERTISEMENT'
    
    title = recipe['title']
    ingredients = recipe['ingredients']
    instructions = recipe['instructions']
    
    ingredients_string = ''
    for ingredient in ingredients:
        ingredient = ingredient['text']
        if ingredient:
            ingredients_string += f'• {ingredient}\n'
    
    instructions_string = ''
    for instruction in instructions:
        instruction = instruction['text']
        if instruction:
            instructions_string += f'▪︎ {instruction}\n'
    
    return f'{STOP_WORD_TITLE}{title}\n{STOP_WORD_INGREDIENTS}{ingredients_string}{STOP_WORD_INSTRUCTIONS}{instructions_string}'

def recipe_sequence_to_string(recipe_sequence):
    recipe_stringified = tokenizer.sequences_to_texts([recipe_sequence])[0]
    print(recipe_stringified)

def split_input_target(recipe):
    input_text = recipe[:-1]
    target_text = recipe[1:]
    
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        batch_input_shape=[batch_size, None]
    ))
    model.add(tf.keras.layers.LSTM(
        units=rnn_units,
        return_sequences=True,
        stateful=True,
        recurrent_initializer=tf.keras.initializers.GlorotNormal()
    ))
    model.add(tf.keras.layers.Dense(vocab_size))
    
    return model

# An objective function.
# The function is any callable with the signature scalar_loss = fn(y_true, y_pred).
def loss(labels, logits):
    entropy = tf.keras.losses.sparse_categorical_crossentropy(
      y_true=labels,
      y_pred=logits,
      from_logits=True
    )
    
    return entropy

def generate_text(model, start_string, num_generate = 1000, temperature=1.0):
    # Evaluation step (generating text using the learned model)
    
    padded_start_string = STOP_WORD_TITLE + start_string
    # Converting our start string to numbers (vectorizing).
    input_indices = np.array(tokenizer.texts_to_sequences([padded_start_string]))
    # Empty string to store our results.
    text_generated = []
    # Here batch size == 1.
    model.reset_states()
    for char_index in range(num_generate):
        predictions = model(input_indices)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        # Using a categorical distribution to predict the character returned by the model.
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions,
            num_samples=1
        )[-1, 0].numpy()
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state.
        input_indices = tf.expand_dims([predicted_id], 0)
        
        next_character = tokenizer.sequences_to_texts(input_indices.numpy())[0]
        text_generated.append(next_character)
    return (padded_start_string + ''.join(text_generated))

def generate_combinations(model):
    recipe_length = 1000
    try_letters = ['', '\n', 'A', 'B', 'C', 'O', 'L', 'Mushroom', 'Apple', 'Slow', 'Christmass', 'The', 'Banana', 'Homemade']
    try_temperature = [1.0, 0.8, 0.4, 0.2]
    for letter in try_letters:
        for temperature in try_temperature:
            generated_text = generate_text(
                model,
                start_string=letter,
                num_generate = recipe_length,
                temperature=temperature
            )
            print(f'Attempt: "{letter}" + {temperature}')
            print('-----------------------------------')
            print(generated_text)
            print('\n\n')
     

















STOP_WORD_TITLE = 'titlename' 
STOP_WORD_INGREDIENTS = 'ingname'
STOP_WORD_INSTRUCTIONS = 'insts'




dataset_raw = load_dataset()

dataset_validated = [recipe for recipe in dataset_raw if recipe_validate_required_fields(recipe)]
print('Dataset size BEFORE validation', len(dataset_raw))
print('Dataset size AFTER validation', len(dataset_validated))
print('Number of incomplete recipes', len(dataset_raw) - len(dataset_validated))

dataset_stringified = [recipe_to_string(recipe) for recipe in dataset_validated]
print('Stringified dataset size: ', len(dataset_stringified))

for recipe_index, recipe_string in enumerate(dataset_stringified[:3]):
    print('Recipe #{}\n---------'.format(recipe_index + 1))
    print(recipe_string)
    print('\n')

print(dataset_stringified[50000])


MAX_RECIPE_LENGTH = 2000

dataset_filtered = [recipe_text for recipe_text in dataset_stringified if filter_recipes_by_length(recipe_text)]
print('Dataset size BEFORE filtering: ', len(dataset_stringified))
print('Dataset size AFTER filtering: ', len(dataset_filtered))
print('Number of eliminated recipes: ', len(dataset_stringified) - len(dataset_filtered))


TOTAL_RECIPES_NUM = len(dataset_filtered)
print('MAX_RECIPE_LENGTH: ', MAX_RECIPE_LENGTH)
print('TOTAL_RECIPES_NUM: ', TOTAL_RECIPES_NUM)

STOP_SIGN = '␣'
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    char_level=True,
    filters='',
    lower=False,
    split=''
)
# Stop word is not a part of recipes, but tokenizer must know about it as well.
tokenizer.fit_on_texts([STOP_SIGN])
tokenizer.fit_on_texts(dataset_filtered)
tokenizer.get_config()

VOCABULARY_SIZE = len(tokenizer.word_counts) + 1
array_vocabulary = tokenizer.sequences_to_texts([[word_index] for word_index in range(VOCABULARY_SIZE)])
print('vocabulary size: ', VOCABULARY_SIZE)
dataset_vectorized = tokenizer.texts_to_sequences(dataset_filtered)
print('Vectorized dataset size', len(dataset_vectorized))


recipe_sequence_to_string(dataset_vectorized[0])


for recipe_index, recipe in enumerate(dataset_vectorized[:10]):
    print('Recipe #{} length: {}'.format(recipe_index + 1, len(recipe)))

dataset_vectorized_padded_without_stops = tf.keras.preprocessing.sequence.pad_sequences(
    dataset_vectorized,
    padding='post',
    truncating='post',
   
    maxlen=MAX_RECIPE_LENGTH-1,
    value=tokenizer.texts_to_sequences([STOP_SIGN])[0]
)
dataset_vectorized_padded = tf.keras.preprocessing.sequence.pad_sequences(
    dataset_vectorized_padded_without_stops,
    padding='post',
    truncating='post',
    maxlen=MAX_RECIPE_LENGTH+1,
    value=tokenizer.texts_to_sequences([STOP_SIGN])[0]
)
for recipe_index, recipe in enumerate(dataset_vectorized_padded[:10]):
    print('Recipe #{} length: {}'.format(recipe_index, len(recipe)))


recipe_sequence_to_string(dataset_vectorized_padded[0])

dataset = tf.data.Dataset.from_tensor_slices(dataset_vectorized_padded)
print(dataset)

dataset_targeted = dataset.map(split_input_target)
print(dataset_targeted)

for input_example, target_example in dataset_targeted.take(1):
    print('Input sequence size:', repr(len(input_example.numpy())))
    print('Target sequence size:', repr(len(target_example.numpy())))
    
    
    input_stringified = tokenizer.sequences_to_texts([input_example.numpy()[:50]])[0]
    target_stringified = tokenizer.sequences_to_texts([target_example.numpy()[:50]])[0]

BATCH_SIZE = 32

SHUFFLE_BUFFER_SIZE = 1000
dataset_train = dataset_targeted \
.shuffle(SHUFFLE_BUFFER_SIZE) \
.batch(BATCH_SIZE, drop_remainder=True) \
.repeat()
print(dataset_train)

for input_text, target_text in dataset_train.take(1):
    print('1st batch: input_text:', input_text)
    print()
    print('1st batch: target_text:', target_text)

model = build_model(
  vocab_size=VOCABULARY_SIZE,
  embedding_dim=256,
  rnn_units=512,
  batch_size=BATCH_SIZE
)
model.summary()


for input_example_batch, target_example_batch in dataset_train.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    tmp_logits = [
  [-0.95, 0, 0.95],
]

tmp_samples = tf.random.categorical(
    logits=tmp_logits,
    num_samples=5
)
print(tmp_samples)
     
sampled_indices = tf.random.categorical(
    logits=example_batch_predictions[0],
    num_samples=1
)
sampled_indices = tf.squeeze(
    input=sampled_indices,
    axis=-1
).numpy()
sampled_indices.shape



model.compile( optimizer=tf.keras.optimizers.RMSprop(),loss=loss)

EPOCHS = 15
INITIAL_EPOCH = 1
STEPS_PER_EPOCH = 250


adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=adam_optimizer,
    loss=loss
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=5,
    monitor='loss',
    restore_best_weights=True,
    verbose=1
)


chkpnt_dir = "/c:/Users/smber/code/CS534/PlateMate/data/models"
os.makedirs(chkpnt_dir, exist_ok=True)
chkpnt_pref = os.path.join(chkpnt_dir, 'ckpt_{epoch}')
chkpnt_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=chkpnt_pref,
    save_best_only=False, save_weights_only=False, mode='auto', save_freq=1)

history = model.fit(
    x=dataset_train,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    initial_epoch=INITIAL_EPOCH,
    callbacks=[
        early_stopping_callback
    ]
)
# Saving the trained model to file (to be able to re-use it later).
os.listdir(chkpnt_dir)
model_name = 'recipe_maker.h5'
model.save(model_name, save_format='h5')


simplified_batch_size = 1
checkpoint_dir = "/content/drive/MyDrive/recipe_maker.h5"
model_simplified = build_model(
  vocab_size=VOCABULARY_SIZE,
  embedding_dim=256,
  rnn_units=512,
  batch_size=simplified_batch_size)
model_simplified.load_weights(checkpoint_dir)
model_simplified.build(tf.TensorShape([simplified_batch_size, None]))
model_simplified.summary()
     