import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import sklearn.ensemble
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt


credit_record = pd.read_csv('../data/credit_record.csv')
application_record = pd.read_csv('../data/application_record.csv')

print(f"Credit record dataset shape: {credit_record.shape}")
print(f"Application record dataset shape: {application_record.shape}")

columns = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
for objColumn in columns:
    label = LabelEncoder()
    application_record[objColumn] = label.fit_transform(application_record[objColumn].values)

# Confirm that binary categorical variables have been converted to numerical 
print(application_record.head())

# occupation_dict = {'Security staff':0, 'Sales staff':1,
#                   'Accountants':2, 'Laborers':3,
#                   'Managers':4,'Drivers':5,
#                   'Core staff':6, 'High skill tech staff':7,
#                   'Cleaning staff':8,'Private service staff':9,
#                   'Cooking staff':10, 'Low-skill Laborers':11,
#                   'Medicine staff':12, 'Secretaries':13,
#                   'Waiters/barmen staff':14, 'HR staff':15,
#                   'Realty agents':16, "IT staff":17}

# application_record['OCCUPATION_TYPE'] = application_record['OCCUPATION_TYPE'].map(occupation_dict)

# Confirm that the values in occupation type column have been changed
# print(list(application_record['OCCUPATION_TYPE'].unique()))

# OCCUPATION_TYPE = ['Security staff', 
#                    'Sales staff', 
#                    'Accountants', 
#                    'Laborers', 
#                    'Managers', 
#                    'Drivers', 
#                    'Core staff', 
#                    'High skill tech staff', 
#                    'Cleaning staff', 
#                    'Private service staff', 
#                    'Cooking staff', 
#                    'Low-skill Laborers', 
#                    'Medicine staff', 
#                    'Secretaries', 
#                    'Waiters/barmen staff', 
#                    'HR staff', 
#                    'Realty agents', 
#                    'IT staff']

# Drop unnessessary columns
application_record.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], inplace=True, axis=1)

# Confirm that columns have been dropped
print(application_record.head())

# Convert categorical to numerical variables and map to either 0 or 1 because it is a binary classification task
# 1 includes users who took no loans that month paid within the month or 30 days past the due date while
# 0 includes users who pay within 30 to 149 days past the due date or have overdue debts for more than 150 days
map_status = {'C' : 1,
              'X' : 1,
              '0' : 1,
              '1' : 0,
              '2' : 0,
              '3' : 0,
              '4' : 0,
              '5' : 0}
credit_record["STATUS"] = credit_record['STATUS'].map(map_status)

# Confirm the above
print(credit_record['STATUS'].value_counts())

# Merge both credit and applications records to create a comprehensive dataset
df_credit = application_record.merge(credit_record, how='inner', on=['ID'])
print(df_credit.head())

target = df_credit['STATUS']
features = df_credit.drop(['STATUS'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.75)

train_data = pd. concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

print(train_data['NAME_INCOME_TYPE'].head())
print(train_data['NAME_EDUCATION_TYPE'].head())
print(train_data['NAME_FAMILY_STATUS'].head())
print(train_data['NAME_HOUSING_TYPE'].head())
print(train_data['FLAG_MOBIL'].head())

train_data_file = "train_data.csv"
test_data_file = "test_data.csv"

train_data.to_csv(train_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)

CSV_HEADER = ['ID', 
              'CODE_GENDER', 
              'FLAG_OWN_CAR', 
              'FLAG_OWN_REALTY', 
              'CNT_CHILDREN', 
              'AMT_INCOME_TOTAL', 
              'NAME_INCOME_TYPE', 
              'NAME_EDUCATION_TYPE', 
              'NAME_FAMILY_STATUS', 
              'NAME_HOUSING_TYPE', 
              'FLAG_MOBIL', 
              'FLAG_WORK_PHONE', 
              'FLAG_PHONE', 
              'FLAG_EMAIL', 
              'OCCUPATION_TYPE', 
              'CNT_FAM_MEMBERS', 
              'MONTHS_BALANCE', 
              'STATUS']

# A list of the numerical feature names.
NUMERIC_FEATURE_NAMES = ['ID', 
              'CODE_GENDER', 
              'FLAG_OWN_CAR', 
              'FLAG_OWN_REALTY', 
              'CNT_CHILDREN', 
              'AMT_INCOME_TOTAL',  
              'FLAG_MOBIL', 
              'FLAG_WORK_PHONE', 
              'FLAG_PHONE', 
              'FLAG_EMAIL', 
              'OCCUPATION_TYPE', 
              'CNT_FAM_MEMBERS', 
              'MONTHS_BALANCE', 
              'STATUS']
# A dictionary of the categorical features and their vocabulary.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "workclass": sorted(list(train_data["workclass"].unique())),
    "education": sorted(list(train_data["education"].unique())),
    "marital_status": sorted(list(train_data["marital_status"].unique())),
    "occupation": sorted(list(train_data["occupation"].unique())),
    "relationship": sorted(list(train_data["relationship"].unique())),
    "race": sorted(list(train_data["race"].unique())),
    "gender": sorted(list(train_data["gender"].unique())),
    "native_country": sorted(list(train_data["native_country"].unique())),
}
# Name of the column to be used as instances weight.
WEIGHT_COLUMN_NAME = "fnlwgt"
# A list of the categorical feature names.
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
# A list of all the input features.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

TARGET_LABELS = ['STATUS']

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 265
NUM_EPOCHS = 15

NUM_TRANSFORMER_BLOCKS = 3  # Number of transformer blocks.
NUM_HEADS = 4  # Number of attention heads.
EMBEDDING_DIMS = 16  # Embedding dimensions of the categorical features.
MLP_HIDDEN_UNITS_FACTORS = [
    2,
    1,
]  # MLP hidden layer units, as factors of the number of inputs.
NUM_MLP_BLOCKS = 2  # Number of MLP blocks in the baseline model.


target_label_lookup = layers.StringLookup(
    vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
)


# def prepare_example(features, target):
#     target_index = target_label_lookup(target)
#     weights = features.pop(WEIGHT_COLUMN_NAME)
#     return features, target_index, weights


def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        num_epochs=1,
        header=False,
        na_value="?",
        shuffle=shuffle,
    )
    return dataset.cache()

def run_experiment(
    model,
    train_data_file,
    test_data_file,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )


    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    validation_dataset = get_dataset_from_csv(test_data_file, batch_size)

    print("Start training the model...")
    history = model.fit(
        train_dataset, epochs=num_epochs, validation_data=validation_dataset
    )
    print("Model training finished")

    _, accuracy = model.evaluate(validation_dataset, verbose=0)

    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")

    return history

def create_model_inputs():
    inputs = {}
    for feature_name in CSV_HEADER:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.float32
        )
    return inputs

def encode_inputs(inputs, embedding_dims):

    encoded_categorical_feature_list = []
    numerical_feature_list = []

    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:

            # Get the vocabulary of the categorical feature.
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]

            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int",
            )

            # Convert the string input values into integer indices.
            encoded_feature = lookup(inputs[feature_name])

            # Create an embedding layer with the specified dimensions.
            embedding = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )

            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        else:

            # Use the numerical features as-is.
            numerical_feature = tf.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):

    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)


def create_tabtransformer_classifier(
    num_transformer_blocks,
    num_heads,
    embedding_dims,
    mlp_hidden_units_factors,
    dropout_rate,
    use_column_embedding=False,
):

    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Stack categorical feature embeddings for the Tansformer.
    encoded_categorical_features = tf.stack(encoded_categorical_feature_list, axis=1)
    # Concatenate numerical features.
    numerical_features = layers.concatenate(numerical_feature_list)

    # Add column embedding to categorical feature embeddings.
    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = layers.Embedding(
            input_dim=num_columns, output_dim=embedding_dims
        )
        column_indices = tf.range(start=0, limit=num_columns, delta=1)
        encoded_categorical_features = encoded_categorical_features + column_embedding(
            column_indices
        )

    # Create multiple layers of the Transformer block.
    for block_idx in range(num_transformer_blocks):
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"multihead_attention_{block_idx}",
        )(encoded_categorical_features, encoded_categorical_features)
        # Skip connection 1.
        x = layers.Add(name=f"skip_connection1_{block_idx}")(
            [attention_output, encoded_categorical_features]
        )
        # Layer normalization 1.
        x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
        # Feedforward.
        feedforward_output = create_mlp(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{block_idx}",
        )(x)
        # Skip connection 2.
        x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
        # Layer normalization 2.
        encoded_categorical_features = layers.LayerNormalization(
            name=f"layer_norm2_{block_idx}", epsilon=1e-6
        )(x)

    # Flatten the "contextualized" embeddings of the categorical features.
    categorical_features = layers.Flatten()(encoded_categorical_features)
    # Apply layer normalization to the numerical features.
    numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
    # Prepare the input for the final MLP block.
    features = layers.concatenate([categorical_features, numerical_features])

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


tabtransformer_model = create_tabtransformer_classifier(
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_heads=NUM_HEADS,
    embedding_dims=EMBEDDING_DIMS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)

print("Total model weights:", tabtransformer_model.count_params())
keras.utils.plot_model(tabtransformer_model, show_shapes=True, rankdir="LR")

history = run_experiment(
    model=tabtransformer_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)