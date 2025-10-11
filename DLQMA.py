import os
import pickle
import datetime
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, layers, models, optimizers, callbacks
from tensorflow.keras.layers import Layer
import numpy as np



class SpatialPyramidPooling(Layer):
    def __init__(self, pool_list, **kwargs):
        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {tf, th}'
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i * i for i in pool_list])
        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"SPP：{len(input_shape)}")
        self.nb_channels = int(input_shape[-1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, ):
        input_shape = K.shape(x)
        num_rows = input_shape[1]
        num_cols = input_shape[2]
        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]
        outputs = []
        for pool_num, num_pool_regions in enumerate(self.pool_list):
            for ix in range(num_pool_regions):
                for iy in range(num_pool_regions):
                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]
                    y1 = iy * row_length[pool_num]
                    y2 = iy * row_length[pool_num] + row_length[pool_num]
                    x1 = K.cast(K.round(x1), 'int32')
                    x2 = K.cast(K.round(x2), 'int32')
                    y1 = K.cast(K.round(y1), 'int32')
                    y2 = K.cast(K.round(y2), 'int32')
                    new_shape = [input_shape[0], y2 - y1,
                                 x2 - x1, input_shape[3]]
                    x_crop = x[:, y1:y2, x1:x2, :]
                    xm = K.reshape(x_crop, new_shape)
                    pooled_val = K.max(xm, axis=(1, 2))
                    outputs.append(pooled_val)
        if self.dim_ordering == 'channels_first':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'channels_last':
            outputs = K.concatenate(outputs, axis=0)
            outputs = K.reshape(outputs, (self.num_outputs_per_channel, input_shape[0], self.nb_channels))
            outputs = K.permute_dimensions(outputs, (1, 0, 2))
            outputs = K.reshape(outputs, (input_shape[0], self.num_outputs_per_channel * self.nb_channels))
        return outputs


def create_input_layers(xshapes):
    inputs = []
    for xshape in xshapes:
        input_shape_x = (xshape[1], 1)
        input_x = Input(shape=input_shape_x)
        inputs.append(input_x)
    return inputs


def create_convolution_layers(inputs, num_layers=0):
    convs = []
    for input_x in inputs:
        conv = layers.Conv1D(32, 5, kernel_initializer='he_normal', input_shape=input_x.get_shape())(input_x)
        conv = layers.Activation('relu')(conv)
        conv = layers.MaxPooling1D(strides=2, padding='valid')(conv)
        for i in range(num_layers):
            conv = layers.Conv1D(32, 5, kernel_initializer='he_normal')(conv)
            conv = layers.Activation('relu')(conv)
            conv = layers.MaxPooling1D(strides=2, padding='valid')(conv)
        convs.append(conv)
    return convs


def create_classification_model(xshapes, num_conv_layers, lr=0.00001):
    inputs = create_input_layers(xshapes)
    convs = create_convolution_layers(inputs, num_layers=num_conv_layers)
    if len(convs) >= 2:
        conv_merge = layers.concatenate(convs, 2)
        conv_merge = tf.expand_dims(conv_merge, -1)
        conv1 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(conv_merge)
        conv1 = tf.keras.layers.Activation('relu')(conv1)
    else:
        conv1 = convs[0]
        conv1 = tf.expand_dims(conv1, -1)
    spp = SpatialPyramidPooling([1, 2, 3, 4])(conv1)
    dense = layers.Dense(64, activation='relu')(spp)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(0.3)(dense)
    output = layers.Dense(1, activation='sigmoid', name='classification')(dense)
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def create_multitask_model(xshapes, num_conv_layers, lr=0.0001):
    inputs = create_input_layers(xshapes)
    convs = create_convolution_layers(inputs, num_layers=num_conv_layers)
    if len(convs) >= 2:
        conv_merge = layers.concatenate(convs, 2)
        conv_merge = tf.expand_dims(conv_merge, -1)
        conv1 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(conv_merge)
        conv1 = tf.keras.layers.Activation('relu')(conv1)
    else:
        conv1 = convs[0]
        conv1 = tf.expand_dims(conv1, -1)
    shared_features = SpatialPyramidPooling([1, 2, 3, 4])(conv1)
    classification_dense1 = layers.Dense(64, activation='relu', name='classification_dense1')(shared_features)
    classification_bn1 = layers.BatchNormalization(name='classification_batch_normalization')(classification_dense1)
    classification_dropout = layers.Dropout(0.3, name='classification_dropout')(classification_bn1)
    classification_output = layers.Dense(1, activation='sigmoid', name='classification')(classification_dropout)
    regression_dense1 = layers.Dense(128, activation='relu', name='regression_dense1')(shared_features)
    regression_bn1 = layers.BatchNormalization(name='regression_batch_normalization_1')(regression_dense1)
    regression_dropout1 = layers.Dropout(0.3, name='regression_dropout_1')(regression_bn1)
    regression_dense2 = layers.Dense(64, activation='relu', name='regression_dense2')(regression_dropout1)
    regression_bn2 = layers.BatchNormalization(name='regression_batch_normalization_2')(regression_dense2)
    regression_output = layers.Dense(1, activation='linear', name='regression')(regression_bn2)
    model = models.Model(inputs=inputs, outputs=[classification_output, regression_output])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss={
            'classification': 'binary_crossentropy',
            'regression': 'mse'
        },
        loss_weights={
            'classification': 0.6,
            'regression': 0.4
        },
        metrics={
            'classification': [tf.keras.metrics.BinaryAccuracy(name='accuracy')],
            'regression': [tf.keras.metrics.MeanAbsoluteError(name='mae')]
        }
    )
    model.summary()
    return model


def train_enhanced_model(model, Xs, y, conc, batch_size=64,
                         phase1_epochs=80, phase2_epochs=20, phase3_epochs=30,
                         Xs_valid=None, y_valid=None, conc_valid=None, callbacks=None):
    Xs_3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    if Xs_valid is not None and y_valid is not None and conc_valid is not None:
        Xs_valid_3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs_valid]
        validation_data = (
            Xs_valid_3d,
            {
                'classification': y_valid,
                'regression': conc_valid
            }
        )
    else:
        validation_data = None

    for layer in model.layers:
        if 'regression' in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss={
            'classification': 'binary_crossentropy',
            'regression': 'mse'
        },
        loss_weights={
            'classification': 1.0,
            'regression': 0.0
        },
        metrics={
            'classification': ['accuracy'],
            'regression': ['mae']
        }
    )
    history1 = model.fit(
        Xs_3d,
        {'classification': y, 'regression': conc},
        batch_size=batch_size,
        epochs=phase1_epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )

    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.00005),
        loss={
            'classification': 'binary_crossentropy',
            'regression': 'mse'
        },
        loss_weights={
            'classification': 0.7,
            'regression': 0.3
        },
        metrics={
            'classification': ['accuracy'],
            'regression': ['mae']
        }
    )
    history2 = model.fit(
        Xs_3d,
        {'classification': y, 'regression': conc},
        batch_size=batch_size,
        epochs=phase2_epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )

    print("\3")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.00001),
        loss={
            'classification': 'binary_crossentropy',
            'regression': 'mse'
        },
        loss_weights={
            'classification': 0.6,
            'regression': 0.4
        },
        metrics={
            'classification': ['accuracy'],
            'regression': ['mae']
        }
    )
    history3 = model.fit(
        Xs_3d,
        {'classification': y, 'regression': conc},
        batch_size=batch_size,
        epochs=phase3_epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )

    combined_history = {}
    for h in [history1.history, history2.history, history3.history]:
        for k, v in h.items():
            if k not in combined_history:
                combined_history[k] = []
            combined_history[k].extend(v)
    return combined_history


def save_model(model, model_path):
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(f"{model_path}.h5")
    print(f"save {model_path}.h5")
    if hasattr(model, 'history') and hasattr(model.history, 'history'):
        with open(f"{model_path}_history.pkl", 'wb') as f:
            pickle.dump(model.history.history, f)
        print(f"save {model_path}_history.pkl")


if __name__ == "__main__":
    model_dir = 'model/'
    os.makedirs(model_dir, exist_ok=True)
    classification_model_path = os.path.join(model_dir, 'classification_model')
    combined_model_path = os.path.join(model_dir, 'combined_model')

    BATCH_SIZE = 64
    CLASSIFICATION_EPOCHS = 100
    REGRESSION_EPOCHS = 200
    LEARNING_RATE_CLASS = 0.0001
    LEARNING_RATE_REG = 0.0001
    CONV_LAYERS = 6
    TRAIN_MODEL = False
    ENABLE_PREDICTION = True

    print("Loading data...")
    try:
        with open('aug/data_augment_train.pkl', 'rb') as f:
            aug_train = pickle.load(f)
            with open('aug/data_augment_valid.pkl', 'rb') as f:
                aug_valid = pickle.load(f)
        with open('aug/data_augment_test.pkl', 'rb') as f:
            aug_test = pickle.load(f)
        for dataset in [aug_train, aug_valid, aug_test]:
            if 'conc' not in dataset:
                print(f"Warning: No concentration labels in dataset, using zero vector")
                dataset['conc'] = np.zeros((dataset['y'].shape[0], 1))
            elif len(dataset['conc'].shape) == 1:
                dataset['conc'] = dataset['conc'].reshape(-1, 1)
        print(
            f"Data loaded: Training {len(aug_train['y'])} samples, Validation {len(aug_valid['y'])} samples, Test {len(aug_test['y'])} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001),
        callbacks.ModelCheckpoint(
            filepath=f'{combined_model_path}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        callbacks.CSVLogger(
            f'{model_dir}/training_log.csv',
            append=True,
            separator=','
        )
    ]

    if TRAIN_MODEL:
        print("\n===== Training Multi-task Model =====")
        if os.path.exists(f"{combined_model_path}.h5"):
            print(f"Loading existing combined model: {combined_model_path}.h5")
            combined_model = create_multitask_model(
                [aug_train['R'].shape, aug_train['S'].shape],
                num_conv_layers=CONV_LAYERS,
                lr=LEARNING_RATE_REG
            )
        else:
            print("Creating and training a new combined model...")
            combined_model = create_multitask_model(
                [aug_train['R'].shape, aug_train['S'].shape],
                num_conv_layers=CONV_LAYERS,
                lr=LEARNING_RATE_REG
            )
            train_enhanced_model(
                combined_model,
                [aug_train['R'], aug_train['S']],
                aug_train['y'],
                aug_train['conc'],
                batch_size=BATCH_SIZE,
                phase1_epochs=80,
                phase2_epochs=20,
                phase3_epochs=30,
                Xs_valid=[aug_valid['R'], aug_valid['S']],
                y_valid=aug_valid['y'],
                conc_valid=aug_valid['conc'],
                callbacks=callbacks_list
            )
            save_model(combined_model, combined_model_path)
