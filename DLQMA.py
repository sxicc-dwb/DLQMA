import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Input, layers, models, optimizers, callbacks
from tensorflow.keras.layers import Layer
import seaborn as sns

from tensorflow.keras.models import Model


import matplotlib as mpl
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, \
    ReduceLROnPlateau
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr

from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error

from sklearn.manifold import TSNE

def visualize_features(model, Xs, y, filename='tsne_visualization.png'):

    feature_extractor = Model(inputs=model.inputs,
                            outputs=model.get_layer('regression_batch_normalization_2').output)


    Xs_3d = [X.reshape((X.shape[0], X.shape[1], 1)) if X.ndim == 2 else X for X in Xs]


    features = feature_extractor.predict(Xs_3d)

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=y.flatten(),
                   palette='viridis', alpha=0.7)
    plt.title('t-SNE Visualization of Final Layer Features')
    plt.savefig(filename)
    plt.close()



mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False



def make_gradcam_heatmap(model, Xs, last_conv_layer_name, pred_index=0):

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(Xs)
        class_channel = preds[:, pred_index]


    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)


    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()


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


class SpatialPyramidPooling(Layer):


    def __init__(self, pool_list, **kwargs):
        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {tf, th}'
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i * i for i in pool_list])
        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"{len(input_shape)}")
        self.nb_channels = int(input_shape[-1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None, **kwargs):
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

    print("ok")
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

    print("model ok")
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


    print("next")

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


    print("\n")

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


    print("\n")
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


    if validation_data is not None:

        Xs_valid_3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs_valid]
        y_valid = validation_data[1]['classification']
        visualize_features(model, Xs_valid_3d, y_valid, 'validation_tsne.png')

    visualize_features(model, Xs_3d, y, 'training_tsne.png')

    return combined_history


def save_model(model, model_path):

    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)


    model.save(f"{model_path}.h5")
    print(f"save to {model_path}.h5")


    if hasattr(model, 'history') and hasattr(model.history, 'history'):
        with open(f"{model_path}_history.pkl", 'wb') as f:
            pickle.dump(model.history.history, f)
        print(f"save to {model_path}_history.pkl")


def load_model(model_path):

    model_file = f"{model_path}.h5"
    history_file = f"{model_path}_history.pkl"

    if not os.path.exists(model_file):
        print(f"no {model_file} exist")
        return None

    model = models.load_model(model_file, custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})
    print(f"loaded {model_file}")


    if os.path.exists(history_file):
        try:
            with open(history_file, 'rb') as f:
                history_dict = pickle.load(f)


            history = callbacks.History()
            history.history = history_dict
            model.history = history

            print(f"ok: {history_file}")
            print(f"ok: {list(history_dict.keys())}")
        except Exception as e:
            print(f"wow: {e}")

    return model


def predict(model, Xs):

    Xs_3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    return model.predict(Xs_3d)


def evaluate(model, Xs, y, conc=None):

    Xs_3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]

    if isinstance(model.output, list) and conc is not None:

        eval_results = model.evaluate(Xs_3d, {'classification': y, 'regression': conc})
        predictions = model.predict(Xs_3d)
        y_pred, conc_pred = predictions


        y_pred_class = (y_pred > 0.5).astype(int)
        auc = roc_auc_score(y, y_pred)
        precision = precision_score(y, y_pred_class)
        recall = recall_score(y, y_pred_class)
        f1 = f1_score(y, y_pred_class)


        rmse = np.sqrt(mean_squared_error(conc, conc_pred))
        pearson_r, _ = pearsonr(conc.flatten(), conc_pred.flatten())

        return eval_results + [auc, precision, recall, f1, rmse, pearson_r]
    else:

        eval_results = model.evaluate(Xs_3d, y)
        predictions = model.predict(Xs_3d)
        auc = roc_auc_score(y, predictions)
        precision = precision_score(y, (predictions > 0.5).astype(int))
        recall = recall_score(y, (predictions > 0.5).astype(int))
        f1 = f1_score(y, (predictions > 0.5).astype(int))

        return eval_results + [auc, precision, recall, f1]


def plot_training_history(history_dict, model_type='classification'):

    if not history_dict:
        print("Warning: No training history to plot")
        return


    if model_type == 'classification':

        acc_key = next((k for k in history_dict.keys() if 'acc' in k.lower() and not k.startswith('val_')), None)
        val_acc_key = next((k for k in history_dict.keys() if 'acc' in k.lower() and k.startswith('val_')), None)
        loss_key = 'loss'
        val_loss_key = 'val_loss'

        if acc_key and val_acc_key:
            plt.figure(figsize=(12, 5))


            plt.subplot(1, 2, 1)
            plt.plot(history_dict[acc_key], label='Training')
            plt.plot(history_dict[val_acc_key], label='Validation')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()


            plt.subplot(1, 2, 2)
            plt.plot(history_dict[loss_key], label='Training')
            plt.plot(history_dict[val_loss_key], label='Validation')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()

            plt.tight_layout()
        else:
            print(f"no: {list(history_dict.keys())}")

    elif model_type == 'combined':

        class_acc_key = next((k for k in history_dict.keys() if
                              'classification' in k and 'acc' in k.lower() and not k.startswith('val_')), None)
        val_class_acc_key = next(
            (k for k in history_dict.keys() if 'classification' in k and 'acc' in k.lower() and k.startswith('val_')),
            None)
        reg_loss_key = next(
            (k for k in history_dict.keys() if 'regression' in k and 'loss' in k.lower() and not k.startswith('val_')),
            None)
        val_reg_loss_key = next(
            (k for k in history_dict.keys() if 'regression' in k and 'loss' in k.lower() and k.startswith('val_')),
            None)

        if class_acc_key and val_class_acc_key and reg_loss_key and val_reg_loss_key:
            plt.figure(figsize=(12, 5))


            plt.subplot(1, 2, 1)
            plt.plot(history_dict[class_acc_key], label='Training')
            plt.plot(history_dict[val_class_acc_key], label='Validation')
            plt.title('Classification Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()


            plt.subplot(1, 2, 2)
            plt.plot(history_dict[reg_loss_key], label='Training')
            plt.plot(history_dict[val_reg_loss_key], label='Validation')
            plt.title('Regression Loss')
            plt.ylabel('MSE')
            plt.xlabel('Epoch')
            plt.legend()

            plt.tight_layout()
        else:
            print(f"0: {list(history_dict.keys())}")
            if class_acc_key:
                print(f"cla: {class_acc_key}")
            if val_class_acc_key:
                print(f"val: {val_class_acc_key}")
            if reg_loss_key:
                print(f"reloss: {reg_loss_key}")
            if val_reg_loss_key:
                print(f"valreloss: {val_reg_loss_key}")


def normalize_spectra(spectra_data):
    normalized = np.zeros_like(spectra_data)
    for i in range(spectra_data.shape[0]):
        spectrum = spectra_data[i]
        mean = np.mean(spectrum)
        std = np.std(spectrum) + 1e-8
        normalized[i] = (spectrum - mean) / std
    return normalized


def analyze_residuals(y_true, y_pred):
    residuals = y_pred - y_true
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('True Value')
    plt.ylabel('Residual')
    plt.title('Residuals vs True Value')

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=20)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.savefig('residual_analysis.png')


def process_mixture_results(result_df, threshold=0.5):

    valid_compounds = result_df[result_df['Probability'] > threshold].copy()

    if len(valid_compounds) > 0:

        total_contribution = valid_compounds['ModelContribution'].sum()


        result_df['Relative_Concentration'] = 0.0


        if total_contribution > 0:
            result_df.loc[valid_compounds.index, 'Relative_Concentration'] = \
                (valid_compounds['ModelContribution'] / total_contribution) * 100


        result_df = result_df.sort_values(by=['Probability'], ascending=False)

    return result_df




def visualize_pretrained_model(model_path, Xs, y, filename):

    model = load_model(model_path)
    if model is not None:
        visualize_features(model, Xs, y, filename)
        print(f"save {filename}")


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
        EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001),
        ModelCheckpoint(
            filepath=f'{combined_model_path}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
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
            combined_model = load_model(combined_model_path)
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


        print("\nEvaluating combined model...")
        results = evaluate(
            combined_model,
            [aug_valid['R'], aug_valid['S']],
            aug_valid['y'],
            aug_valid['conc']
        )
        print(
            f"Validation Performance: Total Loss={results[0]:.4f}, Classification Loss={results[1]:.4f}, Regression Loss={results[2]:.4f}")
        print(f"Classification Accuracy: {results[3]:.4f}")


        if hasattr(combined_model, 'history') and combined_model.history.history:
            plot_training_history(combined_model.history.history, 'combined')
            plt.savefig(f"{combined_model_path}_history.png")
            plt.show()


        print("\n===== Testing Model =====")


        results = evaluate(
            combined_model,
            [aug_test['R'], aug_test['S']],
            aug_test['y'],
            aug_test['conc']
        )
        print(
            f"Test Performance: Total Loss={results[0]:.4f}, Classification Loss={results[1]:.4f}, Regression Loss={results[2]:.4f}")
        print(f"Classification Accuracy: {results[3]:.4f}")

        if 'classification' in combined_model.output_names:
            y_pred, conc_pred = predict(combined_model, [aug_test['R'], aug_test['S']])
            y_pred_class = (y_pred > 0.5).astype(int)
            cnf_matrix = confusion_matrix(aug_test['y'], y_pred_class)

            eval_dir = os.path.join(combined_model_path, 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            class_report = classification_report(
                aug_test['y'],
                (y_pred > 0.5).astype(int),
                output_dict=True
            )
            pd.DataFrame(class_report).transpose().to_csv(
                os.path.join(eval_dir, f'classification_report_{timestamp}.csv')
            )

            reg_metrics = pd.DataFrame({
                'metric': ['MSE', 'MAE', 'RMSE', 'Pearson R'],
                'value': [
                    mean_squared_error(aug_test['conc'], conc_pred),
                    mean_absolute_error(aug_test['conc'], conc_pred),
                    np.sqrt(mean_squared_error(aug_test['conc'], conc_pred)),
                    pearsonr(aug_test['conc'].flatten(), conc_pred.flatten())[0]
                ]
            })
            reg_metrics.to_csv(
                os.path.join(eval_dir, f'regression_metrics_{timestamp}.csv'),
                index=False
            )


            print(f"Classification confusion matrix:\n{cnf_matrix}")
            np.savetxt('test_set_cnf_matrix.csv', cnf_matrix, delimiter=',')


            mse = mean_squared_error(aug_test['conc'], conc_pred)
            mae = mean_absolute_error(aug_test['conc'], conc_pred)
            print(f"Regression performance: MSE={mse:.6f}, MAE={mae:.6f}")


            conc_df = pd.DataFrame({
                'true_conc': aug_test['conc'].flatten(),
                'pred_conc': conc_pred.flatten()
            })
            conc_df.to_csv('concentration_results.csv', index=False)
            print("Concentration prediction results saved to concentration_results.csv")


    if ENABLE_PREDICTION:
        print("\n===== Executing Prediction =====")


        if os.path.exists(f"{combined_model_path}.h5"):
            combined_model = load_model(combined_model_path)
            print(f"Loaded combined model: {combined_model_path}.h5")
        else:
            print(f"Error: Combined model file {combined_model_path}.h5 not found")
            exit(1)


        results = evaluate(
            combined_model,
            [aug_test['R'], aug_test['S']],
            aug_test['y'],
            aug_test['conc']
        )
        print(
            f"Test Performance: Total Loss={results[0]:.4f}, Classification Loss={results[1]:.4f}, Regression Loss={results[2]:.4f}")
        print(f"Classification Accuracy: {results[3]:.4f}")


        from readBruker import read_bruker_hs_base

        standards = read_bruker_hs_base('mydata/standards', False, True, False)
        mixture = read_bruker_hs_base('mydata/mixture', False, True, False)

        os.makedirs('results', exist_ok=True)

        for i, query in enumerate(mixture[:32]):
            print(f"Processing mixture {i + 1}/{len(mixture)}: {query['name']}")

            p = query['ppm'].shape[0]
            n_std = len(standards)


            R = np.zeros((n_std, p), dtype=np.float32)
            Q = np.zeros((n_std, p), dtype=np.float32)

            for j in range(n_std):
                R[j,] = standards[j]['fid']
                Q[j,] = query['fid']


            y_pred, conc_pred = predict(combined_model, [R, Q])


            result_df = pd.DataFrame({
                'Name': [standards[j]['name'] for j in range(n_std)],
                'Probability': y_pred.flatten(),
                'ModelContribution': conc_pred.flatten()
            })


            result = process_mixture_results(result_df, threshold=0.5)


            output_path = f"results/mixture_result_{query['name']}.csv"
            result.to_csv(output_path, sep=',', encoding='utf_8_sig', index=False)


            print(f"\nAnalysis results for mixture {query['name']}:")
            print("Detected compounds and their contents:")
            valid_results = result[result['Probability'] > 0.5]
            if len(valid_results) > 0:
                display_results = valid_results[
                    ['Name', 'Probability', 'ModelContribution', 'Relative_Concentration']].copy()
                display_results = display_results.rename(columns={
                    'Relative_Concentration': 'Relative_%',
                    'ModelContribution': 'Contribution_Factor'
                })

                display_results['Relative_%'] = display_results['Relative_%'].map('{:.2f}%'.format)

                print(display_results.to_string(index=False))
                print(f"\nTotal relative content: {valid_results['Relative_Concentration'].sum():.2f}%")


                print("\nConcentration Explanation:")
                print(
                    "- Contribution_Factor: Model predicted contribution factor - represents the relative signal strength of this component in the mixture")
                print("- Relative_%: Mixture composition ratio - relative percentage of each component in the mixture")


                print("\nMixture composition summary:")
                ratio_summary = ""
                for idx, row in display_results.sort_values('Relative_%', ascending=False).iterrows():
                    ratio_summary += f"{row['Name']}:{row['Relative_%']} "
                print(ratio_summary)
            else:
                print("No compounds detected")

