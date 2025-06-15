import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# DNN Architecture
DNN_ARCHITECTURES = [
    {'layers': [256, 128, 64], 'dropouts': [0.3, 0.3, 0.2]},
    {'layers': [512, 256, 128], 'dropouts': [0.4, 0.3, 0.2]},
    {'layers': [128, 64, 32], 'dropouts': [0.2, 0.2, 0.1]},
    {'layers': [256, 256, 128, 64], 'dropouts': [0.3, 0.3, 0.2, 0.1]},
    {'layers': [512, 128, 64], 'dropouts': [0.4, 0.2, 0.1]}
]


def load_and_preprocess_data(file_path):
    # load data
    data = pd.read_csv(file_path)

    # extract features and targets
    wl_columns = [col for col in data.columns if col.startswith('WL_')]
    X = data[wl_columns].values
    y = data[['Data XRF - Ni', 'Data XRF - Fe', 'Data XRF - SiO2', 'Data XRF - MgO']].values

    # calculate spectral derivatives
    def calculate_derivatives(spectra, window_length=5, polyorder=2):
        first_deriv = np.gradient(spectra, axis=1)
        second_deriv = np.gradient(first_deriv, axis=1)
        first_deriv_sg = signal.savgol_filter(spectra, window_length=window_length,
                                              polyorder=polyorder, deriv=1, axis=1)
        second_deriv_sg = signal.savgol_filter(spectra, window_length=window_length,
                                               polyorder=polyorder, deriv=2, axis=1)
        return first_deriv, second_deriv, first_deriv_sg, second_deriv_sg

    first_deriv, second_deriv, first_deriv_sg, second_deriv_sg = calculate_derivatives(X)
    X_combined = np.concatenate([X, first_deriv, second_deriv, first_deriv_sg, second_deriv_sg], axis=1)

    # apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    pca = PCA(n_components=75, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_)}")

    # split data based on Dataset column
    train_mask = data['Dataset'] == 'Train'
    test_mask = data['Dataset'] == 'Test'

    X_train = X_pca[train_mask]
    y_train = y[train_mask]
    X_test = X_pca[test_mask]
    y_test = y[test_mask]

    return X_train, y_train, X_test, y_test


def build_dnn_model(input_shape, architecture):
    model = Sequential()

    # input layer
    model.add(Dense(architecture['layers'][0], activation='relu',
                    input_shape=(input_shape,), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(architecture['dropouts'][0]))

    # hidden layers
    for units, dropout in zip(architecture['layers'][1:-1], architecture['dropouts'][1:-1]):
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    # output layer
    model.add(Dense(4))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # define target names
    targets = ['Ni', 'Fe', 'SiO2', 'MgO']

    # create evaluation table
    results = []

    for i, target in enumerate(targets):
        # calculate metrics for train set
        train_mae = mean_absolute_error(y_train[:, i], y_train_pred[:, i])
        train_rmse = np.sqrt(mean_squared_error(y_train[:, i], y_train_pred[:, i]))
        train_r2 = r2_score(y_train[:, i], y_train_pred[:, i])

        # calculate metrics for test set
        test_mae = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
        test_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i]))
        test_r2 = r2_score(y_test[:, i], y_test_pred[:, i])

        # append results
        results.append([f"{target} - Train", train_mae, train_rmse, train_r2])
        results.append([f"{target} - Test", test_mae, test_rmse, test_r2])

    # create DataFrame for nice display
    eval_df = pd.DataFrame(results, columns=['Target', 'MAE', 'RMSE', 'R2'])

    return eval_df, y_train_pred, y_test_pred


def plot_actual_vs_predicted(y_true, y_pred, target_names, set_type='Train'):
    plt.figure(figsize=(12, 8))
    for i, target in enumerate(target_names):
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        plt.plot([min(y_true[:, i]), max(y_true[:, i])],
                 [min(y_true[:, i]), max(y_true[:, i])],
                 '--r')
        plt.title(f'{target} - {set_type}\nActual vs Predicted')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_and_evaluate():
    file_path = 'new-spectral-senggigi-raw.csv'
    X_train, y_train, X_test, y_test = load_and_preprocess_data(file_path)

    results = []

    for i, arch in enumerate(DNN_ARCHITECTURES):
        print(f"\nTraining Model {i + 1} with architecture: {arch}")

        model = build_dnn_model(X_train.shape[1], arch)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # evaluate
        eval_results, y_train_pred, y_test_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
        eval_results['Model'] = f"Model {i + 1}"
        results.append(eval_results)

        # plot actual vs predicted
        target_names = ['Ni', 'Fe', 'SiO2', 'MgO']
        plot_actual_vs_predicted(y_train, y_train_pred, target_names, 'Train')
        plot_actual_vs_predicted(y_test, y_test_pred, target_names, 'Test')

        # plot training history
        plt.figure(figsize=(12, 5))
        plt.suptitle(f"Model Architecture: {arch['layers']}")
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Val MAE')
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # evaluate
    final_results = pd.concat(results)
    print("\nFinal Evaluation Results for All Models:")
    print(final_results.to_string(index=False))

    # save evaluate results to CSV
    final_results.to_csv('dnn_architecture_results.csv', index=False)
    print("\nResults saved to 'dnn_architecture_results.csv'")


if __name__ == "__main__":
    train_and_evaluate()