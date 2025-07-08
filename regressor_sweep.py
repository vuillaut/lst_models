import wandb
import lightgbm as lgb
from data import read_dl2, read_training_gammas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# --- 1. W&B Sweep Configuration ---
# This dictionary defines the hyperparameter search space and strategy.
sweep_config = {
    'method': 'bayes',  # Bayesian optimization is efficient
    'metric': {
      'name': 'mse',
      'goal': 'minimize'   # We want to minimize the Mean Squared Error
    },
    'parameters': {
        'n_estimators': {
            'distribution': 'int_uniform',
            'min': 200,
            'max': 2000
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-3,
            'max': 0.3
        },
        'num_leaves': {
            'distribution': 'int_uniform',
            'min': 20,
            'max': 150
        },
        'max_depth': {
            'distribution': 'int_uniform',
            'min': -1,
            'max': 50
        },
        'reg_alpha': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 1.0
        },
        'reg_lambda': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 1.0
        },
        'colsample_bytree': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.0
        },
        'subsample': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.0
        },
    }
}

# --- 2. Data Loading and Preprocessing ---
# Encapsulate data loading in a function to avoid reloading in every sweep run
def load_data():
    print("Loading and preparing data...")
    training_data = read_training_gammas(decs=['dec_2276'])
    test_filename = "/mustfs/LAPP-DATA/cta/Data/LST1/MC/DL2/AllSky/20240918_v0.10.12_allsky_nsb_tuning_0.00/TestingDataset/Gamma/dec_2276/node_theta_23.630_az_100.758_/dl2_20240909_allsky_nsb_tuning_0.00_Gamma_test_node_theta_23.630_az_100.758__merged.h5"
    test_data = read_dl2(test_filename)

    energy_regression_features = [
        "log_intensity", "width", "length", "x", "y", "wl", "skewness",
        "kurtosis", "time_gradient", "leakage_intensity_width_2",
        "sin_az_tel", "alt_tel"
    ]
    target = "log_mc_energy"

    X = training_data[energy_regression_features]
    y = training_data[target]

    # Split training data to create a validation set for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(test_data[energy_regression_features])
    y_test = test_data[target]
    
    print("Data loaded and prepared.")
    return X_train, y_train, X_val, y_val, X_test, y_test, test_data

# --- 3. Training Function ---
# This function will be called by the wandb agent for each run.
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Define the model with hyperparameters from wandb
        model = lgb.LGBMRegressor(
            device='gpu',  # Use GPU for training
            random_state=42,
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            colsample_bytree=config.colsample_bytree,
            subsample=config.subsample,
            n_jobs=-1
        )

        print("Training model with params:", config)
        # Train the model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics to wandb
        wandb.log({'mse': mse, 'r2': r2})
        print(f"Run finished. MSE: {mse:.4f}, R2: {r2:.4f}")


# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # Load data once
    X_train, y_train, X_val, y_val, X_test, y_test, test_data = load_data()

    # Initialize the sweep
    # You will be prompted to log in to W&B if you haven't already.
    project_name = "lstchain_models"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    print(f"\n--- Starting W&B Sweep ---")
    print(f"Project: {project_name}")
    print(f"Sweep ID: {sweep_id}")
    print("Run the following command to start the agent:")
    print(f"wandb agent {sweep_id}")

    # This will start the agent. We'll set it to run a specific number of trials.
    # You can change the 'count' to run more or fewer experiments.
    wandb.agent(sweep_id, train, count=50)

    # After the sweep, you can go to the W&B dashboard to find the best model,
    # then use its parameters to train a final model for deployment.
    print("\n--- Sweep Finished ---")
    print(f"Visit the W&B project page to see your results: https://wandb.ai/gammalearn/{project_name}")
    print("Replace YOUR_USERNAME with your actual W&B username.")
