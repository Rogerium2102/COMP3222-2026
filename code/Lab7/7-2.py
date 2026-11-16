from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def calculate_metrics(y_test, y_pred):
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

def predict_output(model,name):
    y_pred = model.predict(X_test)
    rmse, mae, r2 = calculate_metrics(y_test, y_pred)
    print(name)
    print("RMSE: $" + str(rmse))
    print("MAE: $" + str(mae))
    print("R2: ", r2)

def create_plot(mlp):
    loss_curve = mlp.loss_curve_
    val_scores = mlp.validation_scores_
    if loss_curve is None or val_scores is None:
        print(loss_curve, val_scores)
    epochs = range(1, len(loss_curve) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left y-axis: Training Loss
    color_loss = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color=color_loss)
    line1 = ax1.plot(epochs, loss_curve, color=color_loss, label="Training Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right y-axis: Validation Score
    ax2 = ax1.twinx()
    color_val = "tab:blue"
    ax2.set_ylabel("Validation Score (Accuracy)", color=color_val)
    line2 = ax2.plot(epochs, val_scores, color=color_val, linestyle="--", label="Validation Score")
    ax2.tick_params(axis="y", labelcolor=color_val)

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.title("MLP Training Loss and Validation Score per Epoch")
    fig.tight_layout()
    plt.show()


X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lin_reg = Pipeline([
    ('scalar', StandardScaler()),
    ('lin_reg', LinearRegression())
])

lin_reg.fit(X_train, y_train)
predict_output(lin_reg,"LINEAR REGRESSION")

mlp_reg = Pipeline([
    ('scalar', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(64,64), early_stopping=True))
])

mlp_reg.fit(X_train, y_train)
predict_output(mlp_reg, "MLP REGRESSION")
train_rmse = root_mean_squared_error(y_train, mlp_reg.predict(X_train))
print(f"TRAIN RMSE: ${train_rmse}")
mlp_pure = mlp_reg.named_steps['mlp']
if mlp_pure is not None:
    create_plot(mlp_pure)
