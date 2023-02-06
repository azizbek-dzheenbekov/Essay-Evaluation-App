def mcrmse_score(y_true, y_pred):
    rmse_per_column = []
    for idx, col in enumerate(y_true):
        rmse = (((y_pred[idx] - y_true[col]) ** 2).mean()) ** 0.5
        rmse_per_column.append(rmse)
        print(f"RMSE score for column '{col}': {rmse}")

    mcrmse = sum(rmse_per_column)/len(y_pred)
    return mcrmse