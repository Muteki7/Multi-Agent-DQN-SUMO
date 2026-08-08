import optuna

def objective(trail):
    x= trail.suggest_float("x", -100, 100)
    y=trail.suggest_categorical("y", [-1,0,1])
    return (x**2) +  y

study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="Optuna_test_quadratic"
)
study.optimize(objective, n_trials=100)
print(f"Best value: {study.best_value}, (Best params: {study.best_params})")
