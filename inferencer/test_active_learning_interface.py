from active_learning_interface import active_learning_interface

fold = 10
rounds = 100
al = active_learning_interface(
    target_building='rice',
    fold=fold,
    rounds=rounds
    )

al.run_auto()

