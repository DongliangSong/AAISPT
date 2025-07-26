# -*- coding: utf-8 -*-
# @Time    : 2024/7/7 20:11
# @Author  : Dongliang

# Multi-process hyperparameter optimization
import importlib
import multiprocessing

# Defines the packages and modules to be run sequentially.
scripts = [
    ("Decision_Tree", "Optimalization"),
    ("Decision_Tree", "Decision_Tree"),
    ("Random_forest", "Optimalization"),
    ("Random_forest", "Random_forest"),
    ("KNN", "Optimalization"),
    ("KNN", "KNN"),
    ("GBDT", "Optimalization"),
    ("GBDT", "GBDT"),
    ("Logistic_regression", "Logistic_regression"),
    ("SVM", "Optimalization"),
    ("SVM", "SVM"),
]


def run_script(package, module):
    spec = importlib.util.find_spec(f"{package}.{module}")
    if spec is None:
        print(f"Cannot find module {package}.{module}")
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'main'):
        module.main()
    else:
        print(f"{package}.{module} does not have a main function")


def run_script_in_process(package, module):
    p = multiprocessing.Process(target=run_script, args=(package, module))
    p.start()
    p.join()


def main(scripts):
    for package, module in scripts:
        run_script_in_process(package, module)


if __name__ == "__main__":
    main(scripts)
