import os


steps = [

    "python src/load_data.py",

    "python src/preprocess.py",

    "python src/feature_extraction.py",

    "python src/train.py",

    "python src/evaluate.py",

    "python src/predict.py"

]


for step in steps:

    print("\n==============================")
    print("Running:", step)
    print("==============================\n")

    os.system(step)


print("\nPROJECT EXECUTION COMPLETED")