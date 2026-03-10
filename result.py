import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

def get_flops(model):

    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        tf.TensorSpec([1,128,128,3], tf.float32)
    )

    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

        flops = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=run_meta,
            cmd='op',
            options=opts
        )

    return flops.total_float_ops


DATASET_PATH = "croppedImages_aug/val"
MODEL_PATH = "best_model.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

labels = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor"
}

# ===============================
# Carrega dataset
# ===============================

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
class_names = dataset.class_names

for i in range(len(class_names)):
    class_names[i] = labels[int(class_names[i])]


print(class_names)

NUM_CLASSES = len(class_names)

print("Classes:", class_names)


# ===============================
# Carrega modelo
# ===============================

model = tf.keras.models.load_model(MODEL_PATH)


# ===============================
# Predições
# ===============================

y_true = []
y_pred = []

print("Running inference...")

for images, labels in dataset:

    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)

    y_pred.extend(preds)
    y_true.extend(labels.numpy())


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ===============================
# Métricas gerais
# ===============================

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro')
balanced_acc = balanced_accuracy_score(y_true, y_pred)

print("\n===== METRICS =====")

print("Accuracy:", accuracy)
print("Macro F1:", macro_f1)
print("Balanced Accuracy:", balanced_acc)


# ===============================
# Precision / Recall / F1 por classe
# ===============================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv("classification_report.csv")

print("\nClassification report saved.")


# ===============================
# Matriz de confusão
# ===============================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

print("Confusion matrix saved.")


# ===============================
# Tempo de inferência
# ===============================

sample_batch = next(iter(dataset))[0]

start = time.time()

model.predict(sample_batch)

end = time.time()

time_per_batch = end - start
time_per_image = time_per_batch / len(sample_batch)

fps = 1 / time_per_image

print("\n===== INFERENCE =====")

print("Time per image:", time_per_image)
print("Images per second (FPS):", fps)


# ===============================
# Parâmetros
# ===============================

params = model.count_params()

print("\nParameters:", params)


# ===============================
# FLOPs
# ===============================

flops = get_flops(model)
print("FLOPs:", flops)


# ===============================
# Número de imagens
# ===============================

num_images = len(y_true)

print("Images evaluated:", num_images)


# ===============================
# Salvar resumo
# ===============================

summary = {

    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "balanced_accuracy": balanced_acc,
    "fps": fps,
    "time_per_image": time_per_image,
    "parameters": params,
    "flops": flops,
    "num_images": num_images
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv("model_metrics_summary.csv", index=False)

print("\nSummary saved to model_metrics_summary.csv")