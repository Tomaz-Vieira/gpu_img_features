import shutil
import subprocess
from concurrent import futures
from pathlib import Path
from time import perf_counter, time

import fastfilters as ff
import h5py
import imageio.v3 as iio
import numpy
from sklearn.ensemble import RandomForestClassifier as ScikitForest
from sklearn.tree import export_graphviz
"""
Benchmarking script to compare inference on CPU to inference on GPU (in Rust).
`working_dir` must contain ilastik features and labels export at `features_path` and `labels_path`.

Trains a random forest, exports it to `working_dir/benchmark_trees`, and provides benchmark for 
running inference with this forest on a random image in Python on CPU vs. using gpu_filters.

To be run from a Python interpreter with an ilastik environment.
"""

working_dir = Path(__file__).parent  # ./bench
generated_raw_path = working_dir / "out" / "raw.tif"
cpu_seg_output_path = working_dir / "out" / "segmentation_cpu.png"
gpu_seg_output_path = working_dir / "out" / "segmentation_gpu.png"
gpu_filters_bin = working_dir.parent / "target" / "release" / "gpu_filters"
max_workers = 16
seed = 1337


def get_features(img):
    filters = [0.3, 0.7, 0.9, 1.0, 1.6, 3.5, 4.0, 5.0, 7.0, 10.0]
    futs = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, sigma in enumerate(filters):
            for ch in range(4):
                futs.append(pool.submit(ff.gaussianSmoothing, img[..., ch], window_size=3.5, sigma=sigma))
    feature_img = numpy.array([fut.result() for fut in futs])  # (c, y, x)
    features_for_classifier = feature_img.transpose(1, 2, 0).reshape(-1, feature_img.shape[0])  # (y*x, c)
    return features_for_classifier


def create_training_data():
    """
    This was done once manually, code here is for the record.
    """
    shape = (2048, 2048, 4)
    img = numpy.random.randint(0, 255, shape)
    iio.imwrite(working_dir / "randint2048x2048x4.tif", img)
    # Resulting tif file loaded in ilastik pixel classification.
    # Features: Gaussian smoothing with sigmas 0.3, 0.7, 0.9, 1.0, 1.6, 3.5, 4.0, 5.0, 7.0, 10.0
    # (0.9, 4.0, 7.0 added manually as these are not included by default in ilastik)
    # Labelled a few dark pixels as class 1 and a few bright pixels as class 2
    # Exported Features and Labels.
    labels_path = working_dir / "randint2048x2048x4_Labels.h5"
    features_path = working_dir / "randint2048x2048x4_Features.h5"
    h5subpath = "exported_data"
    
    fl = h5py.File(labels_path)
    labels = numpy.array(fl[h5subpath])  # shape: 2048, 2048, 1
    fl.close()
    label_coords = labels.nonzero()
    label_values = labels[label_coords]
    ff = h5py.File(features_path)
    features = numpy.array(ff[h5subpath])  # shape: 2048, 2048, 40
    ff.close()
    y, x, _ = label_coords
    feature_values = features[y, x, :]
    numpy.save(working_dir/"features.npy", feature_values)
    numpy.save(working_dir/"labels.npy", label_values)


def get_training_data():
    f = numpy.load(working_dir / "features.npy")
    l = numpy.load(working_dir / "labels.npy")
    return f, l


def train_classifier(training_data):
    features, labels = training_data
    clf = ScikitForest(random_state=seed, n_estimators=100)
    clf.fit(features, labels)
    return clf


def export_classifier(classifier, class_names):
    tree_dir = working_dir / "out" / "benchmark_trees"
    shutil.rmtree(tree_dir, ignore_errors=True)
    tree_dir.mkdir(parents=True, exist_ok=True)
    for (tree_idx, tree) in enumerate(classifier.estimators_):
        _ = export_graphviz(
            decision_tree=tree,
            out_file=str(tree_dir / f"tree_{tree_idx}.txt"),
            impurity=False,
            node_ids=True,
            precision=17,
            class_names=class_names,
        )


def setup_target_image():
    shape = (2048, 2048, 4)
    numpy.random.seed(seed)
    img = numpy.random.randint(0, 255, shape).astype(numpy.uint8)
    numpy.random.seed(int(time()))
    iio.imwrite(generated_raw_path, img)


def run_on_cpu(classifier):
    # The steps that the rust binary runs (be it on cpu or gpu) all need to be timed
    # I.e.: Load image, comput features, predict, write segmentation
    start = perf_counter()
    img = iio.imread(generated_raw_path)
    features = get_features(img)
    prediction = classifier.predict(features)
    iio.imwrite(cpu_seg_output_path, prediction.reshape(img.shape[0], img.shape[1]))
    end = perf_counter()
    return end - start


def run_on_gpu():
    # gpu_filters loads raw.png, 
    # computes features from it,
    # parses the exported classifier from benchmark_trees,
    # runs inference on the image,
    # and writes segmentation.
    # Here, we just have to run gpu_filters_bin and time the execution.
    start = perf_counter()
    subprocess.run([str(gpu_filters_bin)], check=True)
    end = perf_counter()
    return end - start


if __name__ == "__main__":
    training_data = get_training_data()
    classifier = train_classifier(training_data)
    num_classes = len(numpy.unique(training_data[1]))
    export_classifier(classifier, [str(idx) for idx in range(num_classes)])
    setup_target_image()  # Generates an image and writes it on disk. Both cpu and gpu side load it.
    cpu_time = run_on_cpu(classifier)
    gpu_time = run_on_gpu()
    print(f"CPU time: {cpu_time:.2f} seconds")
    print(f"GPU time: {gpu_time:.2f} seconds")
