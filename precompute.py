
from tools import utils
from tools.inception_v3_imagenet import model
import tensorflow as tf
import numpy as np

import pickle
import sys
import os

# Disable eager execution (required for TF1-style code)
tf.compat.v1.disable_eager_execution()

# Set the path to your ImageNet-like dataset
IMAGENET_PATH = "archive/tiny-imagenet-200/tiny-imagenet-200"

if __name__ == "__main__":
    if IMAGENET_PATH == "":
        raise ValueError("Please open precompute.py and set IMAGENET_PATH")
    if not os.path.exists(IMAGENET_PATH):
        raise FileNotFoundError(f"ImageNet path does not exist: {IMAGENET_PATH}")

    dataset = sys.argv[1] if len(sys.argv) > 1 else "imagenet"

    # Create a TensorFlow session
    sess = tf.compat.v1.InteractiveSession()
    print("[INFO] TF1-style session created.")

    # Define input placeholder for a batch of images
    x = tf.compat.v1.placeholder(tf.float32, [None, 299, 299, 3])
    print("[INFO] Input placeholder created.")

    # Get predictions from model
    _, preds = model(sess, x)

    print("[INFO] Model initialized and prediction tensor built.")

    label_dict = {}
    last_j = 0

    # Go through the first 1000 ImageNet labels
    for i in range(1, 1000):
        print(f"[INFO] Looking for label {i}...")
        if i in label_dict:
            continue

        # Search through 50,000 images to find one with the correct label
        for j in range(last_j, 10000):
            try:
                im, lab = utils.get_image(j, IMAGENET_PATH)
                print(f"[DEBUG] Image index {j} shape: {im.shape}, label: {lab}")

                # Ensure image is of shape (1, 299, 299, 3)
                if im.ndim == 3:
                    im = np.expand_dims(im, axis=0)

                pred = sess.run(preds, {x: im})[0]

                if pred == lab:
                    label_dict[lab] = j
                    print(f"[INFO] Found matching label: {lab} at index {j}")
                if lab == i:
                    label_dict[i] = j
                    last_j = j
                    break
            except Exception as e:
                print(f"[WARNING] Error processing image index {j}: {e}")
                continue

    # Save to file
    os.makedirs("tools/data", exist_ok=True)
    with open("tools/data/imagenet.pickle", "wb") as f:
        pickle.dump(label_dict, f)
        print("[INFO] Saved label_dict to tools/data/imagenet.pickle")
