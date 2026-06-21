from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from urllib.request import urlretrieve

import kagglehub
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from tfrecord.reader import tfrecord_loader


N_PER_DIGIT = 250
IMG_SIZE = 28
RNG = np.random.default_rng(0)

LANGUAGES = [
    "English",
    "Mandarin Chinese",
    "Hindi",
    "Arabic",
    "Bengali",
    "Urdu/Persian",
    "Telugu",
]

SOURCES = {
    "English": "OpenML MNIST 784",
    "Mandarin Chinese": "Kaggle gpreda/chinese-mnist",
    "Hindi": "Kaggle anurags397/hindi-mnist-data",
    "Arabic": "Kaggle mloey1/ahdd1",
    "Bengali": "Kaggle wchowdhu/bengali-digits",
    "Urdu/Persian": "Kaggle teerathkumar142/urdumnist",
    "Telugu": "CMATERdb Telugu numerals",
}

GLYPHS = {
    "English": list("0123456789"),
    "Mandarin Chinese": list("零一二三四五六七八九"),
    "Hindi": list("०१२३४५६७८९"),
    "Arabic": list("٠١٢٣٤٥٦٧٨٩"),
    "Bengali": list("০১২৩৪৫৬৭৮৯"),
    "Urdu/Persian": list("۰۱۲۳۴۵۶۷۸۹"),
    "Telugu": list("౦౧౨౩౪౫౬౭౮౯"),
}

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def border_mean(images: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            images[:, 0, :],
            images[:, -1, :],
            images[:, :, 0],
            images[:, :, -1],
        ],
        axis=1,
    ).mean(axis=1)


def remove_background_artifacts(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    mid_gray = (image >= 0.18) & (image <= 0.82)
    labels, count = ndimage.label(mid_gray)

    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < 18:
            continue

        ys, xs = np.where(component)
        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        fill = area / (width * height)
        touches_border = xs.min() == 0 or ys.min() == 0 or xs.max() == IMG_SIZE - 1 or ys.max() == IMG_SIZE - 1
        broad_patch = area >= 36 and fill >= 0.45 and image[component].mean() < 0.78
        mostly_background = area >= IMG_SIZE * IMG_SIZE * 0.35 and image[component].mean() < 0.85

        if touches_border and (broad_patch or mostly_background):
            image[component] = 0

    image[image < 0.04] = 0
    return image


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    if x.max() > 1:
        x /= 255

    # Put every dataset into the same convention: bright digit on dark background.
    images = x.reshape(-1, IMG_SIZE, IMG_SIZE)
    bright_background = border_mean(images) > 0.5
    images[bright_background] = 1 - images[bright_background]

    x = images.reshape(len(images), -1)
    maxes = x.max(axis=1, keepdims=True)
    x = np.divide(x, maxes, out=np.zeros_like(x), where=maxes > 0)

    images = np.array([remove_background_artifacts(image) for image in x.reshape(-1, IMG_SIZE, IMG_SIZE)], dtype="float32")

    x = images.reshape(len(images), -1)
    maxes = x.max(axis=1, keepdims=True)
    return np.divide(x, maxes, out=np.zeros_like(x), where=maxes > 0)


def balance(x: np.ndarray, y: np.ndarray, n: int = N_PER_DIGIT) -> tuple[np.ndarray, np.ndarray]:
    keep = []
    for digit in range(10):
        candidates = np.flatnonzero(y == digit)
        if len(candidates) < n:
            raise ValueError(f"Digit {digit} only has {len(candidates)} samples; need {n}.")
        keep.append(RNG.choice(candidates, n, replace=False))
    keep = np.concatenate(keep)
    return normalize_rows(x[keep]), y[keep].astype(int)


def image_to_row(path_or_image: Path | Image.Image) -> np.ndarray:
    image = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    arr = np.asarray(image.convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype="float32").reshape(1, -1)
    return normalize_rows(arr)[0]


def usable_row(row: np.ndarray) -> bool:
    image = row.reshape(IMG_SIZE, IMG_SIZE)
    return int((image > 0.35).sum()) >= 8 and int((image > 0.04).sum()) >= 12


def images_to_rows(images: np.ndarray) -> np.ndarray:
    rows = []
    for image in images:
        if image.ndim == 3:
            image = image.mean(axis=2)
        rows.append(image_to_row(Image.fromarray(image.astype("uint8"))))
    return np.array(rows, dtype="float32")


def load_image_folders(base: Path, pattern: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for digit in range(10):
        paths = np.array(sorted((base / str(digit)).glob(pattern)), dtype=object)
        rows = [image_to_row(Path(path)) for path in paths]
        usable = np.array([row for row in rows if usable_row(row)], dtype="float32")
        if len(usable) < N_PER_DIGIT:
            raise ValueError(f"{base / str(digit)} only has {len(usable)} usable files after cleanup.")
        for row in usable[RNG.choice(len(usable), N_PER_DIGIT, replace=False)]:
            x.append(row)
            y.append(digit)
    return np.array(x, dtype="float32"), np.array(y, dtype=int)


def load_english() -> tuple[np.ndarray, np.ndarray]:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    return balance(mnist.data, mnist.target.astype(int))


def load_chinese() -> tuple[np.ndarray, np.ndarray]:
    path = Path(kagglehub.dataset_download("gpreda/chinese-mnist"))
    rows = np.loadtxt(path / "chinese_mnist.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3), dtype=int)
    value_by_key = {s * 1_000_000 + sample * 1_000 + code: value for s, sample, code, value in rows}

    x, y = [], []
    for record in tfrecord_loader(str(path / "chinese_mnist.tfrecords"), None, None):
        value = value_by_key[int(record["label"][0])]
        if value <= 9:
            image = np.frombuffer(record["image_raw"], dtype=np.uint8).reshape(64, 64)
            x.append(image_to_row(Image.fromarray(image)))
            y.append(value)
    return balance(np.array(x), np.array(y))


def load_hindi() -> tuple[np.ndarray, np.ndarray]:
    path = Path(kagglehub.dataset_download("anurags397/hindi-mnist-data"))
    base = path / "DevanagariHandwrittenDigitDataset" / "DevanagariHandwrittenDigitDataset" / "Train"
    x, y = [], []
    for digit in range(10):
        paths = np.array(sorted((base / f"digit_{digit}").glob("*.png")), dtype=object)
        rows = np.array([image_to_row(Path(path)) for path in paths], dtype="float32")
        usable = rows[[usable_row(row) for row in rows]]
        if len(usable) < N_PER_DIGIT:
            raise ValueError(f"{base / f'digit_{digit}'} only has {len(usable)} usable files after cleanup.")
        for row in usable[RNG.choice(len(usable), N_PER_DIGIT, replace=False)]:
            x.append(row)
            y.append(digit)
    return np.array(x, dtype="float32"), np.array(y, dtype=int)


def load_arabic() -> tuple[np.ndarray, np.ndarray]:
    path = Path(kagglehub.dataset_download("mloey1/ahdd1")) / "Arabic Handwritten Digits Dataset CSV"
    x = np.loadtxt(path / "csvTrainImages 60k x 784.csv", delimiter=",", max_rows=N_PER_DIGIT * 10, dtype="float32")
    y = np.loadtxt(path / "csvTrainLabel 60k x 1.csv", delimiter=",", max_rows=N_PER_DIGIT * 10, dtype=int).reshape(-1)
    return balance(x, y)


def load_bengali() -> tuple[np.ndarray, np.ndarray]:
    path = Path(kagglehub.dataset_download("wchowdhu/bengali-digits")) / "bengali_digits"
    return load_image_folders(path, "*.jpg")


def load_urdu() -> tuple[np.ndarray, np.ndarray]:
    path = Path(kagglehub.dataset_download("teerathkumar142/urdumnist")) / "UrduDataset" / "x_train" / "x_train"
    return load_image_folders(path, "*.jpg")


def load_telugu() -> tuple[np.ndarray, np.ndarray]:
    path = Path("data_sources/cmaterdb_telugu/telugu-training-images.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(
            "https://raw.githubusercontent.com/prabhuomkar/CMATERdb/master/datasets/telugu-numerals/training-images.npz",
            path,
        )
    archive = np.load(path)
    return balance(images_to_rows(archive["images"]), archive["labels"].astype(int))


LOADERS = {
    "English": load_english,
    "Mandarin Chinese": load_chinese,
    "Hindi": load_hindi,
    "Arabic": load_arabic,
    "Bengali": load_bengali,
    "Urdu/Persian": load_urdu,
    "Telugu": load_telugu,
}


def row_to_png_base64(row: np.ndarray) -> str:
    image = Image.fromarray(np.clip(row.reshape(IMG_SIZE, IMG_SIZE) * 255, 0, 255).astype("uint8"), mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def nearest_neighbors(z: np.ndarray) -> list[list[list[float | int]]]:
    distances = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    nearest = []
    for i in range(len(z)):
        idx = np.argpartition(distances[i], 6)[:6]
        idx = idx[idx != i]
        idx = idx[np.argsort(distances[i, idx])][:5]
        nearest.append([[int(j), round(float(distances[i, j] ** 0.5), 4)] for j in idx])
    return nearest


def language_id(language: str) -> str:
    return language.lower().replace("/", "-").replace(" ", "-")


def build_payload() -> dict:
    languages = []
    for language in LANGUAGES:
        print(f"Loading {language}...")
        x, y = LOADERS[language]()
        z = PCA(n_components=2, random_state=0).fit_transform(x)
        neighbors = nearest_neighbors(z)

        points = []
        for i, (coords, digit, row) in enumerate(zip(z, y, x)):
            points.append(
                [
                    round(float(coords[0]), 5),
                    round(float(coords[1]), 5),
                    int(digit),
                    row_to_png_base64(row),
                    neighbors[i],
                ]
            )

        languages.append(
            {
                "id": language_id(language),
                "name": language,
                "source": SOURCES[language],
                "glyphs": GLYPHS[language],
                "points": points,
            }
        )
        print(f"  {len(points)} samples")

    return {"colors": COLORS, "languages": languages}


def main() -> None:
    payload = build_payload()
    Path("interactive_digit_pca_data.js").write_text(
        "window.DIGIT_PCA_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    print("Wrote interactive_digit_pca_data.js")


if __name__ == "__main__":
    main()
