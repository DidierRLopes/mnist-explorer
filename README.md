# MNIST-Style Digit PCA Explorer

Interactive PCA visualizations for real MNIST-style handwritten digit datasets across multiple numeral systems. Short demo [here](https://www.youtube.com/watch?v=yKWpzT3czzQ).

<img width="3446" height="1734" alt="CleanShot 2026-06-22 at 09 13 20@2x" src="https://github.com/user-attachments/assets/e78691f4-d610-425e-bf68-d6c38fad1a6f" />

## What is included

- `mnist_pca_2d.ipynb` loads the datasets and shows 2D PCA views in a notebook.
- `interactive_digit_pca.html` is a standalone browser explorer.
- `interactive_digit_pca_data.js` contains the precomputed PCA coordinates, sample images, and nearest-neighbor data used by the HTML page.
- `generate_interactive_digit_pca_data.py` rebuilds the interactive data payload from the source datasets.

## Datasets

The explorer uses real MNIST-style handwritten digit datasets:

- English: OpenML MNIST 784
- Mandarin Chinese: Kaggle `gpreda/chinese-mnist`
- Hindi: Kaggle `anurags397/hindi-mnist-data`
- Arabic: Kaggle `mloey1/ahdd1`
- Bengali: Kaggle `wchowdhu/bengali-digits`
- Urdu/Persian: Kaggle `teerathkumar142/urdumnist`
- Telugu: CMATERdb Telugu numerals

Each language is sampled to 250 examples per digit, for 2,500 plotted samples per language.

## Use the explorer

Clone the repo and serve the folder locally:

```bash
git clone git@github.com:DidierRLopes/mnist-explorer.git
cd mnist-explorer
python3 -m http.server 8765 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8765/interactive_digit_pca.html
```

The page lets you select a language, inspect the 2D PCA plot, click samples, and page through nearest neighbors. On mobile, sample details open as a modal.

You can also open the HTML file directly if your browser allows local JavaScript files:

```text
interactive_digit_pca.html
```

## Rebuild the data

Install the generator dependencies:

```bash
python -m pip install numpy pillow scipy scikit-learn pandas kagglehub tfrecord
```

Then regenerate:

```bash
python generate_interactive_digit_pca_data.py
```

The generator normalizes all datasets to bright digits on dark backgrounds, removes large gray background artifacts from image-folder datasets, filters unusable samples, recomputes PCA, and writes `interactive_digit_pca_data.js`.

After regenerating the data, refresh `interactive_digit_pca.html` in the browser.

## Notes

Some source datasets contain JPEG artifacts or gray background blocks. The generator includes a cleanup step so the interactive viewer shows cleaner digit samples while preserving the handwritten strokes.
