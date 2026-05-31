from PIL import Image
import numpy as np
from collections import Counter
import heapq
import matplotlib.pyplot as plt
import math
import time
import os
import csv

class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_map):
    heap = [Node(sym, freq) for sym, freq in freq_map.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        only = heapq.heappop(heap)
        root = Node(None, only.freq)
        root.left = only
        return root

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)

        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2

        heapq.heappush(heap, merged)

    return heap[0]

def build_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}

    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix if prefix != "" else "0"

        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)

    return codebook

def huffman_encode(values):
    freq_map = Counter(values)
    tree = build_huffman_tree(freq_map)
    codes = build_codes(tree)
    encoded_data = ''.join(codes[value] for value in values)

    return encoded_data, codes, freq_map

def huffman_decode(encoded_data, codes):
    reverse_codes = {v: k for k, v in codes.items()}

    decoded = []
    current_code = ""

    for bit in encoded_data:
        current_code += bit

        if current_code in reverse_codes:
            decoded.append(reverse_codes[current_code])
            current_code = ""

    return decoded

def rle_encode(values):
    encoded = []

    if len(values) == 0:
        return encoded

    current_value = values[0]
    count = 1

    for value in values[1:]:
        if value == current_value:
            count += 1
        else:
            encoded.append((current_value, count))
            current_value = value
            count = 1

    encoded.append((current_value, count))

    return encoded

def rle_decode(encoded):
    decoded = []

    for value, count in encoded:
        decoded.extend([value] * count)

    return decoded

def calculate_rle_bits(encoded):
    return len(encoded) * 24

def calculate_mse(original, restored):
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    return np.mean((original - restored) ** 2)

def calculate_psnr(original, restored):
    mse = calculate_mse(original, restored)

    if mse == 0:
        return float("inf")

    return 10 * math.log10((255 ** 2) / mse)

def entropy(values):
    freq = Counter(values)
    total = len(values)
    result = 0

    for count in freq.values():
        p = count / total
        result -= p * math.log2(p)

    return result

def normalized_autocorrelation(signal):
    signal = signal.astype(np.float64)

    n = len(signal)
    mean = np.mean(signal)
    var = np.var(signal)

    autocorr = []

    for k in range(n // 2):
        numerator = np.sum((signal[:n - k] - mean) * (signal[k:] - mean))
        denominator = (n - k) * var

        autocorr.append(numerator / denominator if denominator != 0 else 0)

    return np.array(autocorr)

def reconstruct_image(diff, first_column, shape):
    rows, cols = shape

    restored = np.zeros((rows, cols), dtype=np.int16)
    restored[:, 0] = first_column

    for j in range(1, cols):
        restored[:, j] = restored[:, j - 1] + diff[:, j - 1]

    return np.clip(restored, 0, 255).astype(np.uint8)

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def analyze_single_image(image_path, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    img = Image.open(image_path).convert("L")
    data = np.array(img, dtype=np.uint8)

    rows, cols = data.shape
    original_bits = data.size * 8

    pixels_flat = data.flatten().tolist()

    encoded_pixels, pixel_codes, pixel_freqs = huffman_encode(pixels_flat)
    decoded_pixels = huffman_decode(encoded_pixels, pixel_codes)

    restored_huffman = np.array(decoded_pixels, dtype=np.uint8).reshape(rows, cols)

    huffman_bits = len(encoded_pixels)
    huffman_ratio = original_bits / huffman_bits
    huffman_redundancy = (1 - huffman_bits / original_bits) * 100
    huffman_psnr = calculate_psnr(data, restored_huffman)
    pixel_entropy = entropy(pixels_flat)

    rle_encoded = rle_encode(pixels_flat)
    rle_decoded = rle_decode(rle_encoded)

    restored_rle = np.array(rle_decoded, dtype=np.uint8).reshape(rows, cols)

    rle_bits = calculate_rle_bits(rle_encoded)
    rle_ratio = original_bits / rle_bits
    rle_redundancy = (1 - rle_bits / original_bits) * 100
    rle_psnr = calculate_psnr(data, restored_rle)

    data_int = data.astype(np.int16)

    diff = data_int[:, 1:] - data_int[:, :-1]
    diff_flat = diff.flatten().tolist()

    encoded_diff, diff_codes, diff_freqs = huffman_encode(diff_flat)
    decoded_diff = huffman_decode(encoded_diff, diff_codes)

    decoded_diff = np.array(decoded_diff, dtype=np.int16).reshape(rows, cols - 1)
    restored_diff = reconstruct_image(decoded_diff, data_int[:, 0], (rows, cols))

    first_column_bits = rows * 8
    diff_huffman_bits = len(encoded_diff) + first_column_bits

    diff_huffman_ratio = original_bits / diff_huffman_bits
    diff_huffman_redundancy = (1 - diff_huffman_bits / original_bits) * 100
    diff_huffman_psnr = calculate_psnr(data, restored_diff)
    diff_entropy = entropy(diff_flat)

    plt.figure(figsize=(6, 5))
    plt.title(f"Оригінальне зображення: {image_name}")
    plt.imshow(data, cmap="gray")
    plt.axis("off")
    save_plot(os.path.join(image_output_dir, "01_original_image.png"))

    plt.figure(figsize=(6, 5))
    plt.title(f"Різницеве зображення: {image_name}")
    plt.imshow(diff, cmap="gray")
    plt.axis("off")
    save_plot(os.path.join(image_output_dir, "02_difference_image.png"))

    plt.figure(figsize=(10, 4))
    plt.title(f"Гістограма яскравостей: {image_name}")
    plt.xlabel("Значення яскравості")
    plt.ylabel("Частота")
    plt.bar(pixel_freqs.keys(), pixel_freqs.values(), color="gray")
    plt.grid(True)
    save_plot(os.path.join(image_output_dir, "03_pixel_histogram.png"))

    plt.figure(figsize=(10, 4))
    plt.title(f"Гістограма різниць: {image_name}")
    plt.xlabel("Значення різниці")
    plt.ylabel("Частота")
    plt.bar(diff_freqs.keys(), diff_freqs.values(), color="gray")
    plt.grid(True)
    save_plot(os.path.join(image_output_dir, "04_difference_histogram.png"))

    sample_row = data[rows // 2]
    autocorr = normalized_autocorrelation(sample_row)

    plt.figure(figsize=(8, 4))
    plt.title(f"Нормалізована автокореляція: {image_name}")
    plt.xlabel("Зсув")
    plt.ylabel("Коефіцієнт кореляції")
    plt.plot(autocorr)
    plt.grid(True)
    save_plot(os.path.join(image_output_dir, "05_autocorrelation.png"))

    methods = ["Хаффман", "RLE", "Різниця + Хаффман"]
    ratios = [huffman_ratio, rle_ratio, diff_huffman_ratio]

    plt.figure(figsize=(8, 4))
    plt.title(f"Порівняння коефіцієнтів стиснення: {image_name}")
    plt.ylabel("Коефіцієнт стиснення")
    plt.bar(methods, ratios, color="gray")
    plt.grid(axis="y")
    save_plot(os.path.join(image_output_dir, "06_compression_ratio_comparison.png"))

    fig, axs = plt.subplots(1, 4, figsize=(16, 5))

    axs[0].imshow(data, cmap="gray")
    axs[0].set_title("Оригінал")
    axs[0].axis("off")

    axs[1].imshow(restored_huffman, cmap="gray")
    axs[1].set_title("Хаффман")
    axs[1].axis("off")

    axs[2].imshow(restored_rle, cmap="gray")
    axs[2].set_title("RLE")
    axs[2].axis("off")

    axs[3].imshow(restored_diff, cmap="gray")
    axs[3].set_title("Різниця + Хаффман")
    axs[3].axis("off")

    save_plot(os.path.join(image_output_dir, "07_restored_images_comparison.png"))

    results = [
        [
            image_name,
            "Хаффман",
            original_bits,
            huffman_bits,
            round(huffman_ratio, 3),
            round(huffman_redundancy, 2),
            "inf" if huffman_psnr == float("inf") else round(huffman_psnr, 2),
            round(pixel_entropy, 4)
        ],
        [
            image_name,
            "RLE",
            original_bits,
            rle_bits,
            round(rle_ratio, 3),
            round(rle_redundancy, 2),
            "inf" if rle_psnr == float("inf") else round(rle_psnr, 2),
            round(pixel_entropy, 4)
        ],
        [
            image_name,
            "Різницеве перетворення + Хаффман",
            original_bits,
            diff_huffman_bits,
            round(diff_huffman_ratio, 3),
            round(diff_huffman_redundancy, 2),
            "inf" if diff_huffman_psnr == float("inf") else round(diff_huffman_psnr, 2),
            round(diff_entropy, 4)
        ]
    ]

    return results

def save_all_results_to_csv(results, filename):
    with open(filename, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Зображення",
            "Метод",
            "Оригінальний розмір, біт",
            "Стиснутий розмір, біт",
            "Коефіцієнт стиснення",
            "Відносна надлишковість, %",
            "PSNR, дБ",
            "Ентропія, біт/символ"
        ])

        writer.writerows(results)

def analyze_folder(input_dir="input_images", output_dir="output_images"):
    start_time = time.time()

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    allowed_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    image_files = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.lower().endswith(allowed_extensions)
    ]

    if not image_files:
        print("У папці input_images немає зображень.")
        print("Додай туди файли .png, .jpg, .jpeg або .bmp і запусти програму ще раз.")
        return

    all_results = []

    for image_path in image_files:
        print(f"Обробка зображення: {image_path}")
        image_results = analyze_single_image(image_path, output_dir)
        all_results.extend(image_results)

    csv_path = os.path.join(output_dir, "all_compression_results.csv")
    save_all_results_to_csv(all_results, csv_path)

    print()
    print("=== Обробку завершено ===")
    print(f"Кількість оброблених зображень: {len(image_files)}")
    print(f"Загальна таблиця результатів: {csv_path}")
    print(f"Папка з усіма графіками: {output_dir}")
    print(f"Час виконання: {time.time() - start_time:.4f} с")

if __name__ == "__main__":
    analyze_folder("input_images", "output_images")