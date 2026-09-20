import json
import os
from PIL import Image, ImageDraw

SPLIT_FILE = "dataset_splits_clean.json"
OUTPUT = "dhash_suspicious_groups.jpg"

CLASSES = [
    "Bacterial_Spot",
    "Cercospora_Leaf_Spot",
    "Curl_Virus",
    "Healthy_Leaf",
    "Nutrition_Deficiency",
    "Powdery_Mildew"
]

def dhash(path):
    img = Image.open(path).convert("L")
    img = img.resize((9, 8))

    pixels = list(img.getdata())

    bits = []

    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits.append("1" if left > right else "0")

    return hex(int("".join(bits), 2))[2:].zfill(16)


with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    splits = json.load(f)

hashes = {}

for split in ["train", "val", "test"]:

    for item in splits[split]:

        path = item["path"]

        try:
            h = dhash(path)
        except:
            continue

        hashes.setdefault(h, []).append(
            (split, path)
        )


groups = []

for h, items in hashes.items():

    split_names = set(x[0] for x in items)

    if len(split_names) > 1:
        groups.append((h, items))


print("Suspicious cross-split dHash groups:", len(groups))

# Create contact sheet
thumb_w = 220
thumb_h = 250
cols = 4

images = []

for h, items in groups:

    print("\nHASH:", h)

    # Maximum 4 images per group
    for split, path in items[:4]:

        print(split, path)

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, 190))

            canvas = Image.new(
                "RGB",
                (thumb_w, thumb_h),
                "white"
            )

            x = (thumb_w - img.width) // 2
            canvas.paste(img, (x, 5))

            draw = ImageDraw.Draw(canvas)

            filename = os.path.basename(path)

            text = f"{split}\n{filename}"

            draw.text(
                (5, 200),
                text,
                fill="black"
            )

            images.append(canvas)

        except Exception as e:
            print("Error:", e)


rows = (len(images) + cols - 1) // cols

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * thumb_h),
    "white"
)

for i, img in enumerate(images):

    x = (i % cols) * thumb_w
    y = (i // cols) * thumb_h

    sheet.paste(img, (x, y))


sheet.save(OUTPUT, quality=95)

print("\nSaved:")
print(os.path.abspath(OUTPUT))