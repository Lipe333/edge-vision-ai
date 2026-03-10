import cv2
import os

images_dir = 'VisDrone2019-DET-val/VisDrone2019-DET-val/images/'
annotations_dir = 'VisDrone2019-DET-val/VisDrone2019-DET-val/annotations/'
SAVING_PATH = "croppedImages/val"

os.makedirs(SAVING_PATH, exist_ok=True)

imageFiles = os.listdir(images_dir)

if not imageFiles:
    raise ValueError(f"Erro ao abrir {images_dir}")

for imfile in imageFiles:

    image_path = os.path.join(images_dir, imfile)
    
    annotation_path = os.path.join(
        annotations_dir,
        imfile.replace(".jpg", ".txt")
    )

    img = cv2.imread(image_path)

    if img is None:
        print(f"Warning: Failed to read image {imfile}. Skipping.")
        continue

    if not os.path.exists(annotation_path):
        print(f"Annotation not found for {imfile}. Skipping.")
        continue

    with open(annotation_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):

        parts = line.strip().split(',')

        if len(parts) < 4:
            continue

        x = int(float(parts[0]))
        y = int(float(parts[1]))
        w = int(float(parts[2]))
        h = int(float(parts[3]))
        score = int(parts[4])
        class_id = int(parts[5])

        # ignora classe 0 e com score 0
        if class_id == 0 or score == 0:
            continue

        x_end = min(x + w, img.shape[1])
        y_end = min(y + h, img.shape[0])

        cropped = img[y:y_end, x:x_end]

        if cropped.size == 0:
            continue

        # Criar pasta da classe
        class_folder = os.path.join(SAVING_PATH, str(class_id))
        os.makedirs(class_folder, exist_ok=True)

        save_name = f"{imfile.replace('.jpg','')}_{idx}.jpg"
        save_path = os.path.join(class_folder, save_name)

        cv2.imwrite(save_path, cropped)

print("Finished cropping!")
    


# first_path = os.path.join(images_dir, imageFiles[0])
# img = cv2.imread(first_path)
# if img is None:
#     raise ValueError(f"Failed to read image: {first_path}")

# plt.figure(figsize=(10, 8))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# Example loop over all images (uncomment to use):
# for fname in imageFiles:
#     full_path = os.path.join(images_dir, fname)
#     img = cv2.imread(full_path)
#     # process each image here
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show() 
