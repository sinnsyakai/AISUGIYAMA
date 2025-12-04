from PIL import Image

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

try:
    img = Image.open("assets/icon.png")
    # Crop to square first
    img = crop_max_square(img)
    # Resize to zoom in (effectively cropping the center 70%)
    width, height = img.size
    new_width = int(width * 0.7)
    new_height = int(height * 0.7)
    img = crop_center(img, new_width, new_height)
    
    img.save("assets/icon_zoomed.png")
    print("Successfully created assets/icon_zoomed.png")
except Exception as e:
    print(f"Error processing image: {e}")
