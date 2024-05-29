from PIL import Image

def is_skin_color(rgb):
    lower_bound = (45, 52, 90)
    upper_bound = (255, 255, 255)
    
    r, g, b = rgb
    return (lower_bound[0] <= r <= upper_bound[0] and
            lower_bound[1] <= g <= upper_bound[1] and
            lower_bound[2] <= b <= upper_bound[2])

def skin_color_thresholding(image):
    width, height = image.size
    pixels = image.load()

    mask = Image.new("L", (width, height))
    mask_pixels = mask.load()

    result = Image.new("RGB", (width, height))
    result_pixels = result.load()

    for y in range(height):
        for x in range(width):
            if is_skin_color(pixels[x, y]):
                mask_pixels[x, y] = 255
                result_pixels[x, y] = pixels[x, y]
            else:
                mask_pixels[x, y] = 0
                result_pixels[x, y] = (0, 0, 0)

    return result, mask

def save_images(original, mask, result, base_filename):
    original.save(f"{base_filename}_original.png")
    mask.save(f"{base_filename}_mask.png")
    result.save(f"{base_filename}_result.png")

def process_single_image(image_path):
    try:
        image = Image.open(image_path)
        result, mask = skin_color_thresholding(image)
        base_filename = image_path.split('.')[0]
        save_images(image, mask, result, base_filename)
        print(f"Processed {image_path}, saved results as {base_filename}_original.png, {base_filename}_mask.png, {base_filename}_result.png")
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")

chosen_image_index = 3
ImageList = ["face1.jpg", "face2.jpg", "face3.jpg", "face4.jpg"]

process_single_image(ImageList[chosen_image_index])
