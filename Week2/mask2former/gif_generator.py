import os
import imageio.v2 as imageio
from PIL import Image

def create_gif(image_folder, output_gif, duration=0.5, max_width=800):
    """
    Reads images from a folder, resizes them, and creates a compressed GIF.
    
    :param image_folder: Path to the folder containing images.
    :param output_gif: Name of the output GIF file.
    :param duration: Duration for each frame in seconds.
    :param max_width: Maximum width for resizing while maintaining aspect ratio.
    """
    images = []
    file_list = sorted(os.listdir(image_folder))  # Sort files by name
    
    for file_name in file_list:
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            file_path = os.path.join(image_folder, file_name)
            img = Image.open(file_path)

            # Resize if the image is too large
            if img.width > max_width:
                aspect_ratio = img.height / img.width
                new_height = int(max_width * aspect_ratio)
                img = img.resize((max_width, new_height), Image.LANCZOS)

            images.append(img.convert("P", palette=Image.ADAPTIVE))  # Reduce colors for compression

    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=True)
        print(f"GIF saved as {output_gif}")
    else:
        print("No images found in the specified folder.")

# Example usage
if __name__ == "__main__":
    create_gif("./finetune_eval/019/", "19eval_finetune_mask2former.gif", duration=0.5)