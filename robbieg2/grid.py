from PIL import Image, ImageDraw, ImageFont

# We need a simple grid: numbers from 1 to 9 in points on an intersection of nxn grid.
# The font size may be 1/5 of the size of the height of the cell.
# Therefore, we need the size of the image and colors, and the file_name. 

def create_grid_image(image_width, image_height, color_circle, color_number, n, file_name):
    cell_width = image_width // n
    cell_height = image_height // n
    font_size = max(cell_height // 5, 20)
    circle_radius = font_size * 7 // 10

    # Create a blank image with transparent background
    img = Image.new('RGBA', (image_width, image_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load a font
    font = ImageFont.truetype("font/arialbd.ttf", font_size)

    # Set the number of cells in each dimension
    num_cells_x = n - 1 
    num_cells_y = n - 1

    # Draw the numbers in the center of each cell
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            number = i * num_cells_y + j + 1
            text = str(number)
            x = (i + 1) * cell_width
            y = (j + 1) * cell_height
            draw.ellipse([x - circle_radius, y - circle_radius, 
                          x + circle_radius, y + circle_radius], 
                          fill=color_circle)
            offset_x = font_size / 4 if number < 10 else font_size / 2
            draw.text((x - offset_x, y - font_size / 2), text, font=font, fill=color_number)

    # Save the image
    img.save(file_name)

def zoom_in(image_path, n, index, upscale):
    img = Image.open(image_path)
    width, height = img.size
    # we need to calculate the cell size
    cell_width = width // n
    cell_height = height // n
    # we need to calculate the x and y coordinates of the cell
    x = ((index - 1) // (n - 1)) * cell_width
    y = ((index - 1) % (n - 1)) * cell_height
    # we need to calculate the x and y coordinates of the top left corner of the cell
    top_left = (x, y)
    # we need to calculate the x and y coordinates of the bottom right corner of the cell
    bottom_right = (x + 2 * cell_width, y + 2 * cell_height)
    # we need to crop the image
    
    cropped_img = img.crop(top_left + bottom_right)
    cropped_img = cropped_img.resize((cropped_img.width * upscale, cropped_img.height * upscale), resample=0)
    return cropped_img, top_left, bottom_right
#    cropped_img.save(new_image_path)
#    return 2 * cell_width, 2 * cell_height

def superimpose_images(image1_path, image2_path, opacity):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Ensure both images have the same size
    if image1.size != image2.size:
        raise ValueError("Images must have the same dimensions.")

    # Convert the images to RGBA mode if they are not already
    image1 = image1.convert("RGBA")
    image2 = image2.convert("RGBA")

    # Create a new image with the same size as the input images
    merged_image = Image.new("RGBA", image1.size)

    # Convert image1 to grayscale
    image1 = image1.convert("L")

    # Paste image1 onto the merged image
    merged_image.paste(image1, (0, 0))

    # Create a new image for image2 with adjusted opacity
    image2_with_opacity = Image.blend(Image.new("RGBA", image2.size, (0, 0, 0, 0)), image2, opacity)

    # Paste image2 with opacity onto the merged image
    merged_image = Image.alpha_composite(merged_image, image2_with_opacity)

    return merged_image



# Example usage
if __name__ == "__main__":
    create_grid_image(2880, 1712, 'yellow', 'green', 6, 'test.png')

