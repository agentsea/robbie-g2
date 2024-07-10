import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from PIL import Image, ImageDraw, ImageFont


def improved_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    return dilated

def group_elements(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    grouped_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 100:  # Ignore very small contours
            continue
        merged = False
        for group in grouped_contours:
            if any(cv2.boundingRect(c)[0] - 10 <= x <= cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] + 10 and
                   cv2.boundingRect(c)[1] - 10 <= y <= cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] + 10 for c in group):
                group.append(contour)
                merged = True
                break
        if not merged:
            grouped_contours.append([contour])
    
    return grouped_contours

def extract_bounding_boxes(grouped_contours):
    bounding_boxes = []
    for group in grouped_contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(group))
        bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def cluster_bounding_boxes(bounding_boxes, num_clusters):
    if len(bounding_boxes) <= num_clusters:
        return bounding_boxes

    # Calculate the centroids of the bounding boxes
    centroids = np.array([
        [box[0] + box[2] / 2, box[1] + box[3] / 2] for box in bounding_boxes
    ])

    # Apply Hierarchical clustering (Agglomerative Clustering)
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)
    labels = agg_clustering.fit_predict(centroids)

    # Calculate bounding boxes for each cluster
    cluster_bounding_boxes = []
    for i in range(num_clusters):
        cluster_boxes = [box for box, label in zip(bounding_boxes, labels) if label == i]
        if cluster_boxes:
            # Find the minimum and maximum coordinates for all boxes in the cluster
            min_x = min(box[0] for box in cluster_boxes)
            min_y = min(box[1] for box in cluster_boxes)
            max_x = max(box[0] + box[2] for box in cluster_boxes)
            max_y = max(box[1] + box[3] for box in cluster_boxes)
            
            # Calculate the dimensions of the bounding box that encompasses all boxes
            width = max_x - min_x
            height = max_y - min_y
            
            # Create a new bounding box that encompasses all boxes in the cluster
            cluster_box = (min_x, min_y, width, height)
            cluster_bounding_boxes.append(cluster_box)

    return cluster_bounding_boxes

def create_composite_image(bounding_boxes, image):
    number_column_width = 100
    image_column_width = max(box[2] for box in bounding_boxes)
    row_heights = [box[3] + 4 for box in bounding_boxes]
    total_width = number_column_width + image_column_width + 1  # +1 for rightmost line
    total_height = sum(row_heights) + len(bounding_boxes) + 1  # +1 for each row separator and bottom line
    
    composite = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(composite)
    
    try:
        font = ImageFont.truetype("./font/arialbd.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
        print("Arial font not found in ./font directory. Using default font.")
    
    # Draw grid lines
    for i in range(len(bounding_boxes) + 1):
        y = sum(row_heights[:i]) + i
        draw.line([(0, y), (total_width, y)], fill='black', width=1)
    draw.line([(number_column_width, 0), (number_column_width, total_height)], fill='black', width=1)
    draw.line([(total_width - 1, 0), (total_width - 1, total_height)], fill='black', width=1)
    
    y_offset = 1  # Start after the top line
    for i, box in enumerate(bounding_boxes):
        # Draw number
        number_text = str(i+1)
        text_bbox = draw.textbbox((0, 0), number_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (number_column_width - text_width) // 2
        text_y = y_offset + (row_heights[i] - text_height) // 2 + 2
        draw.text((text_x, text_y), number_text, font=font, fill='red')
        
        # Paste image slice
        box_pil = image.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
        paste_x = number_column_width + 1
        composite.paste(box_pil, (paste_x, y_offset + 2))
        # Draw a rectangle around the pasted image
        draw.rectangle(
            [
                (paste_x, y_offset + 2),
                (paste_x + box[2] - 1, y_offset + 2 + box[3] - 1)
            ],
            outline="green",
            width=2
        )
        
        y_offset += row_heights[i] + 1  # Move to next row, accounting for grid line
    
    return composite

def create_composite(image_path, num_clusters):
    image = cv2.imread(image_path)
    pil_image = Image.open(image_path)

    edges = improved_canny(image)
    grouped_contours = group_elements(edges)
    bounding_boxes = extract_bounding_boxes(grouped_contours)
    clustered_boxes = cluster_bounding_boxes(bounding_boxes, num_clusters)
    composite_image = create_composite_image(clustered_boxes, pil_image)
    return composite_image, clustered_boxes
