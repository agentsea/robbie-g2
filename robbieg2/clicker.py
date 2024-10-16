import hashlib
import logging
import os
import re

from typing import List, Optional, Tuple

from PIL import Image, ImageDraw
from difflib import SequenceMatcher
from mllm import RoleMessage, RoleThread, Router
from pydantic import BaseModel, Field
from rich.console import Console
from rich.json import JSON

from .img import Box, image_to_b64
from .grid import create_grid_image, zoom_in, superimpose_images
from .easyocr import find_all_text_with_bounding_boxes
from .canny_composite import create_composite

router = Router.from_env()
console = Console()

logger = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", logging.DEBUG)))

COLOR_NUMBER = os.getenv("COLOR_NUMBER", "yellow")
COLOR_CIRCLE = os.getenv("COLOR_CIRCLE", "red")
GRID_SIZE = int(os.getenv("GRID_SIZE", 8)) # Amound of grid points equals (N-1)^2
NUM_CLUSTERS = int(os.getenv("NUM_CLUSTERS", 10)) # Number of clusters to use for the composite method
UPSCALE_FACTOR = int(os.getenv("UPSCALE_FACTOR", 3)) # How much we upscale the image on each step of zooming in
FIRST_OCR_THRESHOLD = float(os.getenv("FIRST_OCR_THRESHOLD", 0.9)) # Threshold for the first OCR pass
SECOND_OCR_THRESHOLD = float(os.getenv("SECOND_OCR_THRESHOLD", 0.7)) # Threshold for the second OCR pass


class ZoomSelection(BaseModel):
    """Zoom selection model"""
    number: int = Field(..., description="The number of the selected circle")


class CompositeSelection(BaseModel):
    """Composite selection model"""
    number: int = Field(..., description="The number of the selected section of the composite")

def recall_best_method_on_first_iteration(description: str) -> str:
    memory = {
        "calendar": run_grid,
        "date": run_grid,
        "link": run_composite,
        "icon": run_composite,
    }
    for key, value in memory.items():
        if key in description.lower():
            return value
    return run_composite

def recall_best_method_on_second_iteration(description: str) -> str:
    memory = {
        "calendar": run_grid,
        "date": run_grid,
        "link": run_grid,
        "icon": run_composite,
    }
    for key, value in memory.items():
        if key in description.lower():
            return value
    return run_grid

def find_coordinates(semdesk, description: str, text: str) -> dict:
    click_hash = hashlib.md5(description.encode()).hexdigest()[:5]
    bounding_boxes = []
    search_text = text
    if not search_text:
        matches = re.findall(r"['\"](.*?)['\"]", description)
        if len(matches) >= 2:
            search_text = None
        elif len(matches) == 1:
            search_text = matches[0]
        else:
            search_text = None

    # - Setting up the stage

    starting_img= semdesk.desktop.take_screenshots()[0]
    starting_img_path = os.path.join(semdesk.img_path, f"{click_hash}_starting.png")
    starting_img.save(starting_img_path)
    bounding_boxes.append(Box(0, 0, starting_img.width, starting_img.height))

    method = recall_best_method_on_first_iteration(description)

    # - OCR on the starting image

    if search_text and len(search_text) > 3:
        semdesk.task.post_message(
            role="Clicker",
            msg=f"Attempting OCR for: {search_text}",
            thread="debug",
        )
        ocr_results = find_all_text_with_bounding_boxes(starting_img_path)

        best_matches = [box for box in ocr_results if similarity_ratio(box['text'], search_text) >= FIRST_OCR_THRESHOLD]
        if len(best_matches) == 1:
            best_match = best_matches[0]
            x_mid = best_match["x"] + best_match["w"] // 2
            y_mid = best_match["y"] + best_match["h"] // 2
            bounding_boxes.append(Box(best_match['x'], best_match['y'], best_match['x'] + best_match['w'], 
                                      best_match['y'] + best_match['h']))
            semdesk.task.post_message(
                role="Clicker",
                msg=f"Found best matching text: '{best_match['text']}'",
                thread="debug",
            )
            # FIRST POINT OF POTENTIAL RESULT RETURN
            semdesk.results["first_ocr"] += 1
            debug_img = _debug_image(
                starting_img.copy(), bounding_boxes, (x_mid, y_mid)
            )
            semdesk.task.post_message(
                role="Clicker",
                msg="Final debug img",
                thread="debug",
                images=[image_to_b64(debug_img)],
            )
            return {
                "x": x_mid,
                "y": y_mid
            }
        else:
            semdesk.task.post_message(
                role="Clicker",
                msg=f"Found {len(best_matches)} best matches for text '{search_text}'. Continue...",
                thread="debug",
            )
    else:
        semdesk.task.post_message(
            role="Clicker",
            msg="No text to look for in the starting image.",
            thread="debug",
        )
    
    # - Finding a region where the element of interest is located

    semdesk.task.post_message(
        role="Clicker",
        msg="Looking for a region of interest...",
        thread="debug",
    )
    region_of_interest, bounding_box = method(semdesk, starting_img, starting_img_path, description, click_hash, "region")

    # Escape exit, if we didn't find the region of interest because the element is not on a screen:
    # we fall back to bruteforce method
    if region_of_interest is None:
        return backup_find_coordinates(semdesk, description)
    
    region_of_interest_b64 = image_to_b64(region_of_interest)
    semdesk.task.post_message(
        role="Clicker",
        msg="Found region of interest",
        thread="debug",
        images=[region_of_interest_b64],
    )
    bounding_boxes.append(bounding_box)
    region_of_interest_path = os.path.join(semdesk.img_path, f"{click_hash}_region_of_interest.png")
    region_of_interest.save(region_of_interest_path)

    # - OCR on the region we found
    if search_text:
        semdesk.task.post_message(
            role="Clicker",
            msg=f"Attempting OCR for: {search_text} on a region of interest",
            thread="debug",
        )
        zoomed_region_of_interest = region_of_interest.copy()
        zoomed_region_of_interest = zoomed_region_of_interest.resize((zoomed_region_of_interest.width * UPSCALE_FACTOR, zoomed_region_of_interest.height * UPSCALE_FACTOR), resample=0)
        zoomed_region_of_interest_path = os.path.join(semdesk.img_path, f"{click_hash}_zoomed_region_of_interest.png")
        zoomed_region_of_interest.save(zoomed_region_of_interest_path)
        ocr_results = find_all_text_with_bounding_boxes(zoomed_region_of_interest_path)
        best_matches = [box for box in ocr_results if similarity_ratio(box['text'], search_text) >= SECOND_OCR_THRESHOLD]

        # We trust OCR only of exactly one match over the threshold is found. Otherwise, we fall back to Grid/Composite.
        if len(best_matches) != 1:
            semdesk.task.post_message(
                role="Clicker",
                msg=f"No sufficiently similar text found. Found {len(best_matches)} ({best_matches})'",
                thread="debug",
            )
        else:            
            best_match = best_matches[0]
            relative_box = Box(best_match['x'], best_match['y'], best_match['x'] + best_match['w'], 
                                        best_match['y'] + best_match['h'])
            absolute_box = relative_box.to_absolute_with_upscale(bounding_boxes[-1], UPSCALE_FACTOR)
            x_mid, y_mid = absolute_box.center()
            bounding_boxes.append(absolute_box)
            semdesk.task.post_message(
                role="Clicker",
                msg=f"Found best matching text: '{best_match['text']}'",
                thread="debug",
            )
            # SECOND POINT OF POTENTIAL RESULT RETURN
            semdesk.results["second_ocr"] += 1
            debug_img = _debug_image(
                starting_img.copy(), bounding_boxes, (x_mid, y_mid)
            )
            semdesk.task.post_message(
                role="Clicker",
                msg="Final debug img",
                thread="debug",
                images=[image_to_b64(debug_img)],
            )
            return {
                "x": x_mid,
                "y": y_mid
            }

    # - Two passes of Grid/Composite + Zoom

    total_upscale = 1
    method = recall_best_method_on_second_iteration(description)

    region = region_of_interest.copy()
    region = region.resize((region.width * UPSCALE_FACTOR, region.height * UPSCALE_FACTOR), resample=0)
    region_of_interest_path = os.path.join(semdesk.img_path, f"{click_hash}_region_of_interest_zoom_1.png")
    region.save(region_of_interest_path)
    total_upscale *= UPSCALE_FACTOR
    new_region_of_interest, relative_bounding_box = method(semdesk, region, region_of_interest_path, description, click_hash, "zoom_1")

    # Escape exit, if we didn't find the region of interest because the element is not on a screen:
    # we fall back to bruteforce method
    if new_region_of_interest is None:
        return backup_find_coordinates(semdesk, description)

    absolute_box_zoomed = relative_bounding_box.to_absolute_with_upscale(
        bounding_boxes[-1], total_upscale
    )
    bounding_boxes.append(absolute_box_zoomed)
    
    region = new_region_of_interest.copy()
    region = region.resize((region.width * UPSCALE_FACTOR, region.height * UPSCALE_FACTOR), resample=0)
    region_of_interest_path = os.path.join(semdesk.img_path, f"{click_hash}_region_of_interest_zoom_2.png")
    region.save(region_of_interest_path)
    total_upscale *= UPSCALE_FACTOR
    last_region_of_interest, relative_bounding_box = method(semdesk, region, region_of_interest_path, description, click_hash, "zoom_2")

    # Escape exit, if we didn't find the region of interest because the element is not on a screen:
    # we fall back to bruteforce method
    if last_region_of_interest is None:
        return backup_find_coordinates(semdesk, description)

    absolute_box_zoomed = relative_bounding_box.to_absolute_with_upscale(
        bounding_boxes[-1], total_upscale
    )
    bounding_boxes.append(absolute_box_zoomed)

    x_mid, y_mid = bounding_boxes[-1].center()
    logger.info(f"clicking exact coords {x_mid}, {y_mid}")
    semdesk.task.post_message(
        role="Clicker",
        msg=f"Clicking coordinates {x_mid}, {y_mid}",
        thread="debug",
    )

    # LAST POINT OF POTENTIAL RETURN - WE ALWAYS RETURN SOMETHING FROM HERE, UNLESS THERE WAS AN EXCEPTION
    semdesk.results["full_grid"] += 1
    debug_img = _debug_image(
        starting_img.copy(), bounding_boxes, (x_mid, y_mid)
    )
    semdesk.task.post_message(
        role="Clicker",
        msg="Final debug img",
        thread="debug",
        images=[image_to_b64(debug_img)],
    )
    return {
        "x": x_mid,
        "y": y_mid
    }

def backup_find_coordinates(semdesk, description: str) -> dict:
    # This is a backup method of finding coordinates to click on. If the core method above fails at some point, 
    # with a region_of_interest being 0 (i.e. not found), we try ones again through the most bruteforce mechanics that we have: 
    # running three level of Grid + Zoom In; if that fails too, then we surely get back to Big Brain and ask some questions.
    semdesk.task.post_message(
        role="Clicker",
        msg="Coordinates are not found. Falling back to bruteforce 3-level Grid Zoom In.",
        thread="debug",
    )

    click_hash = hashlib.md5(description.encode()).hexdigest()[:5]
    bounding_boxes = []
    total_upscale = 1
    method = run_grid

    starting_img = semdesk.desktop.take_screenshots()[0]
    starting_img_path = os.path.join(semdesk.img_path, f"{click_hash}_starting.png")
    starting_img.save(starting_img_path)
    bounding_boxes.append(Box(0, 0, starting_img.width, starting_img.height))

    region_of_interest = starting_img.copy()

    for i in [0, 1, 2]:
        semdesk.task.post_message(
            role="Clicker",
            msg=f"Zooming in, level {i}...",
            thread="debug",
        )

        region = region_of_interest.copy()
        region = region.resize((region.width * UPSCALE_FACTOR, region.height * UPSCALE_FACTOR), resample=0)
        region_of_interest_path = os.path.join(semdesk.img_path, f"{click_hash}_grid_region_{i}.png")
        region.save(region_of_interest_path)
        total_upscale *= UPSCALE_FACTOR
        region_of_interest, relative_bounding_box = method(semdesk, region, region_of_interest_path, description, click_hash, "zoom_{i}")

        # Escape exit, if we didn't find the region of interest because the element is not on a screen.
        if region_of_interest is None:
            semdesk.task.post_message(
                role="Clicker",
                msg=f"Failed to find {description} on the image. Getting back to Actor.",
                thread="debug",
            )
            return None

        absolute_box_zoomed = relative_bounding_box.to_absolute_with_upscale(
            bounding_boxes[-1], total_upscale
        )
        bounding_boxes.append(absolute_box_zoomed)

    x_mid, y_mid = bounding_boxes[-1].center()
    logger.info(f"clicking exact coords {x_mid}, {y_mid}")
    semdesk.task.post_message(
        role="Clicker",
        msg=f"Clicking coordinates {x_mid}, {y_mid}",
        thread="debug",
    )

    # LAST POINT OF POTENTIAL RETURN - WE ALWAYS RETURN SOMETHING FROM HERE, UNLESS THERE WAS AN EXCEPTION
    semdesk.results["full_grid"] += 1
    debug_img = _debug_image(
        starting_img.copy(), bounding_boxes, (x_mid, y_mid)
    )
    semdesk.task.post_message(
        role="Clicker",
        msg="Final debug img",
        thread="debug",
        images=[image_to_b64(debug_img)],
    )
    return {
        "x": x_mid,
        "y": y_mid
    }


def similarity_ratio(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def run_grid(semdesk, starting_image: Image.Image, starting_path: str, 
             description: str, click_hash: str, postfix: str) -> dict:
    img_width, img_height = starting_image.size
    starting_image_b64 = image_to_b64(starting_image)

    grid_path = os.path.join(semdesk.img_path, f"{click_hash}_grid_{postfix}.png")
    create_grid_image(
        img_width, img_height, COLOR_CIRCLE, COLOR_NUMBER, GRID_SIZE, grid_path
    )

    merged_image_path = os.path.join(
        semdesk.img_path, f"{click_hash}_merge_{postfix}.png"
    )
    merged_image = superimpose_images(starting_path, grid_path, 1)
    merged_image.save(merged_image_path)

    merged_image_b64 = image_to_b64(merged_image)
    semdesk.task.post_message(
        role="Clicker",
        msg="Merged image",
        thread="debug",
        images=[merged_image_b64],
    )

    thread = RoleThread()

    prompt = f"""
    You are an experienced AI trained to find the elements on the screen.
    You see a screenshot of the web application. 
    I have drawn some big {COLOR_NUMBER} numbers on {COLOR_CIRCLE} circles on this image 
    to help you to find required elements.
    Please tell me the closest big {COLOR_NUMBER} number on a {COLOR_CIRCLE} circle to the center of the {description}.
    
    It may be the case, there is no {description} anywhere on the screenshot that you see.
    If you are very sure that there is no {description} anywhere on the screenshot that you see, please return {{"number": 0}}.

    Please note that some circles may lay on the {description}. If that's the case, return the number in any of these circles.
    If the {description} is a long object, please pick the circle that is closest to the left top corner of the {description}.
    I have also attached the entire screenshot without these numbers for your reference.

    Please return you response as raw JSON following the schema {ZoomSelection.model_json_schema()}
    Be concise and only return the raw json, for example if the circle you wanted to select had a number 3 in it
    you would return {{"number": 3}}
    """

    msg = RoleMessage(
        role="user",
        text=prompt,
        images=[merged_image_b64, starting_image_b64],
    )
    thread.add_msg(msg)

    try:
        response = router.chat(
            thread, namespace="grid", expect=ZoomSelection, agent_id="RobbieG2", retries=1
        )
        if not response.parsed:
            raise SystemError("No response parsed from zoom")
        
        semdesk.task.add_prompt(response.prompt)

        zoom_resp = response.parsed
        semdesk.task.post_message(
            role="Clicker",
            msg=f"Selection {zoom_resp.model_dump_json()}",
            thread="debug",
        )
        console.print(JSON(zoom_resp.model_dump_json()))
        chosen_number = zoom_resp.number
    except Exception as e:
        logger.info(f"Error in analyzing grid: {e}.")

    if chosen_number == 0:
        return None, None

    region_of_interest, top_left, bottom_right = zoom_in(
        starting_path, GRID_SIZE, chosen_number, 1
    )
    bounding_box = Box(
        top_left[0], top_left[1], bottom_right[0], bottom_right[1]
    )
    return region_of_interest, bounding_box


def run_composite(semdesk, starting_image: Image.Image, starting_path: str, 
                  description: str, click_hash: str, postfix: str) -> dict:
    composite_path = os.path.join(semdesk.img_path, f"{click_hash}_composite_{postfix}.png")
    composite_pil, bounding_boxes = create_composite(starting_path, NUM_CLUSTERS)
    composite_pil.save(composite_path)
    composite_b64 = image_to_b64(composite_pil)

    starting_image_b64 = image_to_b64(starting_image)

    semdesk.task.post_message(
        role="Clicker",
        msg="Composite image",
        thread="debug",
        images=[composite_b64],
    )

    thread = RoleThread()

    prompt = f"""
    You are an experienced AI trained to find the elements on the screen.
    You see a composite of several section of the screenshpt of the web application.
    You also see the entire screenshot for the reference.

    I have drawn some big {COLOR_NUMBER} numbers on the left panel of the composite image. 
    Please tell me the number of the section of the composite image that contains the {description}.
        
    It may be the case, there is no {description} anywhere on the screenshot that you see.
    If you are very sure that there is no {description} anywhere on the screenshot that you see, please return {{"number": 0}}.

    Please return you response as raw JSON following the schema {CompositeSelection.model_json_schema()}
    Be concise and only return the raw json, for example if the section has a number 3, 
    you should return {{"number": 3}}
    """

    msg = RoleMessage(
        role="user",
        text=prompt,
        images=[composite_b64, starting_image_b64],
    )
    thread.add_msg(msg)

    try:
        response = router.chat(
            thread, namespace="composite", expect=CompositeSelection, agent_id="RobbieG2", retries=1
        )
        if not response.parsed:
            raise SystemError("No response parsed from zoom")
        
        semdesk.task.add_prompt(response.prompt)

        composite_resp = response.parsed
        semdesk.task.post_message(
            role="Clicker",
            msg=f"Selection {composite_resp.model_dump_json()}",
            thread="debug",
        )
        console.print(JSON(composite_resp.model_dump_json()))
        chosen_number = composite_resp.number
    except Exception as e:
        logger.info(f"Error in analyzing composite: {e}.")

    if chosen_number == 0:
        return None, None

    bounding_box = bounding_boxes[chosen_number - 1]
    top_left = (bounding_box[0], bounding_box[1])
    bottom_right = (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
    region_of_interest = starting_image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    box = Box(top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    return region_of_interest, box

def _debug_image(
    img: Image.Image,
    boxes: List[Box],
    final_click: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for box in boxes:
        box.draw(draw)

    if final_click:
        draw.ellipse(
            [
                final_click[0] - 5,
                final_click[1] - 5,
                final_click[0] + 5,
                final_click[1] + 5,
            ],
            fill="red",
            outline="red",
        )
    return img
