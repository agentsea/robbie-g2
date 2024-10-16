import logging
import os
import time
import requests

from agentdesk.device import Desktop
from mllm import Router
from rich.console import Console
from taskara import Task
from toolfuse import Tool, action

from .clicker import find_coordinates

router = Router.from_env()
console = Console()

logger = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", logging.DEBUG)))


class SemanticDesktop(Tool):
    """A semantic desktop replaces click actions with semantic description rather than coordinates"""

    def __init__(
        self, task: Task, desktop: Desktop, data_path: str = "./.data"
    ) -> None:
        """
        Initialize and open a URL in the application.

        Args:
            task: Agent task. Defaults to None.
            desktop: Desktop instance to wrap.
            data_path (str, optional): Path to data. Defaults to "./.data".
        """
        super().__init__(wraps=desktop)
        self.desktop = desktop

        self.data_path = data_path
        self.img_path = os.path.join(self.data_path, "images", task.id)
        os.makedirs(self.img_path, exist_ok=True)

        self.task = task
        self.last_click_failed = False

        self.results = {
            "first_ocr": 0,
            "second_ocr": 0,
            "full_grid": 0,
            "failure": 0,
        }

    @action
    def clean_text(self) -> str:
        """Clean the text input or area currently in focus. 
        Use when you see wrong data in the text input or area which is currently in focus
        and need to clean it before entering new text.
        """
        self.desktop.hot_key(["ctrl", "a"])
        self.desktop.hot_key(["del"])
        
    @action
    def click_object(self, description: str, text: str, type: str, button: str = "left") -> None:
        """Click on an object on the screen

        Args:
            description (str): The description of the object including its general location, for example
                "a round dark blue icon with the text 'Home' in the top-right of the image", please be a generic as possible
            text (str): The text written on the object to click on. For example, 
                for "a round dark blue icon with the text 'Home' in the top-right of the image",
                the text is 'Home'. For input with its name written right inside it, write here the name of the input, 
                for example, 'Where to'. For the calendar date, put here only a day, for example, '15'.
                If the object doesn't have any text on or in it, return emply string.
            type (str): Type of click, can be 'single' for a single click or
                'double' for a double click. If you need to launch an application from the desktop choose 'double'
            button (str, optional): Mouse button to click. Options are 'left' or 'right'. Defaults to 'left'.
        """
        if type not in ["single", "double"]:
            raise ValueError("type must be 'single' or 'double'")
        
        self.task.post_message(
            role="Clicker",
            msg=f"Current statistics: {self.results}",
            thread="debug",
        )
        
        coords = find_coordinates(self, description, text)
        
        if coords:
            click_x, click_y = coords["x"], coords["y"]
            message = f"Attempting to click coordinates {click_x}, {click_y}."
            self.task.post_message(
                role="Clicker",
                msg=message,
                thread="debug",
            )
            self.last_debug_message = message
            self._click_coords(x=click_x, y=click_y, type=type, button=button)
        else:
            # Note: Given that GRID almost always returns something, we should almost never be here
            self.results["failure"] += 1
            self.task.post_message(
                role="Clicker",
                msg=f"No coordinates found for {description}.",
                thread="debug",
            )
            self.last_click_failed = True

    def _click_coords(
        self, x: int, y: int, type: str = "single", button: str = "left"
    ) -> None:
        """Click mouse button

        Args:
            x (Optional[int], optional): X coordinate to move to, if not provided
                it will click on current location. Defaults to None.
            y (Optional[int], optional): Y coordinate to move to, if not provided
                it will click on current location. Defaults to None.
            type (str, optional): Type of click, can be single or double. Defaults to "single".
            button (str, optional): Button to click. Defaults to "left".
        """
        # TODO: fix click cords in agentd
        logging.debug("moving mouse")
        body = {"x": int(x), "y": int(y)}
        resp = requests.post(f"{self.desktop.base_url}/v1/move_mouse", json=body)
        resp.raise_for_status()
        time.sleep(2)

        if type == "single":
            logging.debug("clicking")
            resp = requests.post(
                f"{self.desktop.base_url}/v1/click", json={"button": button}
            )
            resp.raise_for_status()
            time.sleep(2)
        elif type == "double":
            logging.debug("double clicking")
            resp = requests.post(
                f"{self.desktop.base_url}/v1/double_click", json={"button": button}
            )
            resp.raise_for_status()
            time.sleep(2)
        else:
            raise ValueError(f"unkown click type {type}")
        return
