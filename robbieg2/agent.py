import logging
import os
import time
import traceback
from typing import Final, List, Optional, Tuple, Type

from agentdesk.device_v1 import Desktop
from devicebay import Device
from pydantic import BaseModel, Field
from rich.console import Console
from rich.json import JSON
from skillpacks.server.models import V1Action
from surfkit.agent import TaskAgent
from taskara import Task, TaskStatus
from tenacity import before_sleep_log, retry, stop_after_attempt
from threadmem import RoleMessage, RoleThread
from mllm import ChatResponse
from toolfuse.util import AgentUtils

from .tool import SemanticDesktop, router
from .clicker import similarity_ratio
from .cheap_critic import assess_action_result


logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))

console = Console(force_terminal=True)


class RobbieG2Config(BaseModel):
    pass


class ActorThoughts(BaseModel):
    """An represention of thoughts of the Actor part of the brain."""

    observation: str = Field(
        ..., description="Observations of the current state of the environment"
    )
    reason: str = Field(
        ...,
        description="The reason why this action was chosen, explaining the logic or rationale behind the decision.",
    )
    action: V1Action = Field(
        ...,
        description="The action object detailing the specific action to be taken, including its name and parameters.",
    )

class NeocortexPrediction(BaseModel):
    """An represention of thoughts of the Neocortex part of the brain."""
    
    prediction: str = Field(
        ..., description="Prediction about the state of the environment after the current action"
    )
    reason: str = Field(
        ...,
        description="The reason why the next action is chosen, explaining the logic or rationale behind the decision.",
    )
    action: V1Action = Field(
        ...,
        description="The action object detailing the next action to be taken after the current action takes place, including its name and parameters.",
    )

class NeocortexThoughts(BaseModel):
    """An represention of thoughts of the Neocortex part of the brain."""
    
    prediction_1: NeocortexPrediction = Field(
        ..., description="Prediction about the state of the environment after the current action, chosen by Actor, and the most appropriate next action"
    )
    prediction_2: NeocortexPrediction = Field(
        ..., description="Prediction about the state of the environment after the first predicted action, and the most appropriate action after that"
    )

class CriticThoughts(BaseModel):
    """An represention of thoughts of the Critic part of the brain."""

    critic: str = Field(..., description="Critic's thoughts about whether the current state of environment corresponds to a given task, and if not, now to recover.")

class BrainThoughts(BaseModel):
    """An represention of thoughts of the whole brain."""

    critic: CriticThoughts = Field(..., description="Thoughts of the Critic part of the brain.")

    actor: ActorThoughts = Field(..., description="Thoughts of the Actor part of the brain.")

    neocortex: NeocortexThoughts = Field(..., description="Thoughts of the Neocortex part of the brain.")

class InterruptionCriticThoughts(BaseModel):
    """A representation of thoughts of the Critic which was interrupted because we repeat the same actions again and again."""

    critic: str = Field(..., description="Critic's assessment on whether taking the current action is a good idea and whether the previous similar action were appropriate and successful.")

    action: V1Action = Field(..., description="The most appropripriate next action given the entire situation.")


class RobbieG2(TaskAgent):
    """A GUI desktop agent that slices up the image"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_actions = []


    def record_action(self, action: dict) -> None:
        self.past_actions.append(action)


    def find_the_closest_actions(self, action: V1Action, depth: int = 10, threshold: float = 0.8) -> [V1Action]:
        recent_actions = self.past_actions[-depth:]
        closest_actions = []

        for past_action in reversed(recent_actions):
            if action.name == "type_text" or action.name == "click_object":
                action_params = str(action.parameters)
                past_action_parames = str(past_action.parameters)
                similarity = similarity_ratio(action_params, past_action_parames)
                if similarity > threshold:
                    closest_actions.append(past_action)
            else:
                action_str = str(action)
                past_action_str = str(past_action)
                similarity = similarity_ratio(action_str, past_action_str)
                if similarity > 0.95:
                    closest_actions.append(past_action)
        
        return closest_actions


    def solve_task(
        self,
        task: Task,
        device: Optional[Device] = None,
        max_steps: int = 30,
    ) -> Task:
        """Solve a task

        Args:
            task (Task): Task to solve.
            device (Device): Device to perform the task on.
            max_steps (int, optional): Max steps to try and solve. Defaults to 30.

        Returns:
            Task: The task
        """
        start_time = time.time()  # Start time measurement

        # Post a message to the default thread to let the user know the task is in progress
        task.post_message("Actor", f"Starting task '{task.description}'")

        # Create threads in the task to update the user
        console.print("creating threads...")
        task.ensure_thread("debug")
        task.post_message("Actor", "I'll post debug messages here", thread="debug")

        # Check that the device we received is one we support
        if not isinstance(device, Desktop):
            raise ValueError("Only desktop devices supported")

        # Wrap the standard desktop in our special tool
        semdesk = SemanticDesktop(task=task, desktop=device)

        # Add standard agent utils to the device
        semdesk.merge(AgentUtils())

        # Open a site if present in the parameters
        site = task._parameters.get("site") if task._parameters else None
        if site:
            console.print(f"▶️ opening site url: {site}", style="blue")
            task.post_message("Body", f"opening site url {site}...")
            semdesk.desktop.open_url(site)
            console.print("waiting for browser to open...", style="blue")
            time.sleep(10)

        # Get info about the desktop
        info = semdesk.desktop.info()
        screen_size = info["screen_size"]
        console.print(f"Screen size: {screen_size}")

        # Get the json schema for the tools, excluding actions that aren't useful
        tools = semdesk.json_schema(
            exclude_names=[
                "move_mouse",
                "click",
                "drag_mouse",
                "mouse_coordinates",
                "take_screenshots",
                "open_url",
                "double_click",
            ]
        )
        console.print("tools: ", style="purple")
        console.print(JSON.from_data(tools))

        starting_prompt = f"""
You are RobbieG2, an advanced AI agent designed to navigate and interact with web interfaces. Your capabilities include:

1. Mouse control:
   - Move the mouse to specific coordinates
   - Click (single or double) at current or specified locations
   - Retrieve current mouse coordinates

2. Keyboard input:
   - Send key commands, including special keys like Tab, Enter, and arrow keys
   - Type text into form fields

3. Navigation:
   - Use Tab key to move through form elements
   - Scroll web pages

4. Visual analysis:
   - Take screenshots of the current view

5. Advanced interaction:
   - Click on objects based on semantic descriptions

*** Firefox Commands

Specifically, if you are using the Firefox browser, remember that you can use the following key commands:

* Press Ctrl + L or Alt + D to highlight the URL, then press Delete to clear it if there is incorrect text in the URL bar that you need to clear out.
* To clear the text in a field do the following: First, ensure the field is in focus BEFORE using this command. Then use Ctrl + A and then Backspace or Delete: This command first highlights all text in a field and then deletes that text. 
* Ctrl + Shift + Tab switches to the previous tab
* Ctrl + Tab switches to the next tab
* Press Backspace or Alt + Left Arrow to go to the previous page in your browsing history for the tab
* Press Shift + Backspace or Alt + Right Arrow to go to the next page in your browsing history for the tab
* Press F6 or Shift + F6 to switch focus to the next keyboard-accessible pane, which includes:

    Highlights the URL in the address bar
    Bookmarks bar (if visible)
    The main web content
    Downloads bar (if visible)


*** Chrome Commands

Specifically, if you are using the Chrome browser, remember that you can use the following key commands:

* Press Ctrl + L or Alt + D to highlight the URL, then press Delete to clear it if there is incorrect text in the URL bar that you need to clear out.
* `clean_text` is also a special command to clear fields but you MUST ensure the field is IN FOCUS first before using this command.
* Ctrl + Shift + Tab switches to the previous tab which is very useful if a new tab you don't want is opened and you need to get back to the last tab. 
* Ctrl + Tab switches to the next tab.	
* Press Backspace or Alt and the left arrow together - to go to the previous page in your browsing history for the tab.
* Press Shift+Backspace, or Alt and the right arrow together - to go to the next page in your browsing history for the tab. 	
# `Ctrl + A` and then `Backspace` or `Delete` - This command first highlights all text in a field and then deletes that text. It only works IF YOU ARE ALREADY IN THAT FIELD so be sure the field is in focus or clicked already and click it if you are unsure - this is one of the MOST IMPORTANT commands. You can use it to clear existing text from a field that is filled with incorrect information in a web form.
* F6 or Shift+F6 - Switches focus to the next keyboard-accessible pane. Panes include:

    Highlights the URL in the address bar
    Bookmarks bar (if visible)
    The main web content (including any infobars)
    Downloads bar (if visible)	

If you are unsure about whether a field is selected you can try to click it to ensure it is highlighted.  If you take an action several times in a row
and the result has not changed, for example, if a field has not changed to meet you expectation, then explore the idea of clicking it again to change it
and ensure it has the correct text that you want there.

Remember, remember that you DO NOT have the ability to take a screenshot to verify your results. 

Sometimes a page isn't fully loaded yet. If that is the case feel free to wait or pause briefly for the screenshot to indicate a fully loaded page.

If you get stuck in a loop of actions, use your curiosity to explore and try new things with trial and error. Learn from your mistakes and get better with
each new action.
    
The complete list of available tools is: {tools}

Your goal is to efficiently navigate web interfaces and complete tasks by leveraging these capabilities. 
Always consider the most appropriate method for each action, and be prepared to adapt your approach based 
on the results of your actions.

When faced with a task, think step-by-step about the best approach, considering all your available tools and methods. 

You brain consists of three major parts. 

1. The Critic is responsible for evaluating the current state of the environment and deciding whether it corresponds to a given task. 
If it doesn't, the Critic explains how to recover the environment to a state where it can complete the task. Always start with Critic
assessment, before choosing the next actions and predicting the next steps.

2. The Actor is responsible for picking the next action based on the current state of the environment and the tools available. 

3. The Neocortex is responsible for thinking ahead, predicting the state of the environment after the action that the Actor picked, 
choosing the next action after that, and so on. The Neocortex makes three predictions for the actions to be taken AFTER the one that 
the Actor picked. 

Your current task is {task.description}.

For each screenshot I will send you please return the complete thoughts of both parts of your brain as a
raw JSON adhearing to the schema {BrainThoughts.model_json_schema()}.

Let me know when you are ready and I'll send you the first screenshot.
"""

        # Create our thread and start with a system prompt
        thread = RoleThread()
        thread.post(
            role="user",
            msg=starting_prompt,
        )
        response = router.chat(thread, namespace="system")
        console.print(f"system prompt response: {response}", style="blue")
        thread.add_msg(response.msg)

        # Loop to run actions
        for i in range(max_steps):
            console.print(f"-------step {i + 1}", style="green")

            try:
                thread, done = self.take_action(semdesk, task, thread)
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.save()
                task.post_message("Actor", f"❗ Error taking action: {e}")
                end_time = time.time()  # End time measurement
                elapsed_time = end_time - start_time
                console.print(f"Time taken to solve task: {elapsed_time:.2f} seconds", style="green")
                return task

            if done:
                console.print("task is done", style="green")
                end_time = time.time()  # End time measurement
                elapsed_time = end_time - start_time
                console.print(f"Time taken to solve task: {elapsed_time:.2f} seconds", style="green")
                return task

            time.sleep(2)

        task.status = TaskStatus.FAILED
        task.save()
        task.post_message("Actor", "❗ Max steps reached without solving task")
        console.print("Reached max steps without solving task", style="red")

        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time
        console.print(f"Time taken to solve task: {elapsed_time:.2f} seconds", style="green")

        return task
    
    @retry(
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO),            
    )
    def interrupt_flow_and_ask_critic(
        self,
        semdesk: SemanticDesktop,
        task: Task,
        thread: RoleThread,
        current_action: dict
    ) -> dict:
        try:
            _thread = thread.copy()
            screenshot_img = semdesk.desktop.take_screenshots()[0]
            critic_prompt = f"""
You task is {task.description}. The screenshot is attached.
You are attempting to do the following action: {current_action}.
You have already attempted to do very similar actions very recently. 
Please assess if the previous actions very successful, and if you are sure that this action is exactly what needs to be done next.
If you are not sure, please consider various alternative options and pick the action that is most likely to lead us toward completing
the above-mentioned task. 
Give me the action to be done next, along with yours reasons for that.

Unlike other messages in this thread, please return your thoughts as as a
raw JSON adhearing to the schema {InterruptionCriticThoughts.model_json_schema()}.

Please return just the raw JSON.
"""
            # Craft the message asking the MLLM for an action
            msg = RoleMessage(
                role="user",
                text=critic_prompt,
                images=[screenshot_img],
            )
            _thread.add_msg(msg)

            # Make the action selection
            response = router.chat(
                _thread,
                namespace="action",
                expect=InterruptionCriticThoughts,
                agent_id=self.name(),
            )
            task.add_prompt(response.prompt)

            try:
                # Post to the user letting them know what the modle selected
                selection = response.parsed
                if not selection:
                    raise ValueError("No action selection parsed")
                
                task.post_message("Critic", f"🤔 {selection.critic}")
                task.post_message("Critic", f"▶️ I suggest to take action '{selection.action.name}' "+
                                            f"with parameters: {selection.action.parameters}")
                return selection.action
        
            except Exception as e:
                console.print(f"Response failed to parse: {e}", style="red")
                raise

        except Exception as e:
            console.print("Exception taking action: ", e)
            traceback.print_exc()
            task.post_message("Actor", f"⚠️ Error taking action: {e} -- retrying...")
            raise e


    @retry(
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def take_action(
        self,
        semdesk: SemanticDesktop,
        task: Task,
        thread: RoleThread,
    ) -> Tuple[RoleThread, bool]:
        """Take an action

        Args:
            desktop (SemanticDesktop): Desktop to use
            task (str): Task to accomplish
            thread (RoleThread): Role thread for the task

        Returns:
            bool: Whether the task is complete
        """
        try:
            # Check to see if the task has been cancelled
            if task.remote:
                task.refresh()
            console.print("task status: ", task.status.value)
            if (
                task.status == TaskStatus.CANCELING
                or task.status == TaskStatus.CANCELED
            ):
                console.print(f"task is {task.status}", style="red")
                if task.status == TaskStatus.CANCELING:
                    task.status = TaskStatus.CANCELED
                    task.save()
                return thread, True

            console.print("taking action...", style="white")

            # Create a copy of the thread, and remove old images
            _thread = thread.copy()
            _thread.remove_images()

            task.post_message("Actor", "🤔 I'm thinking...")

            # Take a screenshot of the desktop and post a message with it
            screenshot_img = semdesk.desktop.take_screenshots()[0]
            task.post_message(
                "Actor",
                "Current image",
                images=[screenshot_img],
                thread="debug",
            )

            # Get the current mouse coordinates
            x, y = semdesk.desktop.mouse_coordinates()
            console.print(f"mouse coordinates: ({x}, {y})", style="white")

            step_prompt = f"""
Here is a screenshot of the current desktop, please select next and the one after next action from the provided schema.

Critic: Carefully analyze the screenshot and check if the state corresponds to the task we are solving. Remember that 
the task is {task.description}.
Actor: Select a next action and explain why.
Neocortex: Predict the result of the action picked by Actor and pick the next ones.

Watch out for elements that are different from others, for example, have the border of the different color. 
Such elements are usually already in focus, and you can try to type text in them right away. 
However, if you tried to type on a previous step and want to type the same input again, you better 
focus on the input field first by clicking on it. 

Please return just the raw JSON.
"""
            
            # Craft the message asking the MLLM for an action
            msg = RoleMessage(
                role="user",
                text=step_prompt,
                images=[screenshot_img],
            )
            _thread.add_msg(msg)

            # Make the action selection
            response = router.chat(
                _thread,
                namespace="action",
                expect=BrainThoughts,
                agent_id=self.name(),
            )
            task.add_prompt(response.prompt)

            try:
                # Post to the user letting them know what the modle selected
                selection = response.parsed
                if not selection:
                    raise ValueError("No action selection parsed")
                
                task.post_message("Critic", f"🤔 {selection.critic.critic}")

                task.post_message("Actor",  f"👁️ {selection.actor.observation}\n" +
                                            f"💡 {selection.actor.reason}\n" +
                                            f"▶️ I'm going to take action '{selection.actor.action.name}' "+
                                            f"with parameters: {selection.actor.action.parameters}")
                
                task.post_message("Neocortex",  f"🔮 {selection.neocortex.prediction_1.prediction}\n" + 
                                                f"💡 {selection.neocortex.prediction_1.reason}\n" +
                                                f"🔜 The next action to take is '{selection.neocortex.prediction_1.action.name}' "+
                                                f"with parameters: {selection.neocortex.prediction_1.action.parameters}")
                
                task.post_message("Neocortex",  f"🔮 {selection.neocortex.prediction_2.prediction}\n" + 
                                                f"💡 {selection.neocortex.prediction_2.reason}\n" +
                                                f"🔜 The last action to take after that is '{selection.neocortex.prediction_2.action.name}' "+
                                                f"with parameters: {selection.neocortex.prediction_2.action.parameters}")

            except Exception as e:
                console.print(f"Response failed to parse: {e}", style="red")
                raise

            # The agent will return 'result' if it believes it's finished
            if selection.actor.action.name == "result":
                console.print(f"The final result is: {selection.actor.action.parameters['value']}", style="green")
                task.post_message(
                    "Actor",
                    f"✅ I think the task is done, please review the result: {selection.actor.action.parameters['value']}",
                )
                task.status = TaskStatus.FINISHED
                task.save()
                return _thread, True

            im_start = screenshot_img
            continue_chain = True
            interruption_requested = False

            for next_action in [selection.actor.action, 
                                selection.neocortex.prediction_1.action, 
                                selection.neocortex.prediction_2.action]:
                if not continue_chain or next_action.name == "result" or interruption_requested:
                    # Time to think again!
                    break

                # Hack for the cases when AI is willing to press "ctrl+s" or smth like that
                if next_action.name == "press_key" and "+" in next_action.parameters["key"]:
                    next_action.name = "hot_key"
                    next_action.parameters = {"keys": next_action.parameters["key"].split("+")}

                # Additional check to make sure that we are not trapped in a circle, do the same action again and again
                depth = 5 if (next_action.name == "press_key" or next_action.name == "hot_key") else 10
                closest_actions = self.find_the_closest_actions(next_action, depth=depth)
                if len(closest_actions) > 0:
                    task.post_message(
                        "Body",
                        f"Closest actions to the current one: {closest_actions}",
                        thread="debug"
                    )
                if len(closest_actions) >= 2:
                    task.post_message(
                        "Body",
                        "Too many repeated actions. Getting back to Critic.",
                        thread="debug"
                    )
                    # Well, look like it's time to interrupt the flow and reconsider our life choices. 
                    new_action = self.interrupt_flow_and_ask_critic(semdesk, task, thread, next_action)
                    next_action = new_action
                    # We'll run this updated action and get out of the cycle.
                    interruption_requested = True

                task.post_message(
                    "Body",
                    f"▶️ Taking action '{next_action.name}' with parameters: {next_action.parameters}",
                )
                self._take_selected_action(semdesk, next_action, task, _thread, response)
                self.record_action(next_action)

                # If we know for certian that the click was not successful, it's time to stop the chain
                # and think again
                if semdesk.last_click_failed:
                    semdesk.last_click_failed = False
                    break

                # Pressing keys change environment for sure, so we may just stop the exection here and think again
                if next_action.name == "press_key":
                    break

                # We analyze if we want to continue to the next action here. A cheap critic looks at the new screenshot and 
                # decides if we should continue the chain or not. 
                screenshot_upd = semdesk.desktop.take_screenshots()[0]
                ssim, continue_chain = assess_action_result(im_start, screenshot_upd)

                # If we were typing text, and the screen changed too much, then we probably hit some hot keys by accident
                # and scrolled down. We should stop and scroll back up, forcing recovery.
                if next_action.name == "type_text" and ssim < 0.9:
                    semdesk.desktop.scroll(30) # we may need to adjust this number
                    break

                # There is a chance that if the last action was a click, then the result didn't load yet, and the SSIM will be high
                # while it should be low. To avoid this, we check once again for this specific case in 5 seconds:
                if next_action.name == "click_object" and ssim > 0.95:
                    task.post_message("Critic", "😴 Waiting to be sure that the result is loaded...", thread="debug")
                    time.sleep(5)
                    screenshot_upd = semdesk.desktop.take_screenshots()[0]
                    ssim, continue_chain = assess_action_result(im_start, screenshot_upd)
                task.post_message("Critic", f"🔍 SSIM: {ssim}", thread="debug")
                
            return _thread, False

        except Exception as e:
            console.print("Exception taking action: ", e)
            traceback.print_exc()
            task.post_message("Actor", f"⚠️ Error taking action: {e} -- retrying...")
            raise e

    def _take_selected_action(self, semdesk: SemanticDesktop, action: V1Action, 
                              task: Task, thread: RoleThread, response: ChatResponse) -> None:
        """Take the selected action

        Args:
            semdesk (SemanticDesktop): Desktop to use
            action (V1Action): Action to take
        """
        console.log(f"taking action: {action}")

        # Find the selected action in the tool
        desktop_action = semdesk.find_action(action.name)
        console.print(f"found action: {desktop_action}", style="blue")
        if not desktop_action:
            console.print(f"action returned not found: {action.name}")
            raise SystemError("action not found")

        # Take the selected action
        try:
            action_response = semdesk.use(desktop_action, **action.parameters)
        except Exception as e:
            raise ValueError(f"Trouble using action: {e}")

        console.print(f"action output: {action_response}", style="blue")
        if action_response:
            task.post_message(
                "Actor", f"👁️ Result from taking action: {action_response}"
            )

        thread.add_msg(response.msg)

    @classmethod
    def supported_devices(cls) -> List[Type[Device]]:
        """Devices this agent supports

        Returns:
            List[Type[Device]]: A list of supported devices
        """
        return [Desktop]

    @classmethod
    def config_type(cls) -> Type[RobbieG2Config]:
        """Type of config

        Returns:
            Type[DinoConfig]: Config type
        """
        return RobbieG2Config

    @classmethod
    def from_config(cls, config: RobbieG2Config) -> "RobbieG2":
        """Create an agent from a config

        Args:
            config (RobbieG2Config): Agent config

        Returns:
            RobbieG2: The agent
        """
        return RobbieG2()

    @classmethod
    def default(cls) -> "RobbieG2":
        """Create a default agent

        Returns:
            RobbieG2: The agent
        """
        return RobbieG2()

    @classmethod
    def init(cls) -> None:
        """Initialize the agent class"""
        return


Agent = RobbieG2
