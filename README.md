<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">Robbie G2</h1>
    <p align="center">
    <img src="images/robbie.jpg" alt="Robbie G2 Logo" width="200" style="border-radius: 50px;">
    </p>
    <p align="center">
    <strong>Gen 2</strong> AI Agent that uses OCR, Canny Composite, and Grid to navigate GUIs
    <br />
    <br />
    <a href="https://docs.hub.agentsea.ai/introduction"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/agentsea/robbie-g2">View Demo</a>
    ·
    <a href="https://github.com/agentsea/robbie-g2/issues">Report Bug</a>
    ·
    <a href="https://github.com/agentsea/robbie-g2/issues">Request Feature</a>
  </p>
</p>

Meet Robbie, or Gen 2 agent. 

Robbie navigates GUIs to solve tasks for you.  

Unlike other bots, he doesn't just work on the web because he doesn't use Playwright. Robbie is a pure multimodal bot.  He can navigate the web or a desktop.  

That means he can navigate SaaS apps or he can work on a remote desktop and send emails, search for flights, check Slack, do research and more. 

Robbie-g2, aka Gen 2, is a leap from our first gen agents, SurfPizza and SurfSlicer.  He's very capable at navigating complex, never before seen GUIs via a remote virtual desktop which the AgentSea stack serves up as a device to him via DeviceBay. He connects to it via ToolFuse and AgentDesk, which lets him know what he can do with it, like move the mouse, send key commands, etc. 

‣ Check out our community on [Discord](https://discord.gg/hhaq7XYPS6) where we develop in the open, share research and connect with other developers who are building cutting edge agents or who just want to use them to get things done!

[![Watch the video](https://img.youtube.com/vi/R6rR27I6oFg/0.jpg)](https://www.youtube.com/watch?v=R6rR27I6oFg&t=71s)


## Quick Start

### Prerequisites

* [Install Docker](https://docs.docker.com/engine/install/) - you need it to run a Tracker
* [Install QEMU](https://docs.hub.agentsea.ai/configuration/qemu) OR [Configure GCP](https://docs.hub.agentsea.ai/configuration/gcp) OR [Configure AWS](https://docs.hub.agentsea.ai/configuration/aws) - you need one of these to host a Device

### Setup 

1. Setup your OpenAI API key:

```sh
export OPENAI_API_KEY=<your key>
```

2. Install/upgrade SurfKit:

```sh
pip install -U surfkit
```

3. Clone the repository and go to the root folder:

```sh
git clone git@github.com:agentsea/robbie-g2.git && cd robbie-g2
```

4. Install dependencies:

```sh
poetry install
```

### Creating required entities

5. Create a tracker:

```sh
surfkit create tracker --name tracker01
```

6. Create a device:

  - If you are using QEMU:

```sh
surfkit create device --provider qemu --name device01
```

  - If you are using GCE:

```sh
surfkit create device --provider gce --name device01
```

  - If you are using AWS:

```sh
surfkit create device --provider aws --name device01
```

7. Create an agent:

```sh
surfkit create agent --name agent01
```

### Solving a task

```sh
surfkit solve "Search for common varieties of french ducks" \
  --tracker tracker01 \
  --device device01 \
  --agent agent01
```

## Documentation

See our [docs](https://docs.hub.agentsea.ai) for more information on how to use Surfkit.
