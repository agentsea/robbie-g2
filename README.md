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

**Robbie G2** is our cutting-edge, second-generation AI Agent that's revolutionizing the way we interact with complex real-life GUIs! This incredible AI is capable of navigating intricate interfaces and solving a wide array of tasks, from finding information and booking accommodations to sending emails, posting tweets, and even developing programs. Powered by an Ubuntu-based VM, Robbie G2 skillfully manipulates the mouse and keyboard to achieve its goals.

‣ Ready to dive in? Follow the instructions below to get started on your AI adventure!

‣ Curious about the inner workings of Robbie G2? Stay tuned for our upcoming Deep Dive, where you'll discover how this AI thinks and perceives the world!

‣ Join our vibrant community on [Discord](https://discord.gg/hhaq7XYPS6) to discuss Robbie G2, share experiences, and explore the fascinating world of AI Agents!

## Quick Start

### Prerequisites

* [Install Docker](https://docs.docker.com/engine/install/) - you need it to run a Tracker
* [Install QEMU](https://docs.hub.agentsea.ai/configuration/qemu) OR [Configure GCP](https://docs.hub.agentsea.ai/configuration/gcp) OR [Configure AWS](https://docs.hub.agentsea.ai/configuration/aws) - you need one of these to host a Device

### Setup 

1. Setup your OpenAI API key:

```sh
export OPENAI_API_KEY=<your key>
```

2. Clone the repository and go to the root folder:

```sh
git clone git@github.com:agentsea/robbie-g2.git && cd robbie-g2
```

3. Install dependencies:

```sh
poetry install
```

### Creating required entities

4. Create a tracker:

```sh
surfkit create tracker --name tracker01
```

5. Create a device:

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

6. Create an agent:

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
