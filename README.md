# Robbie G2

Robbie G2 is a Gen 2 AI Agent that uses OCR, Canny Composite, and Grid to navigate GUIs.

* [Demo](link)
* [Deep Dive](link)

## Install

```sh
pip install surfkit
```

## Quick Start

Create a tracker

```sh
surfkit create tracker --name tracker01
```

Create a device

```sh
surfkit create device --provider gce --name device01
```

Create the agent

```sh
surfkit create agent -t robbie/RobbieG2 --name agent01
```

Solve a task

```sh
surfkit solve "Search for common varieties of french ducks" \
  --tracker tracker01 \
  --device device01 \
  --agent agent01
```

## Community

Come join us on [Discord](https://discord.gg/hhaq7XYPS6).

## Documentation

See our [docs](https://docs.hub.agentsea.ai) for more information on how to use Surfkit.

## Developing

Install dependencies

```sh
poetry install
```

Create a tracker

```sh
surfkit create tracker --name tracker01
```

Create a device

```sh
surfkit create device --provider gce --name device01
```

Create the agent

```sh
surfkit create agent --name agent01
```

Solve a task

```sh
surfkit solve "Search for common varieties of french ducks" \
  --tracker tracker01 \
  --device device01 \
  --agent agent01
```
