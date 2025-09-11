# Development Guide

This guide provides instructions for setting up your development environment.

## Setup

1.  **Prerequisites**
    - Python 3.11
    - `make`

2.  **Clone the repository and install dependencies**
    ```bash
    git clone https://github.com/your-username/doc2knowledge.git
    cd doc2knowledge
    make install
    ```

## Running checks

-   **Run tests:** `make test`
-   **Run linter and formatter:** `make lint`

## Docker

You can also build the docker image and run commands inside the container.

-   **Build the image:** `make docker-build`
