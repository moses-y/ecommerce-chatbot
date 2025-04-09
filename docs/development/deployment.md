### `docs/development/deployment.md`


# Deployment

This section covers options for deploying the Ari chatbot, focusing on Hugging Face Spaces using Docker.

## Hugging Face Spaces (Recommended)

Deploying as a Docker container on Hugging Face Spaces is the primary method used in this project's CI/CD pipeline.

**Prerequisites:**

*   A Hugging Face account.
*   A Hugging Face Space created, configured to use **Docker**.
*   Secrets (like `GOOGLE_API_KEY`) added to the Hugging Face Space settings (accessible via "Settings" > "Secrets" in your Space).

**Deployment Process (Automated via CI/CD):**

1.  The GitHub Actions workflow defined in `.github/workflows/ci.yml` handles deployment.
2.  On a push to the `main` branch (after tests pass), the workflow checks out the code.
3.  It uses `git` to push the repository contents directly to the Hugging Face Space repository (`https://huggingface.co/spaces/MoeTensors/E-commerce_chatbot`).
4.  Hugging Face Spaces automatically detects the `Dockerfile` in the pushed code.
5.  It builds the Docker image based on the `Dockerfile`.
6.  It runs a container from the built image, injecting the secrets you configured in the Space settings as environment variables.
7.  The application starts inside the container (running `python app.py`).

**Live Application:**

The deployed application is accessible at:
**[https://huggingface.co/spaces/MoeTensors/E-commerce_chatbot](https://huggingface.co/spaces/MoeTensors/E-commerce_chatbot)**

**Manual Deployment (Alternative):**

You can manually push your code to the Hugging Face Space repository using Git, similar to the CI/CD step:

```bash
# Clone your Space repo locally (if you haven't already)
# git clone https://huggingface.co/spaces/MoeTensors/E-commerce_chatbot

# Make changes, commit them

# Add the space remote if needed
# git remote add space https://huggingface.co/spaces/MoeTensors/E-commerce_chatbot

# Push to the space (replace <branch> with main usually)
# git push space <branch>
Docker (Local Development/Other Platforms)
You can build and run the Docker container locally or on other platforms.
```
Build the Docker image:
```bash

docker build -t ecommerce-chatbot:latest .
```
Run the container locally:
```bash
# Ensure you have a .env file in the directory where you run this
docker run -p 7860:7860 -d --name ari-chatbot-local \
  --env-file .env \
  ecommerce-chatbot:latest
Access locally at http://localhost:7860.
Docker Compose (Local Development)
Use Docker Compose for easier local management.
```
Ensure your local .env file is configured.
Run:
```bash
docker-compose up -d
```
This uses the docker-compose.yml file to build and run the chatbot service. Access locally at http://localhost:7860.

## Other Options
Serverless Platforms (Cloud Run, App Runner, etc.): These platforms can often run containers directly. You would build the image (e.g., push it to Google Artifact Registry or Docker Hub) and configure the service to use that image, ensuring environment variables (secrets) are set correctly in the platform's configuration.

Virtual Machines: Requires manual setup of Python, dependencies, potentially a web server gateway like Gunicorn, and a process manager (systemd, supervisor) to run app.py. Less recommended than containerization.

## Considerations
Secrets Management: NEVER commit .env files. Use GitHub Secrets for CI/CD and the hosting platform's secrets management (like Hugging Face Space secrets) for deployment.

Database: The default SQLite database (data/chatbot_data.db) works for simple cases. For production or scaling, 

consider:
Using a managed database service (PostgreSQL, MySQL). Update DATABASE_URL accordingly.
Persisting the SQLite file using volumes if deploying containers outside of ephemeral platforms like standard HF Spaces.

Scaling: Hugging Face Spaces offers paid tiers for more resources. For self-hosting, deploy multiple container instances behind a load balancer.

HTTPS: Hugging Face Spaces provides HTTPS automatically. If self-hosting, ensure HTTPS is configured (e.g., using a reverse proxy like Nginx or Caddy).