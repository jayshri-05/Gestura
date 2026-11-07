# Deploy GESTURA on Render

This package is prepared as one Docker web service. It serves the React site and FastAPI model API from the same HTTPS address, which avoids a separate frontend/API URL setup and allows browser camera access.

## What is included

- React/Vite frontend and local ASL learning SVGs
- FastAPI prediction service
- Trained ASL-HG model checkpoint and its evaluation summary
- Docker deployment configuration

The raw ASL-HG archives, processed image caches, Python virtual environments and `node_modules` are not in this download. They are several gigabytes and are not needed to run the pretrained app. The training script and dataset attribution/verification are included; see `README.md` to reproduce training.

## Deploy steps

1. Download and extract `GESTURA-download.zip` on your computer.
2. Create a **private** empty GitHub repository. Do not add a README or other starter files.
3. Install Git for Windows if `git --version` does not work. Open PowerShell in the extracted `GESTURA` folder and run:

   ```powershell
   git init
   git add .
   git commit -m "Prepare GESTURA deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   git push -u origin main
   ```

   Replace the GitHub URL with the URL of your empty repository. GitHub will prompt you to sign in if needed.

4. Sign in to Render, choose **New → Web Service**, connect the GitHub repository, and select it.
5. Choose **Docker** as the runtime (Render detects the included `Dockerfile`). Choose a region and an instance size with enough memory for PyTorch; the smallest free instances can run out of memory while loading ML libraries.
6. Create the service and wait for the Docker build and deploy to finish. No API keys or environment variables are required.
7. Open the generated `https://…onrender.com` address. Check `/api/health` reports `{"status":"ok","model_loaded":true}`, then return to the home page and allow camera access.

The first Docker build downloads PyTorch and can take several minutes. On Render's free web service, an idle app can spin down and take about a minute to wake; free filesystems are ephemeral. The model and site are baked into the Docker image, so they are restored with each deploy. Learned-letter progress is kept in the user's browser.

## Update after edits

Commit and push the changes to GitHub; Render rebuilds and deploys the service automatically when auto-deploy is enabled.

## Local development

The API uses `http://127.0.0.1:8001` in Vite development mode. Follow the separate backend and frontend commands in `README.md`. The production build uses same-origin API requests.
