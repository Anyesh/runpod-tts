from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
import runpod.serverless

from dreamerytts import tts

RETRY_DELAY_S = os.getenv("RETRY_DELAY_S", 3)
# Maximum number of API check attempts
MAX_RETRIES = os.getenv("MAX_RETRIES", 50)

# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

COMFY_POLLING_MAX_RETRIES = int(os.getenv("COMFY_POLLING_MAX_RETRIES", 500))

COMFY_POLLING_INTERVAL_MS = int(os.getenv("COMFY_POLLING_INTERVAL_MS", 250))

API_FILE = os.getenv("API_FILE", "http://local.app.dreamery.ai/file.service/v1")
API_ML = os.getenv("API_ML", "http://local.app.dreamery.ai/ml.service/v1")
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "dreamery")
temp_dir = Path().home() / ".totoro" / "temp"


def handler(job):
    job_input = job["input"]

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    data = job_input

    prompt = data["prompt"]["text"]
    prompt_id = data.get("prompt_id")
    file_id = data.get("prompt")["file_id"]
    client_id = data["client_id"]
    token = data["token"]

    # Queue the workflow
    auth_cookie = {SESSION_COOKIE_NAME: token}
    # Create a session object
    session = requests.Session()
    # Attach the authentication cookie to the session
    session.cookies.update(auth_cookie)

    try:
        file_path = download_file(file_id, token)
        output_full_path = (
            temp_dir / f"generated/{client_id}/{prompt_id}/{file_id}/output.wav"
        )
        tts(prompt, output_full_path, speaker_wav=file_path)
    except Exception as e:
        return {"error": f"Error queuing workflow: {str(e)}"}

    images_result = process_output(session, file_id, client_id)

    result = {**images_result, "refresh_worker": REFRESH_WORKER}

    return result


def process_output(session, prompt_id, file_id, client_id):

    COMFY_OUTPUT_PATH = (
        f"/comfyui/output/generated/{client_id}/{prompt_id}/{file_id}/output.wav"
    )

    filename = "output.wav"
    if os.path.exists(COMFY_OUTPUT_PATH):
        try:
            files = {"file": (filename, open(COMFY_OUTPUT_PATH, "rb"))}
            file_res = retry_file_upload(
                session, API_FILE + "/upload?is_generated=true", files=files
            )

            if not file_res.ok:
                print(f"runpod-worker-comfy - Error uploading image: {file_res.text}")
                return {"error": f"Error uploading image: {file_res.text}"}

            print(
                f"runpod-worker-comfy - the image was generated and uploaded: {file_res.json()}"
            )

        except Exception as e:
            print(f"runpod-worker-comfy - Error uploading image: {str(e)}")
            return {"error": f"Error uploading image: {str(e)}"}

    data = json.dumps(
        {"prompt_id": prompt_id, "output": file_res.json().get("id")}
    ).encode("utf-8")

    retry_post(session, API_ML + "/set_serverless_output", data=data)

    return {
        "success": True,
        "output": file_res.json(),
        "prompt_id": prompt_id,
        "client_id": client_id,
        "job_id": prompt_id,
    }


def retry_file_upload(session, url, retries=5, files=None):
    for i in range(retries):
        print(f"runpod-worker-comfy - Attempt {i}/{retries} to post to {url}")
        try:
            response = session.post(url, files=files)

            if response.ok:
                return response
        except requests.RequestException:
            pass

        time.sleep(1)

    raise Exception("Error uploading")


def retry_post(session, url, retries=5, **kwargs):
    for i in range(retries):
        print(f"runpod-worker-comfy - Attempt {i}/{retries} to post to {url}")
        try:
            response = session.post(
                url, **kwargs, headers={"Content-Type": "application/json"}
            )

            if response.ok:
                return response
        except requests.RequestException:
            pass

        time.sleep(1)

    raise Exception("Error posting")


def download_file(file_id, auth_cookie):

    temp_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Content-Type": "application/json",
        "Cookie": f"{SESSION_COOKIE_NAME}={auth_cookie}",
    }

    url = f"{API_FILE}/preview/{file_id}?as_attachment=True"

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    content_disposition = response.headers.get("content-disposition")
    filename = content_disposition.split("filename=")[-1]
    local_path = f"{temp_dir}/{filename}"
    with open(local_path, "wb") as f:
        f.write(response.content)

    return local_path


# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
