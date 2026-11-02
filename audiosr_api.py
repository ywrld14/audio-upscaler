#!/usr/bin/env python3
"""AudioSR via HuggingFace Spaces API - no local GPU needed."""
import sys
from gradio_client import Client, handle_file

def upscale_audio(input_path: str, output_path: str):
    print(f"Connecting to AudioSR HuggingFace Space...")
    client = Client("haoheliu/audio-super-resolution")

    print(f"Uploading {input_path}...")
    result = client.predict(
        input_file=handle_file(input_path),
        guidance_scale=3.5,
        ddim_steps=50,
        seed=42,
        api_name="/predict"
    )

    print(f"Result: {result}")

    # Copy result to output path
    import shutil
    shutil.copy(result, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python audiosr_api.py <input> <output>")
        sys.exit(1)
    upscale_audio(sys.argv[1], sys.argv[2])
