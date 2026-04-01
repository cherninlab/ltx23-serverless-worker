# LTX-2.3 Serverless Worker

RunPod serverless worker for fast parallel LTX-2.3 experiments.

## Modes
- `ic_union`: `ltx_pipelines.ic_lora` + Union Control LoRA
- `ic_motion`: `ltx_pipelines.ic_lora` + Motion Track Control LoRA
- `distilled_i2v`: `ltx_pipelines.distilled` image-conditioned generation

## Required input fields
- `hf_token` (or env `HF_TOKEN`): Hugging Face token with access to Gemma and LTX assets
- `first_frame_url` or `first_frame_base64`

For IC modes also provide:
- `conditioning_video_url` or `conditioning_video_base64`

## Google Drive sync
Set both:
- `gdrive_remote_path`
- `rclone_conf_base64`

Worker uploads:
- output video
- logs folder (manifest, command, stdout/stderr, GPU telemetry, timing)

## Example job input (IC Union)
See `sample_request_ic_union.json`.
