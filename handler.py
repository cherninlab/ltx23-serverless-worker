import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gdown
import requests
import runpod
from huggingface_hub import hf_hub_download, snapshot_download


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _run(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    stdout_handle = open(stdout_path, "w", encoding="utf-8") if stdout_path else subprocess.PIPE
    stderr_handle = open(stderr_path, "w", encoding="utf-8") if stderr_path else subprocess.PIPE
    try:
        return subprocess.run(
            cmd,
            env=env,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    finally:
        if stdout_path:
            stdout_handle.close()
        if stderr_path:
            stderr_handle.close()


def _must_run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _download_to_path(url: str, dst: Path) -> None:
    if "drive.google.com" in url:
        res = gdown.download(url=url, output=str(dst), quiet=False, fuzzy=True)
        if res is None:
            raise RuntimeError(f"Failed to download Google Drive URL: {url}")
        return

    with requests.get(url, stream=True, timeout=(20, 600)) as resp:
        resp.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _decode_b64_asset(data: str, dst: Path) -> None:
    payload = data
    if "," in payload and payload.split(",", 1)[0].startswith("data:"):
        payload = payload.split(",", 1)[1]
    dst.write_bytes(base64.b64decode(payload))


def _fetch_asset(job_input: Dict[str, Any], key_prefix: str, ext: str, dst_dir: Path) -> Optional[Path]:
    url_key = f"{key_prefix}_url"
    b64_key = f"{key_prefix}_base64"
    url_value = job_input.get(url_key)
    b64_value = job_input.get(b64_key)
    if not url_value and not b64_value:
        return None
    out_path = dst_dir / f"{key_prefix}.{ext}"
    if url_value:
        _download_to_path(str(url_value), out_path)
    else:
        _decode_b64_asset(str(b64_value), out_path)
    return out_path


def _redacted_manifest(job_input: Dict[str, Any]) -> Dict[str, Any]:
    redacted = dict(job_input)
    for key in ["hf_token", "rclone_conf_base64"]:
        if key in redacted:
            redacted[key] = "<redacted>"
    return redacted


def _resolution(job_input: Dict[str, Any]) -> Tuple[int, int]:
    if "width" in job_input and "height" in job_input:
        return int(job_input["width"]), int(job_input["height"])

    preset = str(job_input.get("resolution_preset", "1080p")).lower()
    if preset == "4k":
        return 3840, 2176
    return 1920, 1088


def _ensure_models(job_input: Dict[str, Any], model_root: Path) -> Dict[str, str]:
    hf_token = str(job_input.get("hf_token", os.getenv("HF_TOKEN", ""))).strip()
    if not hf_token:
        raise RuntimeError("Missing hf_token: provide input.hf_token or HF_TOKEN env var")

    model_root.mkdir(parents=True, exist_ok=True)
    ltx_root = model_root / "LTX-2.3"
    gemma_root = model_root / "gemma-3-12b-it-qat-q4_0-unquantized"

    distilled_ckpt = hf_hub_download(
        repo_id="Lightricks/LTX-2.3",
        filename="ltx-2.3-22b-distilled.safetensors",
        local_dir=str(ltx_root),
        token=hf_token,
    )
    spatial_up = hf_hub_download(
        repo_id="Lightricks/LTX-2.3",
        filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        local_dir=str(ltx_root),
        token=hf_token,
    )

    if not gemma_root.exists() or not any(gemma_root.iterdir()):
        snapshot_download(
            repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
            local_dir=str(gemma_root),
            token=hf_token,
            resume_download=True,
        )

    union_lora = hf_hub_download(
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        filename="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        local_dir=str(model_root / "LTX-2.3-22b-IC-LoRA-Union-Control"),
        token=hf_token,
    )
    motion_lora = hf_hub_download(
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        filename="ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        local_dir=str(model_root / "LTX-2.3-22b-IC-LoRA-Motion-Track-Control"),
        token=hf_token,
    )

    return {
        "distilled_checkpoint": distilled_ckpt,
        "spatial_upsampler": spatial_up,
        "gemma_root": str(gemma_root),
        "union_lora": union_lora,
        "motion_lora": motion_lora,
    }


def _build_command(job_input: Dict[str, Any], paths: Dict[str, str], assets_dir: Path, output_video: Path) -> List[str]:
    py_bin = _which("python3") or _which("python")
    if not py_bin:
        raise RuntimeError("python not found")

    mode = str(job_input.get("mode", "ic_union")).lower()
    prompt = str(job_input.get("prompt", "A realistic talking person, preserving identity and natural texture."))
    seed = int(job_input.get("seed", 42))
    num_frames = int(job_input.get("num_frames", 121))
    steps = int(job_input.get("num_inference_steps", 20))
    width, height = _resolution(job_input)
    frame_rate = float(job_input.get("frame_rate", 24.0))

    common = [
        "--distilled-checkpoint-path",
        paths["distilled_checkpoint"],
        "--spatial-upsampler-path",
        paths["spatial_upsampler"],
        "--gemma-root",
        paths["gemma_root"],
        "--prompt",
        prompt,
        "--output-path",
        str(output_video),
        "--seed",
        str(seed),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num-frames",
        str(num_frames),
        "--frame-rate",
        str(frame_rate),
        "--num-inference-steps",
        str(steps),
    ]

    if bool(job_input.get("enhance_prompt", False)):
        common.append("--enhance-prompt")
    if bool(job_input.get("compile", False)):
        common.append("--compile")
    if "streaming_prefetch_count" in job_input:
        common.extend(["--streaming-prefetch-count", str(job_input["streaming_prefetch_count"])])
    if bool(job_input.get("fp8_cast", True)):
        common.extend(["--quantization", "fp8-cast"])

    frame_asset = assets_dir / "first_frame.png"
    if not frame_asset.exists():
        raise ValueError("first_frame_url or first_frame_base64 is required")

    if mode == "distilled_i2v":
        cmd = [py_bin, "-m", "ltx_pipelines.distilled"] + common
        cmd.extend(["--image", str(frame_asset), "0", str(job_input.get("image_strength", 1.0))])
        return cmd

    cond_video = assets_dir / "conditioning_video.mp4"
    if not cond_video.exists():
        raise ValueError("conditioning_video_url or conditioning_video_base64 is required for ic modes")

    lora_path = paths["union_lora"] if mode == "ic_union" else paths["motion_lora"]
    cmd = [py_bin, "-m", "ltx_pipelines.ic_lora"] + common
    cmd.extend(
        [
            "--video-conditioning",
            str(cond_video),
            str(job_input.get("conditioning_strength", 1.0)),
            "--image",
            str(frame_asset),
            "0",
            str(job_input.get("image_strength", 1.0)),
            "--lora",
            lora_path,
            str(job_input.get("lora_strength", 1.0)),
        ]
    )
    if bool(job_input.get("skip_stage_2", False)):
        cmd.append("--skip-stage-2")
    return cmd


def _start_gpu_monitor(log_file: Path) -> Optional[subprocess.Popen]:
    nvidia_smi = _which("nvidia-smi")
    if not nvidia_smi:
        return None
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("timestamp,index,name,util_gpu,util_mem,mem_used,mem_total,power_w\n")
    out_handle = open(log_file, "a", encoding="utf-8")
    return subprocess.Popen(
        [
            nvidia_smi,
            "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
            "-l",
            "2",
        ],
        stdout=out_handle,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _resolve_remote_target(remote_value: str, local_file: Path) -> str:
    target = str(remote_value).strip()
    if not target:
        raise ValueError("gdrive_remote_path is empty")
    if target.endswith("/"):
        return f"{target}{local_file.name}"
    if target.lower().endswith(".mp4"):
        return target
    return f"{target}/{local_file.name}"


def _upload_artifacts(job_dir: Path, out_video: Path, remote_base: str, rclone_conf_b64: str) -> Dict[str, str]:
    rclone = _which("rclone")
    if not rclone:
        raise RuntimeError("rclone binary not available")

    conf_bytes = base64.b64decode(rclone_conf_b64)
    logs_dir = job_dir / "logs"
    remote_video = _resolve_remote_target(remote_base, out_video)

    with tempfile.TemporaryDirectory(prefix="rclone_conf_") as td:
        conf = Path(td) / "rclone.conf"
        conf.write_bytes(conf_bytes)
        _must_run([rclone, "copyto", str(out_video), remote_video, "--config", str(conf)])

        remote_logs = _resolve_remote_target(remote_base, logs_dir).rstrip("/") + "/"
        _must_run([rclone, "copy", str(logs_dir), remote_logs, "--config", str(conf)])

    return {"video": remote_video, "logs": remote_logs}


def _handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})

    if bool(job_input.get("dry_run", False)):
        return {
            "status": "ready",
            "python": _which("python3") or _which("python"),
            "ffmpeg": _which("ffmpeg"),
            "rclone": _which("rclone"),
            "nvidia_smi": _which("nvidia-smi"),
        }

    with tempfile.TemporaryDirectory(prefix="ltx23_job_") as td:
        job_dir = Path(td)
        logs_dir = job_dir / "logs"
        assets_dir = job_dir / "assets"
        logs_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = logs_dir / "input_manifest.json"
        manifest_path.write_text(json.dumps(_redacted_manifest(job_input), indent=2), encoding="utf-8")

        first_frame = _fetch_asset(job_input, "first_frame", "png", assets_dir)
        if first_frame:
            (assets_dir / "first_frame.png").write_bytes(first_frame.read_bytes())

        cond_video = _fetch_asset(job_input, "conditioning_video", "mp4", assets_dir)
        if cond_video:
            (assets_dir / "conditioning_video.mp4").write_bytes(cond_video.read_bytes())

        model_root = Path(str(job_input.get("model_root", "/workspace/models")))
        paths = _ensure_models(job_input, model_root)

        output_name = str(job_input.get("output_name", f"ltx23_{uuid.uuid4().hex[:8]}.mp4"))
        output_video = job_dir / output_name
        cmd = _build_command(job_input, paths, assets_dir, output_video)
        (logs_dir / "command.txt").write_text(" ".join(cmd), encoding="utf-8")

        run_env = os.environ.copy()
        run_env["PYTORCH_CUDA_ALLOC_CONF"] = str(
            job_input.get("pytorch_cuda_alloc_conf", "expandable_segments:True")
        )

        gpu_log = logs_dir / "gpu_telemetry.csv"
        monitor = _start_gpu_monitor(gpu_log)
        t0 = time.time()
        proc = _run(
            cmd,
            env=run_env,
            stdout_path=logs_dir / "runner_stdout.log",
            stderr_path=logs_dir / "runner_stderr.log",
        )
        elapsed = time.time() - t0

        if monitor is not None:
            monitor.terminate()

        (logs_dir / "timing.json").write_text(
            json.dumps({"inference_seconds": round(elapsed, 3), "return_code": proc.returncode}, indent=2),
            encoding="utf-8",
        )

        if proc.returncode != 0:
            return {
                "status": "failed",
                "return_code": proc.returncode,
                "log_hint": "See runner_stdout.log and runner_stderr.log in uploaded logs",
            }

        response: Dict[str, Any] = {
            "status": "ok",
            "output_filename": output_video.name,
            "output_size_bytes": output_video.stat().st_size,
            "inference_seconds": round(elapsed, 3),
        }

        remote_path = job_input.get("gdrive_remote_path")
        if remote_path:
            rclone_conf_b64 = str(job_input.get("rclone_conf_base64", os.getenv("RCLONE_CONF_BASE64", "")))
            if not rclone_conf_b64.strip():
                raise ValueError("gdrive_remote_path provided but no rclone_conf_base64 supplied")
            uploaded = _upload_artifacts(job_dir, output_video, str(remote_path), rclone_conf_b64)
            response["gdrive_remote_video"] = uploaded["video"]
            response["gdrive_remote_logs"] = uploaded["logs"]

        if str(job_input.get("return_mode", "metadata")) == "base64":
            response["video_base64"] = base64.b64encode(output_video.read_bytes()).decode("utf-8")

        return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": _handler})
