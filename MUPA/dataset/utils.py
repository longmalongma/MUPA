# Modified from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py

import base64
import math
import warnings
from io import BytesIO

import decord
import numpy as np
import torch
from PIL import Image, ImageSequence
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import requests
from MUPA.constants import IGNORE_INDEX
from MUPA.conversation import get_conv

import random as rnd
from typing import Tuple, List, Union, Optional

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    # change order here to ensure not exceeding max_pixels
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    # resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
        ele: dict,
        total_frames: int,
        video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_gif(path):
    gif = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(gif):
        frames.append(np.array(frame.convert('RGB')))
    frames = np.stack(frames, axis=0)
    return frames


def _read_video_decord(ele: dict, ) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if video_path.endswith('.gif'):
        video = _read_video_gif(video_path)
        total_frames, video_fps = video.shape[0], ele.get('fps', FPS)
    else:
        vr = decord.VideoReader(video_path, num_threads=ele.get('num_threads', 0))
        # if 'video_start' in ele or 'video_end' in ele:
        #     raise NotImplementedError("not support start_pts and end_pts in decord for now.")
        total_frames, video_fps = len(vr), vr.get_avg_fps()

    # 1. re-calculate total frames
    # s = ele.get('video_start', 0)
    # e = ele.get('video_end', total_frames / video_fps)
    s = ele.get('video_start')
    s = 0 if s is None else s
    e = ele.get('video_end')
    e = total_frames / video_fps if e is None else e
    s_frame = max(0, round(s * video_fps))
    e_frame = min(round(e * video_fps), total_frames - 1)
    total_frames = (e - s) * video_fps

    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    # idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    # 2. generate frame ids
    idx = torch.linspace(s_frame, e_frame, nframes).round().long().tolist()  # 均匀采样
    assert len(idx) == nframes, (len(idx), nframes)

    if video_path.endswith('.gif'):
        video = video[idx]
    else:
        video = vr.get_batch(idx).asnumpy()

    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video


def _read_video_decord_v1(ele: dict, sampling: str = "uniform") -> Tuple[torch.Tensor, List[int], float]:
    """
    Read and sample video frames using decord.VideoReader, supporting multiple sampling strategies.

    Args:
        ele (dict): Video config dict with keys:
            - video: path or URL to the video file
            - video_start: (optional) start time in seconds
            - video_end: (optional) end time in seconds
            - fps or nframes: sampling target (mutually exclusive)
            - min_frames, max_frames: frame count limits (effective in fps mode)
            - num_threads: thread count for decord.VideoReader
        sampling (str): Sampling strategy: "random", "uniform", or "headtail".

    Returns:
        Tuple[torch.Tensor, List[int], float]:
            video_tensor: sampled video tensor of shape (C, T, H, W)
            indices: list of sampled frame indices
            video_fps: original video frame rate
    """
    video_path = ele["video"]

    # Load video or GIF
    if video_path.endswith('.gif'):
        video = _read_video_gif(video_path)
        total_frames = video.shape[0]
        video_fps = ele.get('fps', FPS)
    else:
        vr = decord.VideoReader(video_path, num_threads=ele.get("num_threads", 0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()

    # Determine frame range based on start/end times
    start_time = ele.get('video_start', 0)
    end_time = ele.get('video_end', total_frames / video_fps)
    s_frame = max(0, round(start_time * video_fps))
    e_frame = min(round(end_time * video_fps), total_frames)
    total_duration_frames = (end_time - start_time) * video_fps

    # Determine number of frames to sample
    nframes = smart_nframes(ele, total_frames=total_duration_frames, video_fps=video_fps)

    # Divide interval into equal segments
    intervals = np.linspace(s_frame, e_frame, num=nframes + 1).astype(int)
    ranges = [(intervals[i], intervals[i + 1]) for i in range(nframes)]

    # Generate sampling indices
    if sampling == "random":
        indices = [rng[0] if rng[0] == rng[1] else rnd.choice(range(rng[0], rng[1])) for rng in ranges]
    elif sampling == "headtail":
        half = nframes // 2
        mid = (e_frame - s_frame) // 2 + s_frame
        head_idxs = sorted(rnd.sample(range(s_frame, mid), half))
        tail_idxs = sorted(rnd.sample(range(mid, e_frame), nframes - half))
        indices = head_idxs + tail_idxs
    elif sampling == "uniform":
        indices = [min((r[0] + r[1]) // 2, total_frames - 1) for r in ranges]
    else:
        raise NotImplementedError(f"Sampling strategy '{sampling}' is not implemented.")

    # Pad indices if needed
    if len(indices) < nframes:
        indices += [indices[-1]] * (nframes - len(indices))
    assert len(indices) == nframes, (len(indices), nframes)

    # Fetch frames
    if video_path.endswith('.gif'):
        sampled = video[indices]
    else:
        sampled = vr.get_batch(indices).asnumpy()

    # Convert to torch tensor and reorder to (C, T, H, W)
    video_tensor = torch.tensor(sampled).permute(3, 0, 1, 2).float()
    return video_tensor, indices, video_fps


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, sanity_check=False) -> Union[
    torch.Tensor, List[Image.Image]]:
    """
    Load and resize video or image sequence based on provided config.

    Args:
        ele (dict): Config dict containing "video" key and optional resize params.
        image_factor (int): Scaling factor for image resizing.
        sanity_check (bool): If True, raise error if output tensor is all zeros.

    Returns:
        torch.Tensor or list[Image.Image]: Processed video tensor or list of images.
    """
    vid = ele["video"]

    if isinstance(vid, str):
        video = _read_video_decord(ele)
        nframes, _, h, w = video.shape

        # Compute resize bounds
        min_pix = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pix = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pix = max(min(VIDEO_MAX_PIXELS, total_pix / nframes * FRAME_FACTOR), int(min_pix * 1.05))
        max_pix = ele.get("max_pixels", max_pix)

        # Determine resize dimensions
        if "resized_height" in ele and "resized_width" in ele:
            rh, rw = smart_resize(ele["resized_height"], ele["resized_width"], factor=image_factor)
        else:
            rh, rw = smart_resize(h, w, factor=image_factor, min_pixels=min_pix, max_pixels=max_pix)

        # Resize video tensor
        video = transforms.functional.resize(
            video, [rh, rw], interpolation=InterpolationMode.BICUBIC, antialias=True
        ).float()

        if sanity_check and (video == 0).all():
            raise ValueError(f"Video '{vid}' contains all zeros")
        return video

    # Handle list of image frames
    assert isinstance(vid, (list, tuple))
    info = {k: v for k, v in ele.items() if k not in ("type", "video")}
    frames = [fetch_image({"image": img, **info}, size_factor=image_factor) for img in vid]
    nframes = ceil_by_factor(len(frames), FRAME_FACTOR)
    if len(frames) < nframes:
        frames.extend([frames[-1]] * (nframes - len(frames)))
    return frames


def fetch_video_v1(ele: dict, image_factor: int = IMAGE_FACTOR, sanity_check=False) -> Union[
    Tuple[torch.Tensor, List[int], float], List[Image.Image]]:
    """
    Variant of fetch_video that returns frame indices and original fps when applicable.
    """
    vid = ele["video"]
    if isinstance(vid, str):
        video, indices, orig_fps = _read_video_decord_v1(ele)
        nframes, _, h, w = video.shape

        # Compute resize bounds
        min_pix = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pix = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pix = max(min(VIDEO_MAX_PIXELS, total_pix / nframes * FRAME_FACTOR), int(min_pix * 1.05))
        max_pix = ele.get("max_pixels", max_pix)

        # Determine resize dimensions
        if "resized_height" in ele and "resized_width" in ele:
            rh, rw = smart_resize(ele["resized_height"], ele["resized_width"], factor=image_factor)
        else:
            rh, rw = smart_resize(h, w, factor=image_factor, min_pixels=min_pix, max_pixels=max_pix)

        # Resize video tensor
        video = transforms.functional.resize(
            video, [rh, rw], interpolation=InterpolationMode.BICUBIC, antialias=True
        ).float()

        if sanity_check and (video == 0).all():
            raise ValueError(f"Video '{vid}' contains all zeros")
        return video, indices, orig_fps

    # Handle list of image frames
    assert isinstance(vid, (list, tuple))
    info = {k: v for k, v in ele.items() if k not in ("type", "video")}
    frames = [fetch_image({"image": img, **info}, size_factor=image_factor) for img in vid]
    nframes = ceil_by_factor(len(frames), FRAME_FACTOR)
    if len(frames) < nframes:
        frames.extend([frames[-1]] * (nframes - len(frames)))
    return frames


def extract_vision_info(conversations: Union[list[dict], list[list[dict]]]) -> List[dict]:
    """
    Extract vision elements (image/video) from a list of conversation messages.
    """
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    vision_infos = []
    for convo in conversations:
        for msg in convo:
            content = msg.get("content")
            if isinstance(content, list):
                for ele in content:
                    if any(key in ele for key in ("image", "image_url", "video")) or ele.get("type") in (
                            "image", "image_url", "video"):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
        conversations: Union[list[dict], list[list[dict]]],
        sanity_check=False) -> Tuple[
    Optional[List[Image.Image]], Optional[List[Union[torch.Tensor, List[Image.Image]]]]]:
    """
    Load images and videos from conversation content.

    Returns:
        image_inputs: list of PIL Images or None
        video_inputs: list of video tensors or None
    """
    vision_infos = extract_vision_info(conversations)
    images, videos = [], []
    for info in vision_infos:
        if "image" in info or "image_url" in info:
            images.append(fetch_image(info))
        elif "video" in info:
            videos.append(fetch_video(info, sanity_check=sanity_check))
        else:
            raise ValueError("Content must contain 'image', 'image_url', or 'video'.")
    return (images or None, videos or None)


def process_vision_info_v1(
        conversations: Union[list[dict], list[list[dict]]],
        sanity_check=False) -> Tuple[
    Optional[List[Image.Image]], Optional[List[torch.Tensor]], Optional[List[int]], Optional[float]]:
    """
    Load images, videos, and return sampling indices and original fps for videos.
    """
    vision_infos = extract_vision_info(conversations)
    images, videos, indices, fps = [], [], None, None
    for info in vision_infos:
        if "image" in info or "image_url" in info:
            images.append(fetch_image(info))
        elif "video" in info:
            video, indices, fps = fetch_video_v1(info, sanity_check=sanity_check)
            videos.append(video)
        else:
            raise ValueError("Content must contain 'image', 'image_url', or 'video'.")
    return (images or None, videos or None, indices, fps)


def preprocess_chatml(input_ids, text, tokenizer):
    conv = get_conv('chatml')

    rounds = [m + conv.seps[0] for m in text.split(conv.seps[0])]
    assert (len(rounds) % 2 == 0) == (conv.system is not None)
    assert rounds[-1] == conv.seps[0]
    rounds = rounds[:-1]

    if conv.system is None:
        rounds = [''.join(rounds[i:i + 2]) for i in range(0, len(rounds), 2)]
    else:
        rounds = [''.join(rounds[:3])] + [''.join(rounds[i:i + 2]) for i in range(3, len(rounds), 2)]

    labels = input_ids.clone()

    sep = conv.seps[0] + conv.roles[1]
    cur_len = 0

    for i, rou in enumerate(rounds):
        if len(rou) == 0:
            break

        ins = sep.join(rou.split(sep)[:-1]) + sep

        rou_len = tokenizer(rou, return_length=True).length[0]
        ins_len = tokenizer(ins, return_length=True).length[0]

        labels[cur_len:cur_len + ins_len] = IGNORE_INDEX
        cur_len += rou_len

    # TODO: sometimes visual tokens are in the assistant round
    # <|vision_start|> <|vision_end|> <|vision_pad|> <|image_pad|> <|video_pad|>
    # for token_id in range(151652, 151657):
    #     labels[labels == token_id] = IGNORE_INDEX

    if labels.size(0) != cur_len:
        warnings.warn(f'Tokenization mismatch: {labels.size(0)} and {cur_len}')

    return labels


def preprocess(input_ids, text, tokenizer, conv_type):
    if conv_type == 'chatml':
        return preprocess_chatml(input_ids, text, tokenizer)
    else:
        raise ValueError(f'unknown conversation type: {conv_type}')
