from typing import Callable, Generator, Optional

import av
import numpy as np

from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs


def read_video(
        video_path: str, dt: float,
        video_start_time: float = 0.0,
        start_time: float = 0.0,
        img_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        thread_type: str = "AUTO",
        thread_count: int = 0,
        max_pad_frames: int = 10,
) -> Generator[np.ndarray, None, None]:
    frame = None
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        stream.thread_count = thread_count
        next_global_idx = 0
        for frame_idx, frame in enumerate(container.decode(stream)):
            frame_time = video_start_time + frame.time
            _, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(
                timestamps=[frame_time], start_time=start_time, dt=dt,
                next_global_idx=next_global_idx,
            )
            if global_idxs:
                img = frame.to_ndarray(format="rgb24")
                if img_transform is not None:
                    img = img_transform(img)
                for _ in global_idxs:
                    yield img
    if frame is not None:
        img = frame.to_ndarray(format="rgb24")
        if img_transform is not None:
            img = img_transform(img)
        for _ in range(max_pad_frames):
            yield img


class VideoRecorder:
    def __init__(self, fps, codec, input_pix_fmt, **kwargs):
        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        self._reset_state()

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    @classmethod
    def create_h264(
            cls, fps, codec="h264", input_pix_fmt="rgb24",
            output_pix_fmt="yuv420p", crf=18, profile="high", **kwargs):
        return cls(
            fps=fps, codec=codec, input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile": profile},
            **kwargs,
        )

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            self.stop()
        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        for key, value in self.kwargs.items():
            setattr(self.stream.codec_context, key, value)
        self.start_time = start_time

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")
        n_repeats = 1
        if self.start_time is not None:
            local_idxs, _, self.next_global_idx = get_accumulate_timestamp_idxs(
                timestamps=[frame_time], start_time=self.start_time,
                dt=1 / self.fps, next_global_idx=self.next_global_idx,
            )
            n_repeats = len(local_idxs)
        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            height, width, _ = img.shape
            self.stream.width = width
            self.stream.height = height
        assert img.shape == self.shape
        assert img.dtype == self.dtype
        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for _ in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self._reset_state()
