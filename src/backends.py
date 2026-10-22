"""백엔드 추상화. 평가·측정 스크립트는 이 인터페이스만 본다.

입력 규약: encode_image 는 common.preprocess 가 만든 float32 [1,3,224,224] (CLIP 정규화 적용)을 받는다.
백엔드마다 모델의 입력 규약이 다를 수 있으므로, 변환이 필요하면 각 백엔드 안에서 한다.
예: Hailo CLIP 설정은 normalize_in_net=true 라 정규화 전 0~255 입력을 기대한다.
그 경우 DFC/HailoRT 백엔드가 정규화를 되돌려 넣어야 하며, 정규화한 값을 그대로 넣으면 이중 정규화가 된다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common as C


class Backend:
    code = "?"          # A0, A1, ...
    name = "?"

    def encode_image(self, x: np.ndarray) -> np.ndarray:
        """float32 [1,3,224,224] -> float32 [512] (정규화 전 임베딩)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def make_ort_session(model_path, threads: int = C.DEFAULT_THREADS, profile_prefix: str | None = None,
                     optimized_out: str | None = None):
    """측정 조건을 고정한 ORT 세션. 스레드 수는 반드시 명시한다."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    if optimized_out:
        so.optimized_model_filepath = optimized_out
    return ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])


class OnnxBackend(Backend):
    """ONNX Runtime CPU. FP32(A0), 동적 INT8(A1), 정적 INT8(A2, A3a, A3b) 공용."""
    name = "onnxruntime-cpu"

    def __init__(self, model_path, code: str = "A0", threads: int = C.DEFAULT_THREADS):
        self.model_path = Path(model_path)
        self.code = code
        self.threads = threads
        self.sess = make_ort_session(self.model_path, threads)
        self.input_name = self.sess.get_inputs()[0].name

    def encode_image(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0][0]


class DfcEmulatorBackend(Backend):
    """Hailo DFC 에뮬레이터 (H0 FP / H1·H2 양자화). GPU 데이 스크립트에서 구현한다.

    구현 시 주의: Hailo CLIP 은 normalize_in_net=true. 입력을 0~255 NHWC 로 되돌려 넣는다.
    """
    name = "hailo-dfc-emulator"

    def __init__(self, har_path, code: str = "H0"):
        raise NotImplementedError("DFC 환경(.venv-hailo)에서 구현 예정. docs/PROTOCOL.md 참조")


class HailoRTBackend(Backend):
    """본선 실기기 (B1 공식 HEF / B2 직접 컴파일한 HEF)."""
    name = "hailort"

    def __init__(self, hef_path, code: str = "B2"):
        raise NotImplementedError("본선(11/6 이후) 보드 수령 후 구현")
