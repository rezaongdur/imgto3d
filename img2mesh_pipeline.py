#!/usr/bin/env python3
"""
End-to-End 2D Image(s) to Calibrated 3D Solid Mesh Reconstruction Pipeline
(Dengan Deteksi Otomatis Marker ArUco cv2.aruco, Multi-Color CAD Layer Extrusion,
Kalibrasi Intrinsik Kamera, Homografi 4-Titik, Open3D Interactive Preview, dan --help).

Tahapan Pipeline:
  - Tahap 0A: Kalibrasi & Undistorsi Matriks Intrinsik Kamera (Brown-Conrady / Zhang)
  - Tahap 0B: Deteksi Otomatis Marker ArUco (cv2.aruco) & Rektifikasi Homografi 4-Titik (1:1)
  - Tahap 1 : Preprocessing Adaptif (CLAHE + Bilateral Filter) & Kalibrasi Skala Metrik
  - Tahap 2 : Rekonstruksi 3D Mode A (Single-Layer atau Multi-Color Multi-Height CAD)
              atau Mode B (Depth Anything V2 / MiDaS / Heightmap + Poisson / Alpha Shapes)
  - Tahap 3 : Mesh Cleaning & Topology Optimization (Watertight, Manifold, Taubin)
  - Tahap 4 : Ekspor Multi-Format (.STL, .OBJ + Multi-Material .MTL, .DAE SketchUp, .PLY)
  - Tahap 5 : Validasi Dimensi Fisik, Integritas Topologi Euler, & Pratinjau 3D Open3D
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import make_valid

__version__ = "2.5.0"

# ==============================================================================
# KONFIGURASI LOGGING & CUSTOM EXCEPTIONS
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Img2Mesh3D")


class PipelineError(Exception):
    """Base exception untuk seluruh kegagalan dalam pipeline rekonstruksi 3D."""


class CalibrationError(PipelineError):
    """Dilempar apabila parameter kalibrasi kamera, ArUco, homografi, atau skala tidak valid."""


class GeometryExtractionError(PipelineError):
    """Dilempar apabila tidak ada kontur atau permukaan valid yang diekstrak."""


class MeshTopologyError(PipelineError):
    """Dilempar apabila perbaikan topologi gagal menghasilkan mesh valid."""


# ==============================================================================
# UTILITAS KOLORIMETRI PERSEPTUAL (RGB <-> CIE L*a*b*)
# ==============================================================================


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Mengonversi string warna HEX ('#FF0000' atau 'FF0000') menjadi tuple (R, G, B)."""
    cleaned = hex_str.strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(ch * 2 for ch in cleaned)
    if len(cleaned) != 6:
        raise CalibrationError(f"Format warna HEX tidak valid: '{hex_str}'. Gunakan '#RRGGBB'.")
    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
        return (r, g, b)
    except ValueError as exc:
        raise CalibrationError(f"Nilai heksadesimal warna tidak valid: '{hex_str}'.") from exc


def rgb_array_to_cie_lab(rgb_arr: np.ndarray) -> np.ndarray:
    """
    Mengonversi array uint8 RGB (..., 3) ke skala perseptual CIE L*a*b* standar:
      L* in [0, 100], a* in [-128, 127], b* in [-128, 127].
    """
    orig_shape = rgb_arr.shape
    flat_rgb = rgb_arr.reshape(-1, 1, 3).astype(np.uint8)
    lab_cv = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_cie = np.empty_like(lab_cv)
    lab_cie[:, :, 0] = lab_cv[:, :, 0] * (100.0 / 255.0)
    lab_cie[:, :, 1] = lab_cv[:, :, 1] - 128.0
    lab_cie[:, :, 2] = lab_cv[:, :, 2] - 128.0
    return lab_cie.reshape(orig_shape)


def cie_lab_to_rgb_tuple(lab_vec: np.ndarray) -> Tuple[int, int, int]:
    """Mengonversi vektor tunggal CIE L*a*b* standar kembali ke tuple uint8 (R, G, B)."""
    l_cv = np.clip(round(float(lab_vec[0]) * (255.0 / 100.0)), 0, 255)
    a_cv = np.clip(round(float(lab_vec[1]) + 128.0), 0, 255)
    b_cv = np.clip(round(float(lab_vec[2]) + 128.0), 0, 255)
    pixel_lab = np.array([[[l_cv, a_cv, b_cv]]], dtype=np.uint8)
    rgb = cv2.cvtColor(pixel_lab, cv2.COLOR_LAB2RGB)[0, 0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


# ==============================================================================
# STRUKTUR DATA KONFIGURASI & LAPORAN VALIDASI
# ==============================================================================

UnitType = Literal["mm", "cm", "m"]
ModeType = Literal["cad", "surface"]
MultiFileStrategy = Literal["batch", "stack", "merge"]
DepthBackend = Literal["heightmap", "depth-anything", "midas"]
SurfaceMethod = Literal["poisson", "alpha", "grid-solid"]
ArucoAnchorType = Literal["outer", "inner", "center"]

SUPPORTED_ARUCO_DICTS: Tuple[str, ...] = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
)


@dataclass
class ArucoConfig:
    """Konfigurasi deteksi otomatis marker ArUco (cv2.aruco) untuk homografi & skala 1:1."""

    dict_name: str = "DICT_4X4_50"
    board_real_size: Optional[Tuple[float, float]] = None  # (width, height) untuk mode 4-sudut
    marker_real_size: Optional[float] = None               # sisi S untuk mode marker tunggal
    corner_ids: Tuple[int, int, int, int] = (0, 1, 2, 3)   # [TL, TR, BR, BL]
    target_single_id: Optional[int] = None
    anchor_mode: ArucoAnchorType = "outer"
    keep_full_canvas: bool = True
    save_debug_overlay: bool = False

    def __post_init__(self) -> None:
        if self.dict_name not in SUPPORTED_ARUCO_DICTS:
            raise CalibrationError(f"Kamus ArUco tidak didukung: '{self.dict_name}'.")
        if self.board_real_size is not None:
            w, h = self.board_real_size
            if w <= 0.0 or h <= 0.0:
                raise CalibrationError("--aruco-board-size harus memiliki lebar dan tinggi > 0.")
        if self.marker_real_size is not None and self.marker_real_size <= 0.0:
            raise CalibrationError("--aruco-marker-size harus lebih besar dari 0.")
        if len(set(self.corner_ids)) != 4:
            raise CalibrationError("--aruco-ids harus berisi 4 ID marker unik berurutan [TL,TR,BR,BL].")

    @property
    def is_enabled(self) -> bool:
        return self.board_real_size is not None or self.marker_real_size is not None


@dataclass
class ColorLayerRule:
    """Aturan ekstrusi 3D untuk satu segmen warna pada denah 2D."""

    name: str
    rgb: Tuple[int, int, int]
    extrude_height: float
    z_base: float = 0.0
    lab: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.extrude_height <= 0.0:
            raise CalibrationError(
                f"Tinggi ekstrusi untuk layer '{self.name}' harus lebih besar dari 0."
            )
        rgb_np = np.array([[self.rgb]], dtype=np.uint8)
        self.lab = rgb_array_to_cie_lab(rgb_np)[0, 0]

    @property
    def hex_code(self) -> str:
        return f"#{self.rgb[0]:02X}{self.rgb[1]:02X}{self.rgb[2]:02X}"


@dataclass
class ColorCadConfig:
    """Konfigurasi segmentasi warna otomatis / eksplisit pada Mode A."""

    rules: List[ColorLayerRule] = field(default_factory=list)
    auto_num_colors: Optional[int] = None
    auto_heights: Optional[List[float]] = None
    delta_e_tolerance: float = 22.0
    bg_delta_e_threshold: float = 15.0

    @property
    def is_enabled(self) -> bool:
        return len(self.rules) > 0 or (self.auto_num_colors is not None and self.auto_num_colors >= 1)


@dataclass
class CameraIntrinsicsConfig:
    """Parameter matriks intrinsik kamera K dan koefisien distorsi lensa."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    calib_image_size: Optional[Tuple[int, int]] = None
    alpha: float = 0.0

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).ravel()
        if self.camera_matrix.shape != (3, 3):
            raise CalibrationError("camera_matrix harus berukuran 3x3.")
        if self.dist_coeffs.size < 4:
            raise CalibrationError("dist_coeffs minimal memiliki 4 elemen [k1, k2, p1, p2].")

    @classmethod
    def from_json(cls, json_path: Path, alpha: float = 0.0) -> "CameraIntrinsicsConfig":
        if not json_path.exists():
            raise FileNotFoundError(f"File kalibrasi kamera tidak ditemukan: {json_path}")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        k_mat = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"], dtype=np.float64)
        img_size = tuple(data["image_size"]) if "image_size" in data and data["image_size"] else None
        return cls(camera_matrix=k_mat, dist_coeffs=dist, calib_image_size=img_size, alpha=alpha)

    @classmethod
    def from_manual_params(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_coeffs: Sequence[float],
        alpha: float = 0.0,
    ) -> "CameraIntrinsicsConfig":
        if fx <= 0 or fy <= 0:
            raise CalibrationError("Focal length fx dan fy harus bernilai positif.")
        k_mat = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return cls(camera_matrix=k_mat, dist_coeffs=np.array(dist_coeffs, dtype=np.float64), alpha=alpha)

    def to_json(self, json_path: Path, reprojection_error: Optional[float] = None) -> None:
        payload = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.calib_image_size) if self.calib_image_size else None,
            "rms_reprojection_error_px": reprojection_error,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class HomographyConfig:
    """Konfigurasi rektifikasi perspektif 4-titik ke bidang ortografis 1:1."""

    src_points: np.ndarray
    rect_real_width: float
    rect_real_height: float
    keep_full_canvas: bool = True
    target_ppu: Optional[float] = None

    def __post_init__(self) -> None:
        self.src_points = np.asarray(self.src_points, dtype=np.float32)
        if self.src_points.shape != (4, 2):
            raise CalibrationError(
                "src_points harus berisi tepat 4 titik koordinat (x, y) berurutan [TL, TR, BR, BL]."
            )
        if self.rect_real_width <= 0.0 or self.rect_real_height <= 0.0:
            raise CalibrationError("Dimensi fisik referensi homografi (width & height) harus > 0.")
        area = cv2.contourArea(self.src_points)
        if area < 100.0:
            raise CalibrationError(
                "Luas poligon 4 titik homografi terlalu kecil atau kolinear (degenerasi)."
            )


@dataclass
class CalibrationConfig:
    """Parameter kalibrasi piksel ke dimensi dunia nyata."""

    ref_real_size: float
    ref_pixels: float
    unit: UnitType = "mm"
    extrude_height: float = 5.0
    invert_polarity: bool = False

    def __post_init__(self) -> None:
        if self.ref_real_size <= 0.0:
            raise CalibrationError("ref_real_size harus lebih besar dari 0.")
        if self.ref_pixels <= 0.0:
            raise CalibrationError("ref_pixels harus lebih besar dari 0.")
        if self.extrude_height <= 0.0:
            raise CalibrationError("extrude_height harus lebih besar dari 0.")
        if self.unit not in ("mm", "cm", "m"):
            raise CalibrationError(f"Satuan tidak didukung: {self.unit}")

    @property
    def scale_per_pixel(self) -> float:
        return self.ref_real_size / self.ref_pixels

    @property
    def meters_per_unit(self) -> float:
        return {"mm": 0.001, "cm": 0.01, "m": 1.0}[self.unit]


@dataclass
class PreviewConfig:
    """Konfigurasi pratinjau interaktif Open3D Visualizer pada tahap validasi akhir."""

    enabled: bool = False
    wireframe: bool = False
    show_back_face: bool = True
    show_guides: bool = True
    confirm_before_export: bool = False
    window_width: int = 1280
    window_height: int = 800


@dataclass
class PipelineConfig:
    """Konfigurasi lengkap seluruh tahapan pipeline."""

    input_paths: List[Path]
    output_dir: Path
    mode: ModeType = "cad"
    multi_strategy: MultiFileStrategy = "batch"
    calibration: CalibrationConfig = field(
        default_factory=lambda: CalibrationConfig(100.0, 100.0, "mm", 5.0)
    )
    # Tahap 0: Camera Intrinsics, ArUco Auto-Calibration, & Homography Perspective Rectification
    camera_intrinsics: Optional[CameraIntrinsicsConfig] = None
    aruco: Optional[ArucoConfig] = None
    homography: Optional[HomographyConfig] = None
    interactive_pick_points: bool = False
    # Tahap 1: Preprocessing
    clahe_clip: float = 2.0
    bilateral_d: int = 9
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    adaptive_block_size: int = 35
    adaptive_c: float = 5.0
    use_otsu: bool = True
    # Tahap 2A: CAD & Multi-Color Layer Parameters
    rdp_epsilon_px: float = 0.35
    min_contour_area_px: float = 25.0
    layer_gap: float = 0.0
    color_cad: ColorCadConfig = field(default_factory=ColorCadConfig)
    # Tahap 2B: Surface / Relief Parameters
    depth_backend: DepthBackend = "heightmap"
    surface_method: SurfaceMethod = "poisson"
    poisson_depth: int = 9
    poisson_trim_quantile: float = 0.02
    alpha_radius_factor: float = 2.5
    close_solid_base: bool = True
    base_thickness: float = 1.0
    max_grid_dim: int = 512
    # Tahap 3: Mesh Cleaning & Optimization
    taubin_iterations: int = 5
    target_reduction: float = 0.0
    enforce_watertight: bool = True
    # Tahap 5: Interactive Preview
    preview: PreviewConfig = field(default_factory=PreviewConfig)


@dataclass
class ValidationReport:
    """Hasil audit geometris dan topologis model 3D akhir."""

    model_name: str
    unit: str
    length_x: float
    width_y: float
    height_z: float
    num_vertices: int
    num_faces: int
    surface_area: float
    volume: Optional[float]
    is_watertight: bool
    is_edge_manifold: bool
    is_vertex_manifold: bool
    is_winding_consistent: bool
    exported_files: Dict[str, Path]


# ==============================================================================
# TAHAP 0A: KALIBRASI MATRIKS INTRINSIK & UNDISTORSI LENSA KAMERA
# ==============================================================================


class CameraIntrinsicsCalibrator:
    """Kalibrasi kamera Zhang (chessboard) dan koreksi distorsi lensa."""

    @staticmethod
    def calibrate_from_chessboard(
        image_paths: Sequence[Path],
        board_size: Tuple[int, int],
        square_size: float,
        save_path: Path,
    ) -> CameraIntrinsicsConfig:
        if square_size <= 0:
            raise CalibrationError("square_size papan catur harus lebih besar dari 0.")

        cols, rows = board_size
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size)

        obj_points: List[np.ndarray] = []
        img_points: List[np.ndarray] = []
        img_shape: Optional[Tuple[int, int]] = None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)

        for path in image_paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Melewati file kalibrasi tidak valid: %s", path)
                continue

            h, w = img.shape[:2]
            if img_shape is None:
                img_shape = (w, h)
            elif img_shape != (w, h):
                logger.warning("Resolusi %s berbeda dari citra kalibrasi pertama, dilewati.", path.name)
                continue

            found, corners = cv2.findChessboardCornersSB(
                img, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
            )
            if not found:
                found, corners = cv2.findChessboardCorners(img, (cols, rows), None)
                if found and corners is not None:
                    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

            if found and corners is not None:
                obj_points.append(objp)
                img_points.append(corners)
                logger.info("Sudut papan catur terdeteksi pada: %s", path.name)
            else:
                logger.warning("Papan catur tidak terdeteksi pada: %s", path.name)

        if len(obj_points) < 3 or img_shape is None:
            raise CalibrationError(
                f"Minimal dibutuhkan 3 citra papan catur valid untuk kalibrasi (ditemukan: {len(obj_points)})."
            )

        rms_error, k_mat, dist_coeffs, _, _ = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )
        logger.info("Kalibrasi Kamera Selesai | RMS Reprojection Error: %.4f piksel", rms_error)

        cfg = CameraIntrinsicsConfig(
            camera_matrix=k_mat,
            dist_coeffs=dist_coeffs,
            calib_image_size=img_shape,
        )
        cfg.to_json(save_path, reprojection_error=float(rms_error))
        logger.info("Parameter intrinsik disimpan ke: %s", save_path.resolve())
        return cfg

    @staticmethod
    def undistort_image(
        img_bgr: np.ndarray, intrinsics: CameraIntrinsicsConfig
    ) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        k_mat = intrinsics.camera_matrix.copy()

        if intrinsics.calib_image_size is not None:
            cal_w, cal_h = intrinsics.calib_image_size
            if (cal_w, cal_h) != (w, h):
                sx = w / float(cal_w)
                sy = h / float(cal_h)
                k_mat[0, 0] *= sx
                k_mat[0, 2] *= sx
                k_mat[1, 1] *= sy
                k_mat[1, 2] *= sy
                logger.info(
                    "Menyesuaikan matriks intrinsik K dari (%dx%d) ke resolusi (%dx%d).",
                    cal_w, cal_h, w, h,
                )

        new_k, _ = cv2.getOptimalNewCameraMatrix(
            k_mat, intrinsics.dist_coeffs, (w, h), alpha=intrinsics.alpha, newImgSize=(w, h)
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            k_mat, intrinsics.dist_coeffs, None, new_k, (w, h), m1type=cv2.CV_32FC1
        )
        return cv2.remap(
            img_bgr, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
        )


# ==============================================================================
# TAHAP 0B-1: DETEKSI OTOMATIS MARKER ARUCO (cv2.aruco) UNTUK HOMOGRAFI & SKALA
# ==============================================================================


class ArucoAutoCalibrator:
    """
    Mendeteksi penanda fiducial ArUco (cv2.aruco) secara sub-piksel untuk:
      1. Ekstraksi otomatis 4 titik sudut referensi homografi [TL, TR, BR, BL], dan
      2. Kalibrasi skala metrik 1:1 dunia nyata tanpa klik manual.
    """

    @staticmethod
    def _get_dictionary(dict_name: str) -> object:
        if not hasattr(cv2, "aruco"):
            raise CalibrationError(
                "Modul 'cv2.aruco' tidak tersedia. Pasang 'opencv-contrib-python' (pip install opencv-contrib-python)."
            )
        dict_id = getattr(cv2.aruco, dict_name, None)
        if dict_id is None:
            raise CalibrationError(f"Kamus ArUco tidak ditemukan di OpenCV: '{dict_name}'.")
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            return cv2.aruco.getPredefinedDictionary(dict_id)
        return cv2.aruco.Dictionary_get(dict_id)

    @staticmethod
    def _create_detector_params() -> object:
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 40
            params.cornerRefinementMinAccuracy = 0.01
        return params

    @classmethod
    def detect_markers(
        cls, img_bgr: np.ndarray, dict_name: str
    ) -> Dict[int, np.ndarray]:
        """
        Mendeteksi semua marker ArUco dalam citra dan mengembalikan dictionary:
          { marker_id: ndarray shape (4, 2) float32 [TL, TR, BR, BL] }.
        """
        aruco_dict = cls._get_dictionary(dict_name)
        params = cls._create_detector_params()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners_list, ids, _ = detector.detectMarkers(gray)
        else:
            corners_list, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        # Coba sekali lagi dengan CLAHE kontras jika pencahayaan tidak merata
        if ids is None or len(ids) == 0:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners_list, ids, _ = detector.detectMarkers(gray_eq)
            else:
                corners_list, ids, _ = cv2.aruco.detectMarkers(gray_eq, aruco_dict, parameters=params)

        if ids is None or len(ids) == 0:
            return {}

        detected_map: Dict[int, np.ndarray] = {}
        for idx, raw_id in enumerate(ids.flatten()):
            m_id = int(raw_id)
            pts = np.asarray(corners_list[idx], dtype=np.float32).reshape(4, 2)
            # Jika terdapat duplikat ID, pilih yang memiliki luas poligon lebih besar
            if m_id not in detected_map or cv2.contourArea(pts) > cv2.contourArea(detected_map[m_id]):
                detected_map[m_id] = pts

        return detected_map

    @classmethod
    def build_homography_from_aruco(
        cls,
        img_bgr: np.ndarray,
        aruco_cfg: ArucoConfig,
        unit: str,
        debug_output_path: Optional[Path] = None,
    ) -> HomographyConfig:
        """
        Menghasilkan HomographyConfig secara otomatis dari marker ArUco terdeteksi.
        """
        detected = cls.detect_markers(img_bgr, aruco_cfg.dict_name)
        found_ids = sorted(detected.keys())

        if not detected:
            raise CalibrationError(
                f"Tidak ada marker ArUco ({aruco_cfg.dict_name}) yang terdeteksi pada citra! "
                "Periksa kamus --aruco-dict, pencahayaan, atau pastikan marker terlihat penuh."
            )

        logger.info("ArUco Terdeteksi (%s) | IDs ditemukan: %s", aruco_cfg.dict_name, found_ids)

        # Kasus 1: Mode 4-Sudut Reference Board (--aruco-board-size WxH)
        if aruco_cfg.board_real_size is not None:
            id_tl, id_tr, id_br, id_bl = aruco_cfg.corner_ids
            required_ids = [id_tl, id_tr, id_br, id_bl]
            missing = [mid for mid in required_ids if mid not in detected]
            if missing:
                raise CalibrationError(
                    f"Marker ArUco sudut referensi tidak lengkap! ID dibutuhkan: {required_ids}, "
                    f"ID hilang: {missing}, ID terdeteksi: {found_ids}."
                )

            c_tl = detected[id_tl]
            c_tr = detected[id_tr]
            c_br = detected[id_br]
            c_bl = detected[id_bl]

            if aruco_cfg.anchor_mode == "outer":
                src_pts = np.array([c_tl[0], c_tr[1], c_br[2], c_bl[3]], dtype=np.float32)
            elif aruco_cfg.anchor_mode == "inner":
                src_pts = np.array([c_tl[2], c_tr[3], c_br[0], c_bl[1]], dtype=np.float32)
            else:  # center
                src_pts = np.array(
                    [c_tl.mean(axis=0), c_tr.mean(axis=0), c_br.mean(axis=0), c_bl.mean(axis=0)],
                    dtype=np.float32,
                )

            real_w, real_h = aruco_cfg.board_real_size
            logger.info(
                "Kalibrasi ArUco 4-Sudut Berhasil | IDs=%s | Anchor='%s' | Bidang Fisik=%.3fx%.3f %s",
                required_ids,
                aruco_cfg.anchor_mode,
                real_w,
                real_h,
                unit,
            )

        # Kasus 2: Mode Marker Tunggal (--aruco-marker-size S)
        elif aruco_cfg.marker_real_size is not None:
            if aruco_cfg.target_single_id is not None:
                if aruco_cfg.target_single_id not in detected:
                    raise CalibrationError(
                        f"Marker ArUco target ID={aruco_cfg.target_single_id} tidak ditemukan "
                        f"(ID terdeteksi: {found_ids})."
                    )
                chosen_id = aruco_cfg.target_single_id
            else:
                # Pilih marker dengan luas piksel terbesar agar presisi sub-piksel maksimal
                chosen_id = max(found_ids, key=lambda mid: float(cv2.contourArea(detected[mid])))

            src_pts = detected[chosen_id].astype(np.float32)  # [TL, TR, BR, BL]
            real_w = float(aruco_cfg.marker_real_size)
            real_h = float(aruco_cfg.marker_real_size)
            logger.info(
                "Kalibrasi ArUco Marker Tunggal Berhasil | ID=%d | Ukuran Sisi=%.3f %s",
                chosen_id,
                real_w,
                unit,
            )
        else:
            raise CalibrationError(
                "Isi --aruco-board-size atau --aruco-marker-size untuk mengaktifkan kalibrasi ArUco."
            )

        if debug_output_path is not None:
            cls._save_debug_overlay(img_bgr, detected, src_pts, debug_output_path)

        return HomographyConfig(
            src_points=src_pts,
            rect_real_width=real_w,
            rect_real_height=real_h,
            keep_full_canvas=aruco_cfg.keep_full_canvas,
        )

    @staticmethod
    def _save_debug_overlay(
        img_bgr: np.ndarray,
        detected_map: Dict[int, np.ndarray],
        homography_pts: np.ndarray,
        save_path: Path,
    ) -> None:
        vis = img_bgr.copy()
        for m_id, pts in detected_map.items():
            pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts_i], isClosed=True, color=(0, 220, 0), thickness=2)
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.putText(
                vis,
                f"ID:{m_id}",
                (cx - 24, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        h_pts_i = np.round(homography_pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [h_pts_i], isClosed=True, color=(0, 180, 255), thickness=3)
        labels = ["TL", "TR", "BR", "BL"]
        for idx, (px, py) in enumerate(homography_pts):
            cv2.circle(vis, (int(px), int(py)), 7, (0, 0, 255), -1)
            cv2.putText(
                vis,
                labels[idx],
                (int(px) + 8, int(py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis)
        logger.info("Visualisasi deteksi ArUco disimpan ke: %s", save_path.resolve())

    @classmethod
    def generate_calibration_sheet(
        cls,
        output_path: Path,
        dict_name: str = "DICT_4X4_50",
        board_size_units: Tuple[float, float] = (210.0, 297.0),
        marker_size_units: float = 30.0,
        corner_ids: Tuple[int, int, int, int] = (0, 1, 2, 3),
        unit: str = "mm",
        ppu: float = 8.0,
    ) -> None:
        """
        Membuat lembar kalibrasi PNG beresolusi tinggi dengan 4 marker ArUco di sudut
        bidang referensi [TL, TR, BR, BL] (anchor='outer') siap cetak.
        """
        aruco_dict = cls._get_dictionary(dict_name)
        board_w, board_h = board_size_units
        if marker_size_units * 2.5 >= min(board_w, board_h):
            raise CalibrationError(
                "Ukuran marker terlalu besar dibandingkan ukuran papan referensi."
            )

        margin_units = max(10.0, marker_size_units * 0.4)
        canvas_w = int(round((board_w + 2.0 * margin_units) * ppu))
        canvas_h = int(round((board_h + 2.0 * margin_units) * ppu))
        marker_px = max(32, int(round(marker_size_units * ppu)))
        margin_px = int(round(margin_units * ppu))
        board_w_px = int(round(board_w * ppu))
        board_h_px = int(round(board_h * ppu))

        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        # Gambar garis batas luar bidang referensi (anchor='outer')
        x0, y0 = margin_px, margin_px
        x1, y1 = margin_px + board_w_px, margin_px + board_h_px
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)

        positions = [
            (corner_ids[0], x0, y0, "TL"),
            (corner_ids[1], x1 - marker_px, y0, "TR"),
            (corner_ids[2], x1 - marker_px, y1 - marker_px, "BR"),
            (corner_ids[3], x0, y1 - marker_px, "BL"),
        ]

        for m_id, px, py, corner_lbl in positions:
            if hasattr(cv2.aruco, "generateImageMarker"):
                m_gray = cv2.aruco.generateImageMarker(aruco_dict, int(m_id), marker_px)
            else:
                m_gray = cv2.aruco.drawMarker(aruco_dict, int(m_id), marker_px)
            m_bgr = cv2.cvtColor(m_gray, cv2.COLOR_GRAY2BGR)
            canvas[py : py + marker_px, px : px + marker_px] = m_bgr

            txt_y = py - 8 if corner_lbl in ("TL", "TR") else py + marker_px + 20
            cv2.putText(
                canvas,
                f"{corner_lbl} (ID {m_id})",
                (px, max(18, min(canvas_h - 8, txt_y))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 60, 60),
                1,
            )

        info_text = (
            f"Img2Mesh3D ArUco Calibration Sheet | {dict_name} | "
            f"Outer Board: {board_w}x{board_h} {unit} | IDs: {list(corner_ids)}"
        )
        cv2.putText(
            canvas,
            info_text,
            (x0, canvas_h - max(10, margin_px // 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 80, 80),
            1,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)
        logger.info(
            "Lembar kalibrasi ArUco 4-sudut dibuat di: %s (%dx%d px | Outer=%gx%g %s)",
            output_path.resolve(),
            canvas_w,
            canvas_h,
            board_w,
            board_h,
            unit,
        )


# ==============================================================================
# TAHAP 0B-2: KOREKSI PERSPEKTIF HOMOGRAFI 4-TITIK (METRIC RECTIFICATION)
# ==============================================================================


class HomographyPerspectiveRectifier:
    """Meluruskan foto miring menjadi tampak atas ortografis 1:1 via homografi 4-titik."""

    @staticmethod
    def interactive_pick_4_points(
        img_bgr: np.ndarray,
        window_title: str = "Pilih 4 Sudut Referensi (TL -> TR -> BR -> BL)",
    ) -> np.ndarray:
        display = img_bgr.copy()
        h, w = display.shape[:2]
        max_disp = 1200.0
        disp_scale = min(1.0, max_disp / float(max(h, w)))
        if disp_scale < 1.0:
            display = cv2.resize(
                display, (int(w * disp_scale), int(h * disp_scale)), interpolation=cv2.INTER_AREA
            )

        clicked: List[Tuple[float, float]] = []
        labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
                orig_x = x / disp_scale
                orig_y = y / disp_scale
                clicked.append((orig_x, orig_y))

                cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    labels[len(clicked) - 1],
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                if len(clicked) > 1:
                    p_prev = (int(clicked[-2][0] * disp_scale), int(clicked[-2][1] * disp_scale))
                    cv2.line(display, p_prev, (x, y), (0, 220, 255), 2)
                if len(clicked) == 4:
                    p_first = (int(clicked[0][0] * disp_scale), int(clicked[0][1] * disp_scale))
                    cv2.line(display, (x, y), p_first, (0, 220, 255), 2)
                cv2.imshow(window_title, display)

        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window_title, on_mouse)
            cv2.imshow(window_title, display)
            logger.info(
                "Klik 4 sudut referensi pada jendela gambar (TL -> TR -> BR -> BL), lalu tekan tombol apa saja."
            )

            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    raise CalibrationError("Pemilihan 4 titik homografi dibatalkan pengguna (ESC).")
                if len(clicked) == 4 and key != 255:
                    break
        except cv2.error as exc:
            raise CalibrationError(
                f"Gagal membuka GUI OpenCV untuk --pick-points (gunakan --aruco-* atau --homography-points): {exc}"
            ) from exc
        finally:
            cv2.destroyAllWindows()

        if len(clicked) != 4:
            raise CalibrationError("Harus memilih tepat 4 titik referensi sebelum melanjutkan.")

        return np.array(clicked, dtype=np.float32)

    @staticmethod
    def rectify(
        img_bgr: np.ndarray, h_cfg: HomographyConfig
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        src = h_cfg.src_points
        top_w_px = float(np.linalg.norm(src[1] - src[0]))
        bot_w_px = float(np.linalg.norm(src[2] - src[3]))
        left_h_px = float(np.linalg.norm(src[3] - src[0]))
        right_h_px = float(np.linalg.norm(src[2] - src[1]))

        if h_cfg.target_ppu is not None and h_cfg.target_ppu > 0:
            ppu = float(h_cfg.target_ppu)
        else:
            ppu_w = max(top_w_px, bot_w_px) / h_cfg.rect_real_width
            ppu_h = max(left_h_px, right_h_px) / h_cfg.rect_real_height
            ppu = max(ppu_w, ppu_h)

        dst_w = h_cfg.rect_real_width * ppu
        dst_h = h_cfg.rect_real_height * ppu

        dst_ref = np.array(
            [[0.0, 0.0], [dst_w, 0.0], [dst_w, dst_h], [0.0, dst_h]],
            dtype=np.float32,
        )
        h_mat = cv2.getPerspectiveTransform(src, dst_ref)

        if not h_cfg.keep_full_canvas:
            out_w = max(16, int(round(dst_w)))
            out_h = max(16, int(round(dst_h)))
            rectified = cv2.warpPerspective(
                img_bgr,
                h_mat,
                (out_w, out_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            final_h = h_mat
        else:
            ih, iw = img_bgr.shape[:2]
            canvas_corners = np.array(
                [[[0.0, 0.0]], [[iw, 0.0]], [[iw, ih]], [[0.0, ih]]],
                dtype=np.float32,
            )
            warped_corners = cv2.perspectiveTransform(canvas_corners, h_mat).reshape(-1, 2)

            min_xy = np.floor(warped_corners.min(axis=0))
            max_xy = np.ceil(warped_corners.max(axis=0))
            out_w = int(max_xy[0] - min_xy[0])
            out_h = int(max_xy[1] - min_xy[1])

            if out_w > 12000 or out_h > 12000 or out_w <= 0 or out_h <= 0:
                raise CalibrationError(
                    f"Ukuran kanvas hasil homografi full-canvas tidak wajar ({out_w}x{out_h} px)."
                )

            t_mat = np.array(
                [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            final_h = t_mat @ h_mat
            rectified = cv2.warpPerspective(
                img_bgr,
                final_h,
                (out_w, out_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

        scale_per_pixel = 1.0 / ppu
        logger.info(
            "Rektifikasi Homografi 4-Titik Selesai | Output: %dx%d px | Skala Presisi: %.6f unit/px",
            rectified.shape[1],
            rectified.shape[0],
            scale_per_pixel,
        )
        return rectified, scale_per_pixel, final_h


# ==============================================================================
# TAHAP 1: PREPROCESSING & KALIBRASI SKALA
# ==============================================================================


class ImagePreprocessor:
    """Menangani pemuatan citra, koreksi lensa/ArUco/perspektif, CLAHE, bilateral filter, dan binarisasi."""

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def load_and_rectify_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"File citra tidak ditemukan: {image_path}")

        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise PipelineError(f"Gagal mendekode citra (format korup/tidak didukung): {image_path}")

        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            alpha = img_bgr[:, :, 3:4].astype(np.float32) / 255.0
            bgr = img_bgr[:, :, :3].astype(np.float32)
            bg_val = 0.0 if self.cfg.calibration.invert_polarity else 255.0
            bg_canvas = np.full_like(bgr, bg_val)
            img_bgr = np.clip(bgr * alpha + bg_canvas * (1.0 - alpha), 0, 255).astype(np.uint8)
        elif img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        # Tahap 0A: Undistorsi Lensa Kamera
        if self.cfg.camera_intrinsics is not None:
            logger.info("Menerapkan koreksi distorsi lensa dari matriks intrinsik kamera...")
            img_bgr = CameraIntrinsicsCalibrator.undistort_image(img_bgr, self.cfg.camera_intrinsics)

        # Tahap 0B: Deteksi Otomatis ArUco -> Homografi 4-Titik & Skala 1:1
        active_homography: Optional[HomographyConfig] = self.cfg.homography
        if self.cfg.aruco is not None and self.cfg.aruco.is_enabled:
            debug_path = (
                self.cfg.output_dir / f"{image_path.stem}_aruco_detected.png"
                if self.cfg.aruco.save_debug_overlay
                else None
            )
            active_homography = ArucoAutoCalibrator.build_homography_from_aruco(
                img_bgr=img_bgr,
                aruco_cfg=self.cfg.aruco,
                unit=self.cfg.calibration.unit,
                debug_output_path=debug_path,
            )

        if active_homography is not None:
            if self.cfg.interactive_pick_points and (self.cfg.aruco is None or not self.cfg.aruco.is_enabled):
                picked = HomographyPerspectiveRectifier.interactive_pick_4_points(img_bgr)
                active_homography.src_points = picked

            img_bgr, new_scale_per_px, _ = HomographyPerspectiveRectifier.rectify(
                img_bgr, active_homography
            )
            self.cfg.calibration.ref_real_size = new_scale_per_px
            self.cfg.calibration.ref_pixels = 1.0

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_bgr, img_rgb

    def enhance_and_denoise(self, img_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.cfg.clahe_clip, tileGridSize=(8, 8))
        l_norm = clahe.apply(l_chan)

        lab_enhanced = cv2.merge((l_norm, a_chan, b_chan))
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2GRAY)

        return cv2.bilateralFilter(
            gray,
            d=self.cfg.bilateral_d,
            sigmaColor=self.cfg.bilateral_sigma_color,
            sigmaSpace=self.cfg.bilateral_sigma_space,
        )

    def enhance_color_preserve_edges(self, img_bgr: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(
            img_bgr,
            d=self.cfg.bilateral_d,
            sigmaColor=self.cfg.bilateral_sigma_color,
            sigmaSpace=self.cfg.bilateral_sigma_space,
        )

    def binarize_high_precision(self, denoised_gray: np.ndarray) -> np.ndarray:
        if self.cfg.use_otsu:
            thresh_type = (
                cv2.THRESH_BINARY if self.cfg.calibration.invert_polarity else cv2.THRESH_BINARY_INV
            )
            _, binary = cv2.threshold(denoised_gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        else:
            block_size = self.cfg.adaptive_block_size
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(3, block_size)
            thresh_type = (
                cv2.THRESH_BINARY if self.cfg.calibration.invert_polarity else cv2.THRESH_BINARY_INV
            )
            binary = cv2.adaptiveThreshold(
                denoised_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresh_type,
                block_size,
                self.cfg.adaptive_c,
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


# ==============================================================================
# TAHAP 2A: REKONSTRUKSI CAD / ORTOGRAFIS (SINGLE & MULTI-COLOR MULTI-HEIGHT Z)
# ==============================================================================


class ColorCadLayerSegmenter:
    """
    Mengelompokkan segmen warna berbeda pada denah 2D dalam ruang perseptual
    CIE L*a*b* dan menghasilkan mask biner presisi untuk tiap aturan tinggi Z.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.c_cfg = config.color_cad

    def _detect_background_lab(self, lab_img: np.ndarray) -> np.ndarray:
        top = lab_img[0, :, :]
        bot = lab_img[-1, :, :]
        left = lab_img[:, 0, :]
        right = lab_img[:, -1, :]
        border_pixels = np.vstack([top, bot, left, right])
        return np.median(border_pixels, axis=0)

    def _discover_auto_color_rules(
        self, fg_lab_pixels: np.ndarray
    ) -> List[ColorLayerRule]:
        k = int(self.c_cfg.auto_num_colors or 1)
        if len(fg_lab_pixels) < k:
            raise GeometryExtractionError(
                f"Piksel latar depan terlalu sedikit ({len(fg_lab_pixels)}) untuk {k} klaster warna."
            )

        max_samples = 60000
        if len(fg_lab_pixels) > max_samples:
            step = len(fg_lab_pixels) // max_samples
            samples = fg_lab_pixels[::step].astype(np.float32)
        else:
            samples = fg_lab_pixels.astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
        _, _, centers = cv2.kmeans(
            samples,
            K=k,
            bestLabels=None,
            criteria=criteria,
            attempts=5,
            flags=cv2.KMEANS_PP_CENTERS,
        )

        order = sorted(
            range(k),
            key=lambda idx: (round(float(centers[idx][0]), 1), float(centers[idx][1]), float(centers[idx][2])),
        )

        base_max_h = self.cfg.calibration.extrude_height
        custom_heights = self.c_cfg.auto_heights or []

        rules: List[ColorLayerRule] = []
        for rank, c_idx in enumerate(order):
            center_lab = centers[c_idx]
            rgb_tuple = cie_lab_to_rgb_tuple(center_lab)
            if rank < len(custom_heights):
                h_val = float(custom_heights[rank])
            else:
                factor = 1.0 - (0.75 * rank / max(1, k - 1)) if k > 1 else 1.0
                h_val = max(0.05, round(base_max_h * factor, 4))

            rule = ColorLayerRule(
                name=f"AutoColor_{rank + 1}_{rgb_tuple[0]:02X}{rgb_tuple[1]:02X}{rgb_tuple[2]:02X}",
                rgb=rgb_tuple,
                extrude_height=h_val,
                z_base=0.0,
            )
            rules.append(rule)

        return rules

    def segment_layers(
        self, denoised_bgr: np.ndarray
    ) -> List[Tuple[ColorLayerRule, np.ndarray]]:
        img_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        lab_img = rgb_array_to_cie_lab(img_rgb)
        h, w = lab_img.shape[:2]

        bg_lab = self._detect_background_lab(lab_img)
        diff_bg = lab_img - bg_lab.reshape(1, 1, 3)
        delta_e_bg = np.linalg.norm(diff_bg, axis=-1)

        fg_mask = delta_e_bg > self.c_cfg.bg_delta_e_threshold
        fg_pixels_lab = lab_img[fg_mask]
        if len(fg_pixels_lab) == 0:
            raise GeometryExtractionError(
                "Tidak ditemukan piksel warna latar depan yang berbeda dari warna latar belakang."
            )

        if len(self.c_cfg.rules) > 0:
            active_rules = self.c_cfg.rules
        else:
            active_rules = self._discover_auto_color_rules(fg_pixels_lab)

        centroids_lab = np.stack([r.lab for r in active_rules], axis=0)
        dists = np.linalg.norm(
            fg_pixels_lab[:, None, :] - centroids_lab[None, :, :],
            axis=-1,
        )
        nearest_idx = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(len(fg_pixels_lab)), nearest_idx]
        valid_fg = nearest_dist <= self.c_cfg.delta_e_tolerance

        fg_rows, fg_cols = np.where(fg_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        segmented_layers: List[Tuple[ColorLayerRule, np.ndarray]] = []

        logger.info("Segmentasi Multi-Warna CAD Aktif (%d aturan warna):", len(active_rules))
        for k_idx, rule in enumerate(active_rules):
            layer_mask = np.zeros((h, w), dtype=np.uint8)
            sel = valid_fg & (nearest_idx == k_idx)
            layer_mask[fg_rows[sel], fg_cols[sel]] = 255
            layer_mask = cv2.morphologyEx(layer_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            px_count = int(np.count_nonzero(layer_mask))
            logger.info(
                "  Layer %-18s | HEX: %s | RGB: %-15s | Z_base: %.3f | Tinggi Z: %.3f %s | Piksel: %d",
                rule.name,
                rule.hex_code,
                str(rule.rgb),
                rule.z_base,
                rule.extrude_height,
                self.cfg.calibration.unit,
                px_count,
            )
            if px_count >= self.cfg.min_contour_area_px:
                segmented_layers.append((rule, layer_mask))
            else:
                logger.warning(
                    "  Layer '%s' (%s) dilewati karena luas piksel (%d) < min_area (%d).",
                    rule.name,
                    rule.hex_code,
                    px_count,
                    int(self.cfg.min_contour_area_px),
                )

        if not segmented_layers:
            raise GeometryExtractionError(
                "Tidak ada segmen warna yang cocok dalam batas toleransi --color-tolerance. "
                "Periksa kode HEX atau naikkan --color-tolerance."
            )

        return segmented_layers


class CadOrthographicReconstructor:
    """
    Mode A: Mengubah citra CAD (tunggal maupun multi-warna) menjadi mesh solid 3D
    watertight menggunakan ekstraksi kontur hierarkis, Ramer-Douglas-Peucker,
    validasi poligon Shapely, dan triangulasi CDT Trimesh.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.color_segmenter = ColorCadLayerSegmenter(config)

    def _refine_contours_subpixel(
        self, gray_img: np.ndarray, contours: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        refined: List[np.ndarray] = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        h, w = gray_img.shape[:2]

        for cnt in contours:
            pts = cnt.reshape(-1, 2).astype(np.float32)
            if len(pts) < 3:
                continue
            valid_mask = (
                (pts[:, 0] >= 3) & (pts[:, 0] < w - 3) & (pts[:, 1] >= 3) & (pts[:, 1] < h - 3)
            )
            if np.all(valid_mask) and len(pts) >= 4:
                pts_sub = pts.reshape(-1, 1, 2).copy()
                try:
                    cv2.cornerSubPix(gray_img, pts_sub, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
                    pts = pts_sub.reshape(-1, 2)
                except cv2.error:
                    pass
            refined.append(pts)
        return refined

    def _extract_shapely_polygons(
        self, binary_mask: np.ndarray, ref_gray: np.ndarray
    ) -> List[Polygon]:
        h, _ = binary_mask.shape[:2]
        scale = self.cfg.calibration.scale_per_pixel

        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if not contours or hierarchy is None:
            return []

        hier = hierarchy[0]
        raw_polygons: List[Polygon] = []

        for idx, cnt in enumerate(contours):
            if hier[idx][3] != -1:
                continue

            area_px = cv2.contourArea(cnt)
            if area_px < self.cfg.min_contour_area_px:
                continue

            approx_ext = cv2.approxPolyDP(cnt, epsilon=self.cfg.rdp_epsilon_px, closed=True)
            refined_ext = self._refine_contours_subpixel(ref_gray, [approx_ext])
            if not refined_ext or len(refined_ext[0]) < 3:
                continue

            ext_pts_px = refined_ext[0]
            ext_coords = [(float(x * scale), float((h - y) * scale)) for x, y in ext_pts_px]

            holes_coords: List[List[Tuple[float, float]]] = []
            child_idx = hier[idx][2]
            while child_idx != -1:
                hole_cnt = contours[child_idx]
                if cv2.contourArea(hole_cnt) >= (self.cfg.min_contour_area_px * 0.25):
                    approx_hole = cv2.approxPolyDP(
                        hole_cnt, epsilon=self.cfg.rdp_epsilon_px, closed=True
                    )
                    refined_hole = self._refine_contours_subpixel(ref_gray, [approx_hole])
                    if refined_hole and len(refined_hole[0]) >= 3:
                        h_pts = refined_hole[0]
                        holes_coords.append(
                            [(float(x * scale), float((h - y) * scale)) for x, y in h_pts]
                        )
                child_idx = hier[child_idx][0]

            poly = Polygon(shell=ext_coords, holes=holes_coords)
            if not poly.is_valid:
                poly = make_valid(poly)

            if isinstance(poly, Polygon) and poly.area > 0:
                raw_polygons.append(poly)
            elif isinstance(poly, (MultiPolygon, GeometryCollection)):
                for geom in poly.geoms:
                    if isinstance(geom, Polygon) and geom.area > 0:
                        raw_polygons.append(geom)

        return raw_polygons

    def _extrude_mask_to_mesh(
        self,
        binary_mask: np.ndarray,
        ref_gray: np.ndarray,
        extrude_height: float,
        z_base: float,
        layer_rgb: Optional[Tuple[int, int, int]] = None,
        source_img_rgb: Optional[np.ndarray] = None,
    ) -> List[trimesh.Trimesh]:
        polygons = self._extract_shapely_polygons(binary_mask, ref_gray)
        parts: List[trimesh.Trimesh] = []

        for poly in polygons:
            try:
                mesh_part = trimesh.creation.extrude_polygon(poly, height=extrude_height)
                if z_base != 0.0:
                    mesh_part.apply_translation([0.0, 0.0, z_base])

                if layer_rgb is not None:
                    rgba = np.array(
                        [[layer_rgb[0], layer_rgb[1], layer_rgb[2], 255]],
                        dtype=np.uint8,
                    )
                    mesh_part.visual.vertex_colors = np.repeat(rgba, len(mesh_part.vertices), axis=0)
                elif source_img_rgb is not None:
                    self._apply_projected_vertex_colors(mesh_part, source_img_rgb)

                parts.append(mesh_part)
            except Exception as exc:
                logger.warning("Melewati satu sub-poligon karena gagal triangulasi CDT: %s", exc)

        return parts

    def reconstruct(
        self,
        binary_mask: np.ndarray,
        denoised_gray: np.ndarray,
        denoised_bgr: np.ndarray,
        img_rgb: np.ndarray,
        z_offset: float = 0.0,
    ) -> trimesh.Trimesh:
        all_parts: List[trimesh.Trimesh] = []

        if self.cfg.color_cad.is_enabled:
            color_layers = self.color_segmenter.segment_layers(denoised_bgr)
            for rule, layer_mask in color_layers:
                layer_parts = self._extrude_mask_to_mesh(
                    binary_mask=layer_mask,
                    ref_gray=denoised_gray,
                    extrude_height=rule.extrude_height,
                    z_base=z_offset + rule.z_base,
                    layer_rgb=rule.rgb,
                )
                if not layer_parts:
                    logger.warning("Layer warna %s tidak menghasilkan poligon solid valid.", rule.hex_code)
                all_parts.extend(layer_parts)
        else:
            all_parts = self._extrude_mask_to_mesh(
                binary_mask=binary_mask,
                ref_gray=denoised_gray,
                extrude_height=self.cfg.calibration.extrude_height,
                z_base=z_offset,
                source_img_rgb=img_rgb,
            )

        if not all_parts:
            raise GeometryExtractionError(
                "Tidak ditemukan kontur geometris valid untuk diekstrusi ke 3D. "
                "Periksa polaritas binarization (--invert) atau toleransi warna."
            )

        return all_parts[0] if len(all_parts) == 1 else trimesh.util.concatenate(all_parts)

    def _apply_projected_vertex_colors(self, mesh: trimesh.Trimesh, img_rgb: np.ndarray) -> None:
        h, w = img_rgb.shape[:2]
        scale = self.cfg.calibration.scale_per_pixel
        verts = mesh.vertices

        u = np.clip(np.round(verts[:, 0] / scale).astype(int), 0, w - 1)
        v = np.clip(np.round(h - (verts[:, 1] / scale)).astype(int), 0, h - 1)
        colors_rgb = img_rgb[v, u]
        alpha = np.full((len(verts), 1), 255, dtype=np.uint8)
        mesh.visual.vertex_colors = np.hstack([colors_rgb, alpha])


# ==============================================================================
# TAHAP 2B: REKONSTRUKSI PERMUKAAN / RELIEF & OBJEK 3D (DEPTH + POISSON/ALPHA)
# ==============================================================================


class SurfaceReliefReconstructor:
    """
    Mode B: Mengubah foto tunggal atau multi-layer menjadi Point Cloud
    terkalibrasi dan membangun permukaan solid 3D melalui Screened Poisson
    Surface Reconstruction, Alpha Shapes, atau Watertight Solid Grid.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self._depth_pipe = None

    def _estimate_depth_map(self, img_rgb: np.ndarray, denoised_gray: np.ndarray) -> np.ndarray:
        backend = self.cfg.depth_backend

        if backend in ("depth-anything", "midas"):
            try:
                from transformers import pipeline as hf_pipeline

                if self._depth_pipe is None:
                    model_id = (
                        "depth-anything/Depth-Anything-V2-Small-hf"
                        if backend == "depth-anything"
                        else "Intel/dpt-hybrid-midas"
                    )
                    logger.info("Memuat model estimasi kedalaman: %s", model_id)
                    self._depth_pipe = hf_pipeline(task="depth-estimation", model=model_id)

                pil_img = Image.fromarray(img_rgb)
                result = self._depth_pipe(pil_img)
                depth_raw = np.array(result["depth"], dtype=np.float32)
                if depth_raw.shape[:2] != img_rgb.shape[:2]:
                    depth_raw = cv2.resize(
                        depth_raw,
                        (img_rgb.shape[1], img_rgb.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
                if d_max - d_min < 1e-8:
                    return np.zeros_like(depth_raw, dtype=np.float32)
                depth_norm = (depth_raw - d_min) / (d_max - d_min)
                if self.cfg.calibration.invert_polarity:
                    depth_norm = 1.0 - depth_norm
                return depth_norm.astype(np.float32)

            except ImportError:
                logger.warning(
                    "Paket 'torch'/'transformers' tidak terpasang. Beralih otomatis ke backend 'heightmap'."
                )
            except Exception as exc:
                logger.warning(
                    "Gagal menjalankan inferensi %s (%s). Beralih ke backend 'heightmap'.",
                    backend,
                    exc,
                )

        depth_norm = denoised_gray.astype(np.float32) / 255.0
        if self.cfg.calibration.invert_polarity:
            depth_norm = 1.0 - depth_norm
        depth_norm = gaussian_filter(depth_norm, sigma=0.8)
        return np.clip(depth_norm, 0.0, 1.0)

    def build_calibrated_point_cloud(
        self, img_rgb: np.ndarray, denoised_gray: np.ndarray
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
        depth_map = self._estimate_depth_map(img_rgb, denoised_gray)
        h, w = depth_map.shape[:2]

        max_dim = max(h, w)
        if max_dim > self.cfg.max_grid_dim:
            ratio = self.cfg.max_grid_dim / float(max_dim)
            new_w = max(16, int(round(w * ratio)))
            new_h = max(16, int(round(h * ratio)))
            depth_small = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            eff_scale = self.cfg.calibration.scale_per_pixel * (w / float(new_w))
        else:
            depth_small = depth_map
            rgb_small = img_rgb
            eff_scale = self.cfg.calibration.scale_per_pixel

        gh, gw = depth_small.shape[:2]
        u_grid, v_grid = np.meshgrid(np.arange(gw, dtype=np.float64), np.arange(gh, dtype=np.float64))

        x_coords = u_grid * eff_scale
        y_coords = (gh - v_grid) * eff_scale
        z_coords = (
            depth_small.astype(np.float64) * self.cfg.calibration.extrude_height
            + self.cfg.base_thickness
        )

        points = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(-1, 3)
        colors = (rgb_small.reshape(-1, 3).astype(np.float64)) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        radius = eff_scale * 3.0
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))

        return pcd, z_coords, rgb_small

    def _build_watertight_grid_solid(
        self, z_grid: np.ndarray, rgb_grid: np.ndarray, eff_scale: float
    ) -> trimesh.Trimesh:
        gh, gw = z_grid.shape
        u = np.arange(gw, dtype=np.float64) * eff_scale
        v = (gh - np.arange(gh, dtype=np.float64)) * eff_scale
        xx, yy = np.meshgrid(u, v)

        top_verts = np.stack([xx, yy, z_grid], axis=-1).reshape(-1, 3)
        bot_verts = np.stack([xx, yy, np.zeros_like(z_grid)], axis=-1).reshape(-1, 3)
        vertices = np.vstack([top_verts, bot_verts])

        top_colors = np.hstack(
            [rgb_grid.reshape(-1, 3), np.full((gh * gw, 1), 255, dtype=np.uint8)]
        )
        bot_colors = top_colors.copy()
        vertex_colors = np.vstack([top_colors, bot_colors])

        faces: List[List[int]] = []
        offset = gh * gw

        def idx_top(r: int, c: int) -> int:
            return r * gw + c

        def idx_bot(r: int, c: int) -> int:
            return offset + r * gw + c

        for r in range(gh - 1):
            for c in range(gw - 1):
                t00, t01 = idx_top(r, c), idx_top(r, c + 1)
                t10, t11 = idx_top(r + 1, c), idx_top(r + 1, c + 1)
                faces.append([t00, t10, t11])
                faces.append([t00, t11, t01])

                b00, b01 = idx_bot(r, c), idx_bot(r, c + 1)
                b10, b11 = idx_bot(r + 1, c), idx_bot(r + 1, c + 1)
                faces.append([b00, b11, b10])
                faces.append([b00, b01, b11])

        for c in range(gw - 1):
            faces.append([idx_top(0, c), idx_top(0, c + 1), idx_bot(0, c + 1)])
            faces.append([idx_top(0, c), idx_bot(0, c + 1), idx_bot(0, c)])
            r_last = gh - 1
            faces.append([idx_top(r_last, c), idx_bot(r_last, c + 1), idx_top(r_last, c + 1)])
            faces.append([idx_top(r_last, c), idx_bot(r_last, c), idx_bot(r_last, c + 1)])

        for r in range(gh - 1):
            faces.append([idx_top(r, 0), idx_bot(r + 1, 0), idx_top(r + 1, 0)])
            faces.append([idx_top(r, 0), idx_bot(r, 0), idx_bot(r + 1, 0)])
            c_last = gw - 1
            faces.append([idx_top(r, c_last), idx_top(r + 1, c_last), idx_bot(r + 1, c_last)])
            faces.append([idx_top(r, c_last), idx_bot(r + 1, c_last), idx_bot(r, c_last)])

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.array(faces, dtype=np.int64),
            vertex_colors=vertex_colors,
            process=True,
        )
        mesh.fix_normals()
        return mesh

    def reconstruct_from_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        z_grid: Optional[np.ndarray] = None,
        rgb_grid: Optional[np.ndarray] = None,
        eff_scale: Optional[float] = None,
    ) -> trimesh.Trimesh:
        if (
            self.cfg.surface_method == "grid-solid"
            and z_grid is not None
            and rgb_grid is not None
            and eff_scale is not None
        ):
            return self._build_watertight_grid_solid(z_grid, rgb_grid, eff_scale)

        if self.cfg.surface_method == "alpha":
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = float(np.mean(distances)) if len(distances) > 0 else 1.0
            alpha = avg_dist * self.cfg.alpha_radius_factor
            logger.info("Menjalankan Alpha Shape reconstruction (alpha=%.4f)...", alpha)
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        else:
            logger.info(
                "Menjalankan Screened Poisson Surface Reconstruction (depth=%d)...",
                self.cfg.poisson_depth,
            )
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.cfg.poisson_depth
            )
            densities_arr = np.asarray(densities)
            if len(densities_arr) > 0 and self.cfg.poisson_trim_quantile > 0.0:
                thresh = np.quantile(densities_arr, self.cfg.poisson_trim_quantile)
                vertices_to_remove = densities_arr < thresh
                mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

            if (
                self.cfg.close_solid_base
                and z_grid is not None
                and rgb_grid is not None
                and eff_scale is not None
            ):
                logger.info("Menutup alas bawah Z=0 untuk menjamin solid watertight siap cetak...")
                z_smoothed = gaussian_filter(z_grid, sigma=0.6)
                return self._build_watertight_grid_solid(z_smoothed, rgb_grid, eff_scale)

        mesh_o3d.compute_vertex_normals()
        verts = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        if len(verts) == 0 or len(faces) == 0:
            raise GeometryExtractionError("Rekonstruksi permukaan menghasilkan mesh kosong.")

        v_colors = None
        if mesh_o3d.has_vertex_colors():
            rgb = np.clip(np.asarray(mesh_o3d.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
            v_colors = np.hstack([rgb, np.full((len(verts), 1), 255, dtype=np.uint8)])

        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=v_colors, process=True)


# ==============================================================================
# TAHAP 3: MESH CLEANING & TOPOLOGY OPTIMIZATION
# ==============================================================================


class MeshTopologyOptimizer:
    """
    Membersihkan artefak topologi, menjamin manifold & watertightness, serta
    melakukan smoothing/decimation yang mempertahankan sudut tajam.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def clean_and_optimize(
        self, mesh: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, o3d.geometry.TriangleMesh]:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()

        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            total_faces = len(mesh.faces)
            significant = [c for c in components if len(c.faces) >= max(4, int(total_faces * 0.001))]
            if significant:
                mesh = (
                    significant[0]
                    if len(significant) == 1
                    else trimesh.util.concatenate(significant)
                )

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))

        has_colors = (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) == len(mesh.vertices)
        )
        if has_colors:
            rgb_norm = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb_norm)

        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_non_manifold_edges()
        o3d_mesh.remove_unreferenced_vertices()

        if self.cfg.taubin_iterations > 0 and self.cfg.mode == "surface":
            o3d_mesh = o3d_mesh.filter_smooth_taubin(
                number_of_iterations=self.cfg.taubin_iterations,
                lambda_filter=0.5,
                mu=-0.53,
            )

        if 0.0 < self.cfg.target_reduction < 0.95:
            current_tris = len(o3d_mesh.triangles)
            target_tris = max(12, int(current_tris * (1.0 - self.cfg.target_reduction)))
            if target_tris < current_tris:
                o3d_mesh = o3d_mesh.simplify_quadric_decimation(
                    target_number_of_triangles=target_tris
                )

        o3d_mesh.compute_vertex_normals()

        opt_verts = np.asarray(o3d_mesh.vertices)
        opt_faces = np.asarray(o3d_mesh.triangles)
        opt_colors = None
        if o3d_mesh.has_vertex_colors():
            rgb = np.clip(np.asarray(o3d_mesh.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
            opt_colors = np.hstack([rgb, np.full((len(opt_verts), 1), 255, dtype=np.uint8)])

        final_tm = trimesh.Trimesh(
            vertices=opt_verts,
            faces=opt_faces,
            vertex_colors=opt_colors,
            process=True,
        )

        if not final_tm.is_watertight:
            trimesh.repair.fill_holes(final_tm)
        trimesh.repair.fix_inversion(final_tm)
        trimesh.repair.fix_normals(final_tm)

        if final_tm.is_watertight and final_tm.volume < 0:
            final_tm.invert()

        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(final_tm.vertices, dtype=np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(final_tm.faces, dtype=np.int32))
        if (
            hasattr(final_tm.visual, "vertex_colors")
            and final_tm.visual.vertex_colors is not None
            and len(final_tm.visual.vertex_colors) == len(final_tm.vertices)
        ):
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.asarray(final_tm.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0
            )
        o3d_mesh.compute_vertex_normals()

        return final_tm, o3d_mesh


# ==============================================================================
# TAHAP 4: MULTI-FORMAT EXPORTER (.STL, .OBJ + MULTI-MATERIAL .MTL, .DAE, .PLY)
# ==============================================================================


class MultiFormatMeshExporter:
    """
    Mengekspor mesh 3D terkalibrasi ke 4 format standar industri sekaligus:
      1. .STL (Binary solid manifold)
      2. .OBJ + .MTL (Multi-material per warna layer CAD, lengkap dengan normal & UV)
      3. .DAE (COLLADA 1.4.1 dengan orientasi Z_UP dan unit skala untuk SketchUp)
      4. .PLY (Binary dengan vertex color RGBA)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def export_all(
        self,
        mesh: trimesh.Trimesh,
        o3d_mesh: o3d.geometry.TriangleMesh,
        base_name: str,
    ) -> Dict[str, Path]:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        exported: Dict[str, Path] = {}

        stl_path = self.cfg.output_dir / f"{base_name}.stl"
        mesh.export(str(stl_path), file_type="stl")
        exported["STL"] = stl_path

        obj_path = self.cfg.output_dir / f"{base_name}.obj"
        self._export_obj_with_multimaterial_mtl(mesh, obj_path, base_name)
        exported["OBJ"] = obj_path
        exported["MTL"] = self.cfg.output_dir / f"{base_name}.mtl"

        dae_path = self.cfg.output_dir / f"{base_name}.dae"
        self._export_sketchup_collada(mesh, dae_path)
        exported["DAE"] = dae_path

        ply_path = self.cfg.output_dir / f"{base_name}.ply"
        o3d.io.write_triangle_mesh(
            str(ply_path),
            o3d_mesh,
            write_ascii=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
        )
        exported["PLY"] = ply_path

        return exported

    def _export_obj_with_multimaterial_mtl(
        self, mesh: trimesh.Trimesh, obj_path: Path, base_name: str
    ) -> None:
        mtl_filename = f"{base_name}.mtl"
        mtl_path = obj_path.with_suffix(".mtl")

        verts = mesh.vertices
        normals = mesh.vertex_normals
        faces = mesh.faces

        has_vcolors = (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) == len(verts)
        )

        if has_vcolors:
            v_rgb = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
            face_rgb = v_rgb[faces[:, 0]]
        else:
            face_rgb = np.tile(np.array([[199, 204, 214]], dtype=np.uint8), (len(faces), 1))

        quant_rgb = (face_rgb // 4) * 4
        unique_colors, inverse_idx = np.unique(quant_rgb, axis=0, return_inverse=True)

        mtl_lines: List[str] = ["# Multi-Material MTL generated by Img2Mesh3D Pipeline"]
        mat_names: List[str] = []

        for idx, _ in enumerate(unique_colors):
            group_mask = inverse_idx == idx
            mean_col = np.mean(face_rgb[group_mask], axis=0)
            r_int, g_int, b_int = int(round(mean_col[0])), int(round(mean_col[1])), int(round(mean_col[2]))
            mat_name = f"cad_layer_{r_int:02X}{g_int:02X}{b_int:02X}_{idx + 1}"
            mat_names.append(mat_name)

            kd_r, kd_g, kd_b = mean_col[0] / 255.0, mean_col[1] / 255.0, mean_col[2] / 255.0
            mtl_lines.extend(
                [
                    f"newmtl {mat_name}",
                    "Ka 0.150000 0.150000 0.150000",
                    f"Kd {kd_r:.6f} {kd_g:.6f} {kd_b:.6f}",
                    "Ks 0.250000 0.250000 0.250000",
                    "Ns 64.000000",
                    "d 1.000000",
                    "illum 2",
                    "",
                ]
            )
        mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")

        min_xy = verts[:, :2].min(axis=0)
        span_xy = np.maximum(verts[:, :2].max(axis=0) - min_xy, 1e-8)
        uvs = (verts[:, :2] - min_xy) / span_xy

        obj_lines: List[str] = [
            f"# Wavefront OBJ exported by Img2Mesh3D ({self.cfg.calibration.unit})",
            f"mtllib {mtl_filename}",
            f"o {base_name}",
        ]
        for v in verts:
            obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for uv in uvs:
            obj_lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
        for vn in normals:
            obj_lines.append(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}")

        obj_lines.append("s 1")
        for idx, mat_name in enumerate(mat_names):
            obj_lines.append(f"g {mat_name}")
            obj_lines.append(f"usemtl {mat_name}")
            group_faces = faces[inverse_idx == idx]
            for f in group_faces:
                i0, i1, i2 = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
                obj_lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")

        obj_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")

    def _export_sketchup_collada(self, mesh: trimesh.Trimesh, dae_path: Path) -> None:
        dae_bytes = trimesh.exchange.dae.export_collada(mesh)
        dae_xml_str = dae_bytes.decode("utf-8") if isinstance(dae_bytes, bytes) else str(dae_bytes)

        ns_uri = "http://www.collada.org/2005/11/COLLADASchema"
        ET.register_namespace("", ns_uri)
        root = ET.fromstring(dae_xml_str)

        asset = root.find(f"{{{ns_uri}}}asset")
        if asset is None:
            asset = ET.Element(f"{{{ns_uri}}}asset")
            root.insert(0, asset)

        unit_name_map = {"mm": "millimeter", "cm": "centimeter", "m": "meter"}
        unit_elem = asset.find(f"{{{ns_uri}}}unit")
        if unit_elem is None:
            unit_elem = ET.SubElement(asset, f"{{{ns_uri}}}unit")
        unit_elem.set("name", unit_name_map[self.cfg.calibration.unit])
        unit_elem.set("meter", f"{self.cfg.calibration.meters_per_unit:.6f}")

        up_axis_elem = asset.find(f"{{{ns_uri}}}up_axis")
        if up_axis_elem is None:
            up_axis_elem = ET.SubElement(asset, f"{{{ns_uri}}}up_axis")
        up_axis_elem.text = "Z_UP"

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(dae_path), encoding="utf-8", xml_declaration=True)


# ==============================================================================
# TAHAP 5: VALIDASI GEOMETRIS, TOPOLOGIS, & PRATINJAU INTERAKTIF OPEN3D
# ==============================================================================


class InteractiveMeshPreviewer:
    """Inspeksi visual 3D interaktif menggunakan Open3D Visualizer."""

    @staticmethod
    def _has_display_server() -> bool:
        if sys.platform.startswith("linux"):
            return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        return True

    @staticmethod
    def _build_ground_grid(
        min_bound: np.ndarray, max_bound: np.ndarray
    ) -> o3d.geometry.LineSet:
        span_x = max(float(max_bound[0] - min_bound[0]), 1.0)
        span_y = max(float(max_bound[1] - min_bound[1]), 1.0)
        max_span = max(span_x, span_y)

        raw_step = max_span / 10.0
        magnitude = 10.0 ** math.floor(math.log10(max(raw_step, 1e-6)))
        residual = raw_step / magnitude
        if residual <= 1.5:
            step = 1.0 * magnitude
        elif residual <= 3.5:
            step = 2.0 * magnitude
        elif residual <= 7.5:
            step = 5.0 * magnitude
        else:
            step = 10.0 * magnitude

        pad = step * 1.0
        x0 = math.floor((min_bound[0] - pad) / step) * step
        x1 = math.ceil((max_bound[0] + pad) / step) * step
        y0 = math.floor((min_bound[1] - pad) / step) * step
        y1 = math.ceil((max_bound[1] + pad) / step) * step

        xs = np.arange(x0, x1 + step * 0.5, step)
        ys = np.arange(y0, y1 + step * 0.5, step)

        points: List[List[float]] = []
        lines: List[List[int]] = []

        for x in xs:
            idx = len(points)
            points.append([float(x), float(y0), 0.0])
            points.append([float(x), float(y1), 0.0])
            lines.append([idx, idx + 1])

        for y in ys:
            idx = len(points)
            points.append([float(x0), float(y), 0.0])
            points.append([float(x1), float(y), 0.0])
            lines.append([idx, idx + 1])

        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
        grid.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
        grid.paint_uniform_color([0.32, 0.35, 0.40])
        return grid

    @classmethod
    def show(
        cls,
        o3d_mesh: o3d.geometry.TriangleMesh,
        report: ValidationReport,
        preview_cfg: PreviewConfig,
        output_dir: Path,
    ) -> bool:
        if not cls._has_display_server():
            logger.warning(
                "Lingkungan headless terdeteksi (tidak ada DISPLAY/WAYLAND_DISPLAY). "
                "Pratinjau 3D Open3D dilewati."
            )
            return True

        u = report.unit
        status_wt = "WATERTIGHT" if report.is_watertight else "OPEN MESH"
        win_title = (
            f"Img2Mesh3D Validator | {report.model_name} | "
            f"{report.length_x:.2f}x{report.width_y:.2f}x{report.height_z:.2f} {u} | "
            f"{report.num_faces:,} tris [{status_wt}]"
        )

        print("  Kontrol Pratinjau 3D Open3D:")
        print("    [Mouse Kiri] Rotasi  | [Scroll] Zoom  | [Mouse Tengah/Shift+Drag] Pan")
        print("    [W] Toggle Wireframe | [B] Toggle Back-Face | [G] Toggle Grid & Sumbu XYZ")
        print("    [R] Reset Kamera     | [S] Simpan Screenshot PNG | [Q / ESC] Tutup Pratinjau\n")

        max_dim = max(report.length_x, report.width_y, report.height_z, 1.0)
        axis_size = max_dim * 0.18
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size, origin=[0.0, 0.0, 0.0]
        )

        aabb = o3d_mesh.get_axis_aligned_bounding_box()
        aabb.color = (0.0, 0.85, 1.0)

        min_b = np.asarray(aabb.min_bound)
        max_b = np.asarray(aabb.max_bound)
        ground_grid = cls._build_ground_grid(min_b, max_b)

        state = {
            "wireframe": preview_cfg.wireframe,
            "back_face": preview_cfg.show_back_face,
            "guides": preview_cfg.show_guides,
            "screenshot_idx": 1,
        }

        try:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            created = vis.create_window(
                window_name=win_title,
                width=preview_cfg.window_width,
                height=preview_cfg.window_height,
            )
            if not created:
                logger.warning("Gagal menginisialisasi konteks OpenGL Open3D. Pratinjau dilewati.")
                return True

            vis.add_geometry(o3d_mesh)
            if state["guides"]:
                vis.add_geometry(coord_frame)
                vis.add_geometry(aabb)
                vis.add_geometry(ground_grid)

            render_opt = vis.get_render_option()
            render_opt.background_color = np.array([0.11, 0.12, 0.15], dtype=np.float64)
            render_opt.mesh_show_wireframe = state["wireframe"]
            render_opt.mesh_show_back_face = state["back_face"]
            render_opt.light_on = True

            ctr = vis.get_view_control()
            ctr.set_front([0.55, -0.65, 0.52])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_lookat(aabb.get_center())
            ctr.set_zoom(0.75)

            def toggle_wireframe(v: o3d.visualization.Visualizer) -> bool:
                state["wireframe"] = not state["wireframe"]
                v.get_render_option().mesh_show_wireframe = state["wireframe"]
                return False

            def toggle_backface(v: o3d.visualization.Visualizer) -> bool:
                state["back_face"] = not state["back_face"]
                v.get_render_option().mesh_show_back_face = state["back_face"]
                return False

            def toggle_guides(v: o3d.visualization.Visualizer) -> bool:
                state["guides"] = not state["guides"]
                if state["guides"]:
                    v.add_geometry(coord_frame, reset_bounding_box=False)
                    v.add_geometry(aabb, reset_bounding_box=False)
                    v.add_geometry(ground_grid, reset_bounding_box=False)
                else:
                    v.remove_geometry(coord_frame, reset_bounding_box=False)
                    v.remove_geometry(aabb, reset_bounding_box=False)
                    v.remove_geometry(ground_grid, reset_bounding_box=False)
                return False

            def save_screenshot(v: o3d.visualization.Visualizer) -> bool:
                output_dir.mkdir(parents=True, exist_ok=True)
                shot_path = output_dir / f"{report.model_name}_preview_{state['screenshot_idx']}.png"
                v.capture_screen_image(str(shot_path), do_render=True)
                logger.info("Screenshot pratinjau disimpan ke: %s", shot_path.resolve())
                state["screenshot_idx"] += 1
                return False

            def reset_camera(v: o3d.visualization.Visualizer) -> bool:
                v.reset_view_point(True)
                return False

            vis.register_key_callback(ord("W"), toggle_wireframe)
            vis.register_key_callback(ord("B"), toggle_backface)
            vis.register_key_callback(ord("G"), toggle_guides)
            vis.register_key_callback(ord("S"), save_screenshot)
            vis.register_key_callback(ord("R"), reset_camera)

            vis.run()
            vis.destroy_window()

        except Exception as exc:
            logger.warning("Terjadi kendala pada jendela Open3D Visualizer: %s", exc)
            return True

        if preview_cfg.confirm_before_export:
            try:
                ans = input(
                    f"Lanjutkan ekspor file 3D untuk '{report.model_name}' ke disk? [Y/n]: "
                ).strip().lower()
                if ans in ("n", "no", "tidak"):
                    logger.warning("Ekspor file untuk '%s' dibatalkan oleh pengguna.", report.model_name)
                    return False
            except EOFError:
                return True

        return True


class MeshValidator:
    """Mengaudit dimensi fisik nyata, volume, dan integritas topologi mesh."""

    @staticmethod
    def validate(
        model_name: str,
        mesh: trimesh.Trimesh,
        o3d_mesh: o3d.geometry.TriangleMesh,
        unit: str,
        exported_files: Optional[Dict[str, Path]] = None,
    ) -> ValidationReport:
        extents = mesh.extents
        is_wt = bool(mesh.is_watertight)
        volume = float(mesh.volume) if is_wt else None

        return ValidationReport(
            model_name=model_name,
            unit=unit,
            length_x=float(extents[0]),
            width_y=float(extents[1]),
            height_z=float(extents[2]),
            num_vertices=int(len(mesh.vertices)),
            num_faces=int(len(mesh.faces)),
            surface_area=float(mesh.area),
            volume=volume,
            is_watertight=is_wt,
            is_edge_manifold=bool(o3d_mesh.is_edge_manifold(allow_boundary_edges=False)),
            is_vertex_manifold=bool(o3d_mesh.is_vertex_manifold()),
            is_winding_consistent=bool(mesh.is_winding_consistent),
            exported_files=exported_files or {},
        )

    @staticmethod
    def print_report(report: ValidationReport) -> None:
        u = report.unit
        vol_str = f"{report.volume:,.4f} {u}^3" if report.volume is not None else "N/A (Open Surface)"
        border = "=" * 72
        print(f"\n{border}")
        print(f" LAPORAN VALIDASI GEOMETRI & TOPOLOGI 3D : {report.model_name}")
        print(border)
        print(f"  Dimensi Fisik (P x L x T) : {report.length_x:.4f} x {report.width_y:.4f} x {report.height_z:.4f} {u}")
        print(f"  Jumlah Vertices           : {report.num_vertices:,}")
        print(f"  Jumlah Faces (Triangles)  : {report.num_faces:,}")
        print(f"  Luas Permukaan Total      : {report.surface_area:,.4f} {u}^2")
        print(f"  Volume Solid              : {vol_str}")
        print(f"  Status Watertight (Solid) : {'YA (VALID)' if report.is_watertight else 'TIDAK (OPEN MESH)'}")
        print(f"  Edge Manifold             : {'YA' if report.is_edge_manifold else 'TIDAK'}")
        print(f"  Vertex Manifold           : {'YA' if report.is_vertex_manifold else 'TIDAK'}")
        print(f"  Normal Winding Consistent : {'YA' if report.is_winding_consistent else 'TIDAK'}")
        if report.exported_files:
            print("-" * 72)
            print("  File Output Terealisasi:")
            for fmt, path in report.exported_files.items():
                print(f"    [{fmt:<3}] -> {path.resolve()}")
        print(f"{border}\n")


# ==============================================================================
# ORKESTRATOR UTAMA PIPELINE (SINGLE & MULTI-FILE)
# ==============================================================================


class ImageTo3DPipeline:
    """Orkestrator end-to-end yang menghubungkan seluruh tahapan presisi."""

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.preprocessor = ImagePreprocessor(config)
        self.cad_reconstructor = CadOrthographicReconstructor(config)
        self.surface_reconstructor = SurfaceReliefReconstructor(config)
        self.optimizer = MeshTopologyOptimizer(config)
        self.exporter = MultiFormatMeshExporter(config)

    def run(self) -> List[ValidationReport]:
        reports: List[ValidationReport] = []
        files = self.cfg.input_paths

        if self.cfg.multi_strategy == "batch" or len(files) == 1:
            for img_path in files:
                logger.info("Memproses file tunggal: %s (Mode: %s)", img_path.name, self.cfg.mode.upper())
                raw_mesh = self._process_single_file(img_path, z_offset=0.0)
                report = self._finalize_and_export(raw_mesh, img_path.stem)
                reports.append(report)

        elif self.cfg.multi_strategy == "stack" and self.cfg.mode == "cad":
            logger.info("Memproses %d file dalam mode STACKED CAD LAYERS...", len(files))
            stacked_meshes: List[trimesh.Trimesh] = []
            current_z = 0.0
            step_z = self.cfg.calibration.extrude_height + self.cfg.layer_gap

            for idx, img_path in enumerate(files):
                logger.info("  Layer %d/%d: %s (Z=%.3f)", idx + 1, len(files), img_path.name, current_z)
                layer_mesh = self._process_single_file(img_path, z_offset=current_z)
                stacked_meshes.append(layer_mesh)
                current_z += step_z

            merged_stack = trimesh.util.concatenate(stacked_meshes)
            report = self._finalize_and_export(merged_stack, "stacked_cad_assembly")
            reports.append(report)

        elif self.cfg.multi_strategy == "merge" and self.cfg.mode == "surface":
            logger.info("Memproses %d file dalam mode MULTI-VIEW POINT CLOUD MERGE...", len(files))
            combined_pcd = o3d.geometry.PointCloud()
            voxel_size = self.cfg.calibration.scale_per_pixel * 1.5

            for idx, img_path in enumerate(files):
                img_bgr, img_rgb = self.preprocessor.load_and_rectify_image(img_path)
                denoised = self.preprocessor.enhance_and_denoise(img_bgr)
                pcd, _, _ = self.surface_reconstructor.build_calibrated_point_cloud(img_rgb, denoised)

                if idx > 0 and len(combined_pcd.points) > 0:
                    reg = o3d.pipelines.registration.registration_icp(
                        pcd,
                        combined_pcd,
                        max_correspondence_distance=voxel_size * 5.0,
                        init=np.eye(4),
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    )
                    pcd.transform(reg.transformation)
                combined_pcd += pcd

            combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
            raw_mesh = self.surface_reconstructor.reconstruct_from_point_cloud(combined_pcd)
            report = self._finalize_and_export(raw_mesh, "merged_surface_model")
            reports.append(report)
        else:
            raise PipelineError(
                f"Kombinasi mode='{self.cfg.mode}' dan multi_strategy='{self.cfg.multi_strategy}' tidak didukung."
            )

        return reports

    def _process_single_file(self, img_path: Path, z_offset: float) -> trimesh.Trimesh:
        img_bgr, img_rgb = self.preprocessor.load_and_rectify_image(img_path)
        denoised_gray = self.preprocessor.enhance_and_denoise(img_bgr)

        if self.cfg.mode == "cad":
            denoised_bgr = self.preprocessor.enhance_color_preserve_edges(img_bgr)
            binary = self.preprocessor.binarize_high_precision(denoised_gray)
            return self.cad_reconstructor.reconstruct(
                binary_mask=binary,
                denoised_gray=denoised_gray,
                denoised_bgr=denoised_bgr,
                img_rgb=img_rgb,
                z_offset=z_offset,
            )
        else:
            pcd, z_grid, rgb_grid = self.surface_reconstructor.build_calibrated_point_cloud(
                img_rgb=img_rgb,
                denoised_gray=denoised_gray,
            )
            eff_scale = self.cfg.calibration.scale_per_pixel * (
                img_rgb.shape[1] / float(z_grid.shape[1])
            )
            mesh = self.surface_reconstructor.reconstruct_from_point_cloud(
                pcd=pcd,
                z_grid=z_grid,
                rgb_grid=rgb_grid,
                eff_scale=eff_scale,
            )
            if z_offset != 0.0:
                mesh.apply_translation([0.0, 0.0, z_offset])
            return mesh

    def _finalize_and_export(self, raw_mesh: trimesh.Trimesh, base_name: str) -> ValidationReport:
        clean_tm, clean_o3d = self.optimizer.clean_and_optimize(raw_mesh)

        report = MeshValidator.validate(
            model_name=base_name,
            mesh=clean_tm,
            o3d_mesh=clean_o3d,
            unit=self.cfg.calibration.unit,
        )

        if self.cfg.preview.enabled and self.cfg.preview.confirm_before_export:
            MeshValidator.print_report(report)
            approved = InteractiveMeshPreviewer.show(
                o3d_mesh=clean_o3d,
                report=report,
                preview_cfg=self.cfg.preview,
                output_dir=self.cfg.output_dir,
            )
            if not approved:
                return report

        exported_files = self.exporter.export_all(clean_tm, clean_o3d, base_name)
        report.exported_files = exported_files
        MeshValidator.print_report(report)

        if self.cfg.preview.enabled and not self.cfg.preview.confirm_before_export:
            InteractiveMeshPreviewer.show(
                o3d_mesh=clean_o3d,
                report=report,
                preview_cfg=self.cfg.preview,
                output_dir=self.cfg.output_dir,
            )

        return report


# ==============================================================================
# PARSING & ANTARMUKA CLI (--help TERSTRUKTUR)
# ==============================================================================


class SmartHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Formatter CLI yang mempertahankan format multi-baris sekaligus menampilkan nilai default."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, indent_increment=2, max_help_position=38, width=104)


def resolve_input_files(raw_inputs: Sequence[str]) -> List[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    resolved: List[Path] = []

    for item in raw_inputs:
        p = Path(item)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.suffix.lower() in valid_exts:
                    resolved.append(child)
        elif "*" in item or "?" in item:
            for match in sorted(glob.glob(item)):
                mp = Path(match)
                if mp.is_file() and mp.suffix.lower() in valid_exts:
                    resolved.append(mp)
        elif p.is_file():
            resolved.append(p)
        else:
            raise FileNotFoundError(f"Input tidak ditemukan: {item}")

    if not resolved:
        raise FileNotFoundError("Tidak ada file gambar valid (JPG/PNG) yang ditemukan.")
    return resolved


def parse_points_string(pts_str: str) -> np.ndarray:
    pairs = [seg.strip() for seg in pts_str.split(";") if seg.strip()]
    if len(pairs) != 4:
        raise CalibrationError(
            f"--homography-points harus berisi tepat 4 titik dipisah ';' (ditemukan: {len(pairs)})."
        )
    coords: List[Tuple[float, float]] = []
    for pair in pairs:
        xy = pair.split(",")
        if len(xy) != 2:
            raise CalibrationError(f"Format koordinat tidak valid: '{pair}'. Gunakan format 'x,y'.")
        coords.append((float(xy[0]), float(xy[1])))
    return np.array(coords, dtype=np.float32)


def parse_wx_h(size_str: str, sep: str = "x") -> Tuple[float, float]:
    parts = size_str.lower().split(sep)
    if len(parts) != 2:
        raise CalibrationError(f"Format ukuran tidak valid: '{size_str}'. Gunakan format 'WxH'.")
    return float(parts[0]), float(parts[1])


def parse_aruco_ids(ids_str: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in ids_str.split(",") if p.strip()]
    if len(parts) != 4:
        raise CalibrationError(
            f"--aruco-ids harus berisi tepat 4 ID dipisah koma [TL,TR,BR,BL] (ditemukan: {len(parts)})."
        )
    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))


def parse_color_map_string(map_str: str) -> List[ColorLayerRule]:
    entries = [item.strip() for item in map_str.split(",") if item.strip()]
    if not entries:
        raise CalibrationError("--color-map tidak boleh kosong.")

    rules: List[ColorLayerRule] = []
    for idx, entry in enumerate(entries):
        if "=" in entry:
            name_part, rest = entry.split("=", 1)
            layer_name = name_part.strip()
        else:
            layer_name = f"ColorLayer_{idx + 1}"
            rest = entry

        if ":" not in rest:
            raise CalibrationError(
                f"Format aturan warna tidak valid: '{entry}'. Gunakan '#RRGGBB:tinggi' atau '#RRGGBB:z_base+tinggi'."
            )
        hex_part, z_spec = rest.split(":", 1)
        rgb = hex_to_rgb(hex_part)

        z_spec = z_spec.strip()
        if "+" in z_spec:
            base_str, height_str = z_spec.split("+", 1)
            z_base = float(base_str)
            height = float(height_str)
        else:
            z_base = 0.0
            height = float(z_spec)

        rules.append(
            ColorLayerRule(
                name=layer_name,
                rgb=rgb,
                extrude_height=height,
                z_base=z_base,
            )
        )
    return rules


def parse_color_map_json(json_path: Path) -> List[ColorLayerRule]:
    if not json_path.exists():
        raise FileNotFoundError(f"File JSON color-map tidak ditemukan: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) == 0:
        raise CalibrationError("File JSON --color-map-file harus berisi array objek aturan warna.")

    rules: List[ColorLayerRule] = []
    for idx, item in enumerate(data):
        rgb = hex_to_rgb(str(item["hex"]))
        rules.append(
            ColorLayerRule(
                name=str(item.get("name", f"ColorLayer_{idx + 1}")),
                rgb=rgb,
                extrude_height=float(item["height"]),
                z_base=float(item.get("z_base", 0.0)),
            )
        )
    return rules


def build_argument_parser() -> argparse.ArgumentParser:
    """Membangun parser CLI lengkap dengan dokumentasi --help dan contoh penggunaan."""
    description = textwrap.dedent(
        """\
        ====================================================================================================
         Img2Mesh3D Pipeline - Rekonstruksi Presisi Tinggi Citra 2D (JPG/PNG) ke Model Solid 3D (1:1)
        ====================================================================================================
        Mengonversi 1 atau banyak citra 2D menjadi mesh 3D watertight & manifold terkalibrasi secara
        metrik (mendukung kalibrasi otomatis marker ArUco cv2.aruco), lalu mengekspornya ke 4 format:
        .STL, .OBJ (+Multi-Material .MTL), .DAE (COLLADA Z-up siap SketchUp), dan .PLY (Vertex Color).
        """
    )

    epilog = textwrap.dedent(
        """\
        ----------------------------------------------------------------------------------------------------
        KONTROL INTERAKTIF SAAT --preview AKTIF (OPEN3D VISUALIZER):
          [Mouse Kiri] Rotasi 3D       | [Scroll] Zoom             | [Shift+Drag / Tengah] Pan
          [W] Toggle Wireframe         | [B] Toggle Back-Face      | [G] Toggle Grid Z=0 & Sumbu XYZ
          [R] Reset Kamera             | [S] Simpan Screenshot PNG | [Q / ESC] Tutup Jendela

        CONTOH PERINTAH CLI SIAP PAKAI:
          1. Kalibrasi Otomatis ArUco 4-Sudut + Multi-Color CAD Denah (Tanpa Klik Manual):
             python img2mesh_pipeline.py -i foto_denah_aruco.jpg -o ./out --mode cad \\
               --aruco-board-size 210x297 --aruco-ids 0,1,2,3 --unit mm \\
               --color-map "Dinding=#FF0000:35.0, Jendela=#0000FF:9.0" --aruco-debug --preview

          2. Kalibrasi Otomatis ArUco Marker Tunggal (1 Marker Persegi di Samping Objek):
             python img2mesh_pipeline.py -i foto_komponen.jpg -o ./out --mode cad \\
               --aruco-marker-size 50.0 --unit mm --extrude-height 12.0 --preview

          3. Cetak Lembar Referensi ArUco 4-Sudut (Ukuran A4 = 210x297 mm):
             python img2mesh_pipeline.py --generate-aruco-sheet ./aruco_a4.png \\
               --aruco-board-size 210x297 --aruco-marker-size 30 --unit mm

          4. Mode A (CAD Standar) dengan Skala Manual Piksel:
             python img2mesh_pipeline.py -i denah.png -o ./out --mode cad \\
               --ref-real-size 6.0 --ref-pixels 1200 --unit m --extrude-height 3.5 --preview

          5. Mode B (Surface / Relief) via Depth Anything V2 + ArUco Otomatis:
             python img2mesh_pipeline.py -i relief_aruco.jpg -o ./out --mode surface \\
               --depth-backend depth-anything --aruco-marker-size 40.0 --unit mm \\
               --extrude-height 15.0 --preview
        ----------------------------------------------------------------------------------------------------
        """
    )

    parser = argparse.ArgumentParser(
        prog="img2mesh_pipeline.py",
        description=description,
        epilog=epilog,
        formatter_class=SmartHelpFormatter,
        add_help=False,
    )

    # --- Grup 0: Bantuan & Informasi Versi ---
    help_group = parser.add_argument_group("0. Bantuan & Informasi Program")
    help_group.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Tampilkan panduan lengkap parameter CLI, kontrol pratinjau 3D, dan contoh perintah",
    )
    help_group.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Tampilkan versi pipeline Img2Mesh3D",
    )

    # --- Grup 1: Input & Output Utama ---
    io_group = parser.add_argument_group("1. Input, Mode Rekonstruksi & Direktori Output")
    io_group.add_argument(
        "-i", "--input",
        nargs="+",
        metavar="PATH",
        help="Path 1 file gambar (JPG/PNG), multi-file, folder, atau pola wildcard (mis. *.png)",
    )
    io_group.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./output_3d"),
        metavar="DIR",
        help="Direktori penyimpanan hasil ekspor (.STL, .OBJ, .MTL, .DAE, .PLY)",
    )
    io_group.add_argument(
        "-m", "--mode",
        choices=["cad", "surface"],
        default="cad",
        help="Mode rekonstruksi: 'cad' (Mode A: kontur & ekstrusi) atau 'surface' (Mode B: depth/relief)",
    )
    io_group.add_argument(
        "--multi-strategy",
        choices=["batch", "stack", "merge"],
        default="batch",
        help="Strategi multi-file: 'batch' (terpisah), 'stack' (tumpuk layer Z Mode A), 'merge' (ICP Mode B)",
    )

    # --- Grup 2: Kalibrasi Skala Metrik ---
    scale_group = parser.add_argument_group("2. Kalibrasi Skala Metrik (Pixel-to-Real-World)")
    scale_group.add_argument(
        "--ref-real-size",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Ukuran fisik referensi nyata (wajib jika tidak memakai --aruco-* atau --homography-size)",
    )
    scale_group.add_argument(
        "--ref-pixels",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Panjang objek referensi dalam piksel (wajib jika tidak memakai --aruco-* atau --homography-size)",
    )
    scale_group.add_argument(
        "--unit",
        choices=["mm", "cm", "m"],
        default="mm",
        help="Satuan fisik model 3D akhir (disematkan juga ke metadata COLLADA .DAE SketchUp)",
    )
    scale_group.add_argument(
        "--extrude-height",
        type=float,
        default=5.0,
        metavar="FLOAT",
        help="Tinggi ekstrusi Z default (Mode A) atau amplitudo relief maksimum (Mode B)",
    )
    scale_group.add_argument(
        "--invert",
        action="store_true",
        help="Balik polaritas objek (gunakan jika objek berwarna putih/terang di atas latar gelap)",
    )

    # --- Grup 3: Kalibrasi Matriks Intrinsik Kamera ---
    calib_group = parser.add_argument_group("3. Kalibrasi Matriks Intrinsik Kamera (Tahap 0A)")
    calib_group.add_argument(
        "--calibrate-chessboard",
        nargs="+",
        metavar="IMG",
        help="Daftar/folder foto papan catur untuk menghitung matriks intrinsik kamera K & distorsi",
    )
    calib_group.add_argument(
        "--chessboard-size",
        type=str,
        default="9x6",
        metavar="CxR",
        help="Jumlah sudut dalam papan catur (KolomxBaris)",
    )
    calib_group.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        metavar="FLOAT",
        help="Ukuran sisi satu kotak papan catur dalam satuan nyata",
    )
    calib_group.add_argument(
        "--save-calib",
        type=Path,
        default=Path("./camera_intrinsics.json"),
        metavar="JSON",
        help="Path file JSON keluaran hasil kalibrasi kamera",
    )
    calib_group.add_argument(
        "--camera-calib",
        type=Path,
        default=None,
        metavar="JSON",
        help="Path file JSON kalibrasi intrinsik kamera untuk menghilangkan distorsi lensa",
    )
    calib_group.add_argument(
        "--intrinsics-manual",
        type=str,
        default=None,
        metavar="PARAMS",
        help="Parameter intrinsik manual 'fx,fy,cx,cy,k1,k2,p1,p2[,k3]' jika tanpa file JSON",
    )

    # --- Grup 4A: Kalibrasi Otomatis Marker ArUco (cv2.aruco) ---
    aruco_group = parser.add_argument_group("4A. Deteksi Otomatis Marker ArUco - Homografi & Skala 1:1 (Tahap 0B)")
    aruco_group.add_argument(
        "--aruco-board-size",
        type=str,
        default=None,
        metavar="WxH",
        help="Mode 4-Sudut: dimensi fisik bidang referensi 4 marker ArUco 'LebarxTinggi' (mis. 210x297)",
    )
    aruco_group.add_argument(
        "--aruco-marker-size",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Mode Marker Tunggal: panjang sisi fisik S satu marker persegi ArUco dalam satuan --unit",
    )
    aruco_group.add_argument(
        "--aruco-dict",
        choices=list(SUPPORTED_ARUCO_DICTS),
        default="DICT_4X4_50",
        metavar="DICT",
        help="Kamus ArUco OpenCV (mis. DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_APRILTAG_36h11)",
    )
    aruco_group.add_argument(
        "--aruco-ids",
        type=str,
        default="0,1,2,3",
        metavar="IDs",
        help="4 ID marker sudut referensi berurutan [TL,TR,BR,BL] untuk --aruco-board-size",
    )
    aruco_group.add_argument(
        "--aruco-target-id",
        type=int,
        default=None,
        metavar="ID",
        help="ID spesifik marker yang dipilih pada mode --aruco-marker-size (default: marker terbesar)",
    )
    aruco_group.add_argument(
        "--aruco-anchor",
        choices=["outer", "inner", "center"],
        default="outer",
        help="Titik jangkar 4 marker sudut: 'outer' (sudut luar), 'inner' (sudut dalam), 'center'",
    )
    aruco_group.add_argument(
        "--aruco-debug",
        action="store_true",
        help="Simpan gambar overlay deteksi marker & titik homografi (*_aruco_detected.png)",
    )
    aruco_group.add_argument(
        "--generate-aruco-sheet",
        type=Path,
        default=None,
        metavar="PNG",
        help="Buat file PNG lembar kalibrasi ArUco 4-sudut siap cetak lalu keluar (atau lanjut proses)",
    )

    # --- Grup 4B: Koreksi Perspektif Homografi 4-Titik Manual/Interaktif ---
    homo_group = parser.add_argument_group("4B. Koreksi Perspektif Homografi 4-Titik Manual / Interaktif (Tahap 0B)")
    homo_group.add_argument(
        "--homography-points",
        type=str,
        default=None,
        metavar="PTS",
        help="4 titik sudut referensi piksel 'x1,y1;x2,y2;x3,y3;x4,y4' berurutan [TL, TR, BR, BL]",
    )
    homo_group.add_argument(
        "--pick-points",
        action="store_true",
        help="Pilih 4 titik sudut referensi secara interaktif dengan klik mouse pada jendela gambar",
    )
    homo_group.add_argument(
        "--homography-size",
        type=str,
        default=None,
        metavar="WxH",
        help="Dimensi fisik nyata bidang 4 titik 'LebarxTinggi' dalam satuan --unit (mis. 210x297)",
    )
    homo_group.add_argument(
        "--homography-crop",
        action="store_true",
        help="Potong output hanya di dalam area 4 titik (berlaku untuk ArUco & Homografi manual)",
    )

    # --- Grup 5: Mode A (CAD & Multi-Color Layer Extrusion) ---
    cad_group = parser.add_argument_group("5. Mode A - Presisi CAD & Multi-Color Layer Extrusion (Tahap 2A)")
    cad_group.add_argument(
        "--rdp-epsilon",
        type=float,
        default=0.35,
        metavar="PX",
        help="Toleransi error aproksimasi poligon Ramer-Douglas-Peucker dalam piksel",
    )
    cad_group.add_argument(
        "--min-area",
        type=float,
        default=25.0,
        metavar="PX2",
        help="Luas kontur minimum dalam piksel persegi untuk menyaring bintik noise",
    )
    cad_group.add_argument(
        "--layer-gap",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Jarak celah vertikal antar layer saat menggunakan --multi-strategy stack",
    )
    cad_group.add_argument(
        "--color-map",
        type=str,
        default=None,
        metavar="SPEC",
        help="Pemetaan warna HEX ke tinggi Z, mis. 'Dinding=#FF0000:3.5, Lintel=#00FFFF:2.1+1.4'",
    )
    cad_group.add_argument(
        "--color-map-file",
        type=Path,
        default=None,
        metavar="JSON",
        help="Path file JSON berisi daftar aturan warna [{'name','hex','z_base','height'}]",
    )
    cad_group.add_argument(
        "--auto-color-layers",
        type=int,
        default=None,
        metavar="K",
        help="Deteksi otomatis K klaster warna dominan pada denah dan ekstrusi ke tinggi Z berbeda",
    )
    cad_group.add_argument(
        "--auto-heights",
        type=str,
        default=None,
        metavar="LIST",
        help="Daftar tinggi Z untuk --auto-color-layers dipisah koma (mis. '3.5,2.1,1.0,0.3')",
    )
    cad_group.add_argument(
        "--color-tolerance",
        type=float,
        default=22.0,
        metavar="DE",
        help="Toleransi jarak warna perseptual CIE Delta-E76 untuk pencocokan piksel",
    )
    cad_group.add_argument(
        "--bg-threshold",
        type=float,
        default=15.0,
        metavar="DE",
        help="Ambang CIE Delta-E76 untuk memisahkan geometri berwarna dari latar belakang",
    )

    # --- Grup 6: Mode B (Surface / Relief & Depth Map) ---
    surf_group = parser.add_argument_group("6. Mode B - Presisi Permukaan / Relief & Depth Map (Tahap 2B)")
    surf_group.add_argument(
        "--depth-backend",
        choices=["heightmap", "depth-anything", "midas"],
        default="heightmap",
        help="Backend estimasi kedalaman: 'heightmap', 'depth-anything', atau 'midas'",
    )
    surf_group.add_argument(
        "--surface-method",
        choices=["poisson", "alpha", "grid-solid"],
        default="poisson",
        help="Algoritma rekonstruksi permukaan: 'poisson', 'alpha', atau 'grid-solid'",
    )
    surf_group.add_argument(
        "--poisson-depth",
        type=int,
        default=9,
        metavar="INT",
        help="Kedalaman octree Screened Poisson Surface Reconstruction (8-11)",
    )
    surf_group.add_argument(
        "--no-solid-base",
        action="store_true",
        help="Jangan tutup alas bawah Z=0 pada Mode B (menghasilkan open surface mesh)",
    )

    # --- Grup 7: Mesh Cleaning & Topology Optimization ---
    opt_group = parser.add_argument_group("7. Pembersihan Mesh & Optimasi Topologi (Tahap 3)")
    opt_group.add_argument(
        "--taubin-iter",
        type=int,
        default=5,
        metavar="INT",
        help="Jumlah iterasi Taubin smoothing bebas-penyusutan volume pada Mode B",
    )
    opt_group.add_argument(
        "--decimate",
        type=float,
        default=0.0,
        metavar="RATIO",
        help="Rasio reduksi segitiga (0.0 - 0.90) dengan Quadric Edge Collapse",
    )

    # --- Grup 8: Validasi & Pratinjau Interaktif Open3D ---
    prev_group = parser.add_argument_group("8. Validasi & Pratinjau Mesh 3D Interaktif Open3D (Tahap 5)")
    prev_group.add_argument(
        "--preview",
        action="store_true",
        help="Buka jendela Open3D Visualizer interaktif pada tahap validasi akhir",
    )
    prev_group.add_argument(
        "--preview-wireframe",
        action="store_true",
        help="Aktifkan tampilan wireframe segitiga secara default saat pratinjau dibuka",
    )
    prev_group.add_argument(
        "--confirm-export",
        action="store_true",
        help="Tampilkan pratinjau dulu dan minta konfirmasi terminal [Y/n] sebelum menyimpan file",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        corner_ids = parse_aruco_ids(args.aruco_ids)

        # Utilitas 1: Buat Lembar Kalibrasi ArUco PNG jika diminta
        if args.generate_aruco_sheet is not None:
            sheet_board = (
                parse_wx_h(args.aruco_board_size)
                if args.aruco_board_size is not None
                else (210.0, 297.0)
            )
            sheet_marker = args.aruco_marker_size if args.aruco_marker_size is not None else 30.0
            ArucoAutoCalibrator.generate_calibration_sheet(
                output_path=args.generate_aruco_sheet,
                dict_name=args.aruco_dict,
                board_size_units=sheet_board,
                marker_size_units=sheet_marker,
                corner_ids=corner_ids,
                unit=args.unit,
            )
            if not args.input and not args.calibrate_chessboard:
                return 0

        # Utilitas 2: Kalibrasi Kamera Papan Catur
        if args.calibrate_chessboard:
            calib_imgs = resolve_input_files(args.calibrate_chessboard)
            cols_f, rows_f = parse_wx_h(args.chessboard_size)
            CameraIntrinsicsCalibrator.calibrate_from_chessboard(
                image_paths=calib_imgs,
                board_size=(int(cols_f), int(rows_f)),
                square_size=args.square_size,
                save_path=args.save_calib,
            )
            if not args.input:
                return 0

        if not args.input:
            raise CalibrationError(
                "Argumen --input wajib diisi untuk menjalankan rekonstruksi 3D. "
                "Jalankan dengan --help untuk melihat panduan lengkap."
            )

        input_files = resolve_input_files(args.input)

        intrinsics_cfg: Optional[CameraIntrinsicsConfig] = None
        if args.camera_calib is not None:
            intrinsics_cfg = CameraIntrinsicsConfig.from_json(args.camera_calib)
        elif args.intrinsics_manual is not None:
            vals = [float(v.strip()) for v in args.intrinsics_manual.split(",") if v.strip()]
            if len(vals) < 8:
                raise CalibrationError("--intrinsics-manual membutuhkan minimal 8 nilai: fx,fy,cx,cy,k1,k2,p1,p2[,k3].")
            intrinsics_cfg = CameraIntrinsicsConfig.from_manual_params(
                fx=vals[0], fy=vals[1], cx=vals[2], cy=vals[3], dist_coeffs=vals[4:]
            )

        # Konfigurasi Deteksi Otomatis ArUco (cv2.aruco)
        aruco_cfg: Optional[ArucoConfig] = None
        if args.aruco_board_size is not None or args.aruco_marker_size is not None:
            board_wh = parse_wx_h(args.aruco_board_size) if args.aruco_board_size is not None else None
            aruco_cfg = ArucoConfig(
                dict_name=args.aruco_dict,
                board_real_size=board_wh,
                marker_real_size=args.aruco_marker_size,
                corner_ids=corner_ids,
                target_single_id=args.aruco_target_id,
                anchor_mode=args.aruco_anchor,
                keep_full_canvas=not args.homography_crop,
                save_debug_overlay=args.aruco_debug,
            )

        # Konfigurasi Homografi Manual/Interaktif
        homography_cfg: Optional[HomographyConfig] = None
        if args.homography_points is not None or args.pick_points:
            if args.homography_size is None:
                raise CalibrationError("--homography-size wajib diisi saat menggunakan homografi 4-titik.")
            h_w, h_h = parse_wx_h(args.homography_size)
            pts = (
                parse_points_string(args.homography_points)
                if args.homography_points is not None
                else np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
            )
            homography_cfg = HomographyConfig(
                src_points=pts,
                rect_real_width=h_w,
                rect_real_height=h_h,
                keep_full_canvas=not args.homography_crop,
            )

        has_metric_rectification = (aruco_cfg is not None and aruco_cfg.is_enabled) or (
            homography_cfg is not None
        )
        if has_metric_rectification:
            calib_cfg = CalibrationConfig(
                ref_real_size=1.0,
                ref_pixels=1.0,
                unit=args.unit,
                extrude_height=args.extrude_height,
                invert_polarity=args.invert,
            )
        else:
            if args.ref_real_size is None or args.ref_pixels is None:
                raise CalibrationError(
                    "Isi --ref-real-size dan --ref-pixels, ATAU aktifkan kalibrasi otomatis ArUco (--aruco-board-size / --aruco-marker-size) atau homografi 4-titik."
                )
            calib_cfg = CalibrationConfig(
                ref_real_size=args.ref_real_size,
                ref_pixels=args.ref_pixels,
                unit=args.unit,
                extrude_height=args.extrude_height,
                invert_polarity=args.invert,
            )

        color_rules: List[ColorLayerRule] = []
        if args.color_map is not None:
            color_rules.extend(parse_color_map_string(args.color_map))
        if args.color_map_file is not None:
            color_rules.extend(parse_color_map_json(args.color_map_file))

        auto_heights: Optional[List[float]] = None
        if args.auto_heights is not None:
            auto_heights = [float(x.strip()) for x in args.auto_heights.split(",") if x.strip()]

        color_cad_cfg = ColorCadConfig(
            rules=color_rules,
            auto_num_colors=args.auto_color_layers,
            auto_heights=auto_heights,
            delta_e_tolerance=args.color_tolerance,
            bg_delta_e_threshold=args.bg_threshold,
        )

        preview_cfg = PreviewConfig(
            enabled=args.preview or args.confirm_export,
            wireframe=args.preview_wireframe,
            confirm_before_export=args.confirm_export,
        )

        config = PipelineConfig(
            input_paths=input_files,
            output_dir=args.output_dir,
            mode=args.mode,
            multi_strategy=args.multi_strategy,
            calibration=calib_cfg,
            camera_intrinsics=intrinsics_cfg,
            aruco=aruco_cfg,
            homography=homography_cfg,
            interactive_pick_points=args.pick_points,
            rdp_epsilon_px=args.rdp_epsilon,
            min_contour_area_px=args.min_area,
            layer_gap=args.layer_gap,
            color_cad=color_cad_cfg,
            depth_backend=args.depth_backend,
            surface_method=args.surface_method,
            poisson_depth=args.poisson_depth,
            close_solid_base=not args.no_solid_base,
            taubin_iterations=args.taubin_iter,
            target_reduction=args.decimate,
            preview=preview_cfg,
        )

        pipeline = ImageTo3DPipeline(config)
        pipeline.run()
        return 0

    except PipelineError as exc:
        logger.error("Kesalahan Pipeline Geometri: %s", exc)
        return 1
    except FileNotFoundError as exc:
        logger.error("Kesalahan Berkas Input: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Proses dibatalkan oleh pengguna.")
        return 130
    except Exception as exc:
        logger.exception("Kesalahan tak terduga: %s", exc)
        return 99


if __name__ == "__main__":
    sys.exit(main())
