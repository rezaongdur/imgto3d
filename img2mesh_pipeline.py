#!/usr/bin/env python3
"""
End-to-End 2D Image(s) to Calibrated 3D Solid Mesh Reconstruction Pipeline
(Dilengkapi Kalibrasi Intrinsik Kamera, Homografi 4-Titik, & Open3D Interactive Preview).

Tahapan Pipeline:
  - Tahap 0A: Kalibrasi & Undistorsi Matriks Intrinsik Kamera (Brown-Conrady / Zhang)
  - Tahap 0B: Koreksi Perspektif Homografi 4-Titik (Metric Bird's-Eye Rectification)
  - Tahap 1 : Preprocessing Adaptif (CLAHE + Bilateral Filter) & Kalibrasi Skala 1:1
  - Tahap 2 : Rekonstruksi 3D Mode A (CAD Kontur Sub-Piksel + Ekstrusi) atau
              Mode B (Depth Anything V2 / MiDaS / Heightmap + Poisson / Alpha Shapes)
  - Tahap 3 : Mesh Cleaning & Topology Optimization (Watertight, Manifold, Taubin)
  - Tahap 4 : Ekspor Multi-Format (.STL, .OBJ + .MTL, .DAE Z-up SketchUp, .PLY)
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
    """Dilempar apabila parameter kalibrasi kamera, homografi, atau skala tidak valid."""


class GeometryExtractionError(PipelineError):
    """Dilempar apabila tidak ada kontur atau permukaan valid yang diekstrak."""


class MeshTopologyError(PipelineError):
    """Dilempar apabila perbaikan topologi gagal menghasilkan mesh valid."""


# ==============================================================================
# STRUKTUR DATA KONFIGURASI & LAPORAN VALIDASI
# ==============================================================================

UnitType = Literal["mm", "cm", "m"]
ModeType = Literal["cad", "surface"]
MultiFileStrategy = Literal["batch", "stack", "merge"]
DepthBackend = Literal["heightmap", "depth-anything", "midas"]
SurfaceMethod = Literal["poisson", "alpha", "grid-solid"]


@dataclass
class CameraIntrinsicsConfig:
    """Parameter matriks intrinsik kamera K dan koefisien distorsi lensa."""

    camera_matrix: np.ndarray  # Bentuk (3, 3) float64
    dist_coeffs: np.ndarray    # Bentuk (N,) float64: [k1, k2, p1, p2, k3]
    calib_image_size: Optional[Tuple[int, int]] = None  # (width, height) saat kalibrasi
    alpha: float = 0.0  # 0.0 = crop piksel hitam tepi, 1.0 = pertahankan semua piksel

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

    src_points: np.ndarray        # Bentuk (4, 2) float32: [TL, TR, BR, BL] dalam piksel
    rect_real_width: float        # Lebar fisik nyata bidang referensi (dalam satuan unit)
    rect_real_height: float       # Tinggi fisik nyata bidang referensi (dalam satuan unit)
    keep_full_canvas: bool = True # True = warp seluruh kanvas, False = crop ke 4 titik
    target_ppu: Optional[float] = None  # Piksel per unit fisik (None = hitung otomatis)

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
        """Mengembalikan faktor konversi fisik (unit dunia nyata per piksel)."""
        return self.ref_real_size / self.ref_pixels

    @property
    def meters_per_unit(self) -> float:
        """Konversi unit ke meter untuk metadata COLLADA (.DAE)."""
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
    # Tahap 0: Camera Intrinsics & Homography Perspective Rectification
    camera_intrinsics: Optional[CameraIntrinsicsConfig] = None
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
    # Tahap 2A: CAD Parameters
    rdp_epsilon_px: float = 0.35
    min_contour_area_px: float = 25.0
    layer_gap: float = 0.0
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
    """
    Menangani kalibrasi kamera Zhang menggunakan pola papan catur (chessboard)
    serta koreksi distorsi radial dan tangensial pada citra input.
    """

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
        undistorted = cv2.remap(
            img_bgr, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
        )
        return undistorted


# ==============================================================================
# TAHAP 0B: KOREKSI PERSPEKTIF HOMOGRAFI 4-TITIK (METRIC RECTIFICATION)
# ==============================================================================


class HomographyPerspectiveRectifier:
    """
    Meluruskan foto miring (oblique perspective) menjadi tampak atas ortografis
    (bird's-eye view) 1:1 menggunakan transformasi homografi planar 4-titik.
    """

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
                f"Gagal membuka GUI OpenCV untuk --pick-points (gunakan --homography-points pada server headless): {exc}"
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
            [
                [0.0, 0.0],
                [dst_w, 0.0],
                [dst_w, dst_h],
                [0.0, dst_h],
            ],
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
                    f"Ukuran kanvas hasil homografi full-canvas tidak wajar ({out_w}x{out_h} px). "
                    "Gunakan --homography-crop atau periksa urutan titik [TL, TR, BR, BL]."
                )

            t_mat = np.array(
                [
                    [1.0, 0.0, -min_xy[0]],
                    [0.0, 1.0, -min_xy[1]],
                    [0.0, 0.0, 1.0],
                ],
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
    """
    Menangani pemuatan citra, koreksi lensa & perspektif (Tahap 0),
    koreksi kontras adaptif CLAHE, bilateral filtering, dan binarisasi.
    """

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
            white_bg = np.full_like(bgr, 255.0)
            img_bgr = np.clip(bgr * alpha + white_bg * (1.0 - alpha), 0, 255).astype(np.uint8)
        elif img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        if self.cfg.camera_intrinsics is not None:
            logger.info("Menerapkan koreksi distorsi lensa dari matriks intrinsik kamera...")
            img_bgr = CameraIntrinsicsCalibrator.undistort_image(img_bgr, self.cfg.camera_intrinsics)

        if self.cfg.homography is not None:
            if self.cfg.interactive_pick_points:
                picked = HomographyPerspectiveRectifier.interactive_pick_4_points(img_bgr)
                self.cfg.homography.src_points = picked

            img_bgr, new_scale_per_px, _ = HomographyPerspectiveRectifier.rectify(
                img_bgr, self.cfg.homography
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

        denoised = cv2.bilateralFilter(
            gray,
            d=self.cfg.bilateral_d,
            sigmaColor=self.cfg.bilateral_sigma_color,
            sigmaSpace=self.cfg.bilateral_sigma_space,
        )
        return denoised

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
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary


# ==============================================================================
# TAHAP 2A: REKONSTRUKSI CAD / ORTOGRAFIS (SUB-PIXEL CONTOUR + EXTRUSION)
# ==============================================================================


class CadOrthographicReconstructor:
    """
    Mode A: Mengubah citra biner terkalibrasi menjadi mesh solid CAD 3D
    menggunakan ekstraksi kontur hierarkis, Ramer-Douglas-Peucker,
    validasi poligon Shapely, dan triangulasi CDT Trimesh.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

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
        self, binary_mask: np.ndarray, denoised_gray: np.ndarray
    ) -> List[Polygon]:
        h, _ = binary_mask.shape[:2]
        scale = self.cfg.calibration.scale_per_pixel

        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if not contours or hierarchy is None:
            raise GeometryExtractionError(
                "Tidak ditemukan kontur geometris pada gambar. Periksa polaritas binarization (--invert)."
            )

        hier = hierarchy[0]
        raw_polygons: List[Polygon] = []

        for idx, cnt in enumerate(contours):
            if hier[idx][3] != -1:
                continue

            area_px = cv2.contourArea(cnt)
            if area_px < self.cfg.min_contour_area_px:
                continue

            approx_ext = cv2.approxPolyDP(cnt, epsilon=self.cfg.rdp_epsilon_px, closed=True)
            refined_ext = self._refine_contours_subpixel(denoised_gray, [approx_ext])
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
                    refined_hole = self._refine_contours_subpixel(denoised_gray, [approx_hole])
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

        if not raw_polygons:
            raise GeometryExtractionError(
                "Seluruh kontur terfilter oleh ambang luas minimum atau tidak valid secara topologis."
            )

        return raw_polygons

    def reconstruct(
        self,
        binary_mask: np.ndarray,
        denoised_gray: np.ndarray,
        img_rgb: np.ndarray,
        z_offset: float = 0.0,
    ) -> trimesh.Trimesh:
        polygons = self._extract_shapely_polygons(binary_mask, denoised_gray)
        extruded_parts: List[trimesh.Trimesh] = []
        height = self.cfg.calibration.extrude_height

        for poly in polygons:
            try:
                mesh_part = trimesh.creation.extrude_polygon(poly, height=height)
                if z_offset != 0.0:
                    mesh_part.apply_translation([0.0, 0.0, z_offset])
                extruded_parts.append(mesh_part)
            except Exception as exc:
                logger.warning("Melewati satu sub-poligon karena gagal triangulasi CDT: %s", exc)

        if not extruded_parts:
            raise GeometryExtractionError("Gagal melakukan ekstrusi 3D pada seluruh poligon.")

        combined = (
            extruded_parts[0]
            if len(extruded_parts) == 1
            else trimesh.util.concatenate(extruded_parts)
        )

        self._apply_projected_vertex_colors(combined, img_rgb)
        return combined

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

        # Sinkronkan kembali normal dan orientasi segitiga final ke o3d_mesh untuk pratinjau & PLY
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(final_tm.vertices, dtype=np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(final_tm.faces, dtype=np.int32))
        o3d_mesh.compute_vertex_normals()

        return final_tm, o3d_mesh


# ==============================================================================
# TAHAP 4: MULTI-FORMAT EXPORTER (.STL, .OBJ + .MTL, .DAE SKETCHUP, .PLY)
# ==============================================================================


class MultiFormatMeshExporter:
    """
    Mengekspor mesh 3D terkalibrasi ke 4 format standar industri sekaligus:
      1. .STL (Binary solid manifold)
      2. .OBJ + .MTL (dengan normal, UV terproyeksi, dan definisi material)
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
        self._export_obj_with_mtl(mesh, obj_path, base_name)
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

    def _export_obj_with_mtl(self, mesh: trimesh.Trimesh, obj_path: Path, base_name: str) -> None:
        mtl_filename = f"{base_name}.mtl"
        mtl_path = obj_path.with_suffix(".mtl")

        kd = (0.78, 0.80, 0.84)
        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) > 0
        ):
            mean_rgb = np.mean(mesh.visual.vertex_colors[:, :3], axis=0) / 255.0
            kd = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))

        mtl_content = (
            f"# Material file generated by Img2Mesh3D Pipeline\n"
            f"newmtl material_calibrated\n"
            f"Ka 0.150000 0.150000 0.150000\n"
            f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}\n"
            f"Ks 0.250000 0.250000 0.250000\n"
            f"Ns 64.000000\n"
            f"d 1.000000\n"
            f"illum 2\n"
        )
        mtl_path.write_text(mtl_content, encoding="utf-8")

        verts = mesh.vertices
        normals = mesh.vertex_normals
        faces = mesh.faces

        min_xy = verts[:, :2].min(axis=0)
        span_xy = np.maximum(verts[:, :2].max(axis=0) - min_xy, 1e-8)
        uvs = (verts[:, :2] - min_xy) / span_xy

        lines: List[str] = [
            f"# Wavefront OBJ exported by Img2Mesh3D ({self.cfg.calibration.unit})",
            f"mtllib {mtl_filename}",
            f"o {base_name}",
        ]
        for v in verts:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for uv in uvs:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
        for vn in normals:
            lines.append(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}")

        lines.append("usemtl material_calibrated")
        lines.append("s 1")
        for f in faces:
            i0, i1, i2 = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
            lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")

        obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    """
    Menyediakan inspeksi visual 3D interaktif menggunakan Open3D Visualizer
    dengan overlay sumbu koordinat metrik, bounding box, dan grid bidang dasar Z=0.
    """

    @staticmethod
    def _has_display_server() -> bool:
        if sys.platform.startswith("linux"):
            return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        return True

    @staticmethod
    def _build_ground_grid(
        min_bound: np.ndarray, max_bound: np.ndarray
    ) -> o3d.geometry.LineSet:
        """Membangun LineSet grid referensi pada bidang Z = 0."""
        span_x = max(float(max_bound[0] - min_bound[0]), 1.0)
        span_y = max(float(max_bound[1] - min_bound[1]), 1.0)
        max_span = max(span_x, span_y)

        # Pilih interval grid kelipatan 1, 2, atau 5 berbasis pangkat 10
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
        """
        Membuka jendela pratinjau Open3D interaktif.
        Mengembalikan True apabila ekspor disetujui/dilanjutkan, atau False jika dibatalkan.
        """
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

        # Siapkan geometri diagnostik
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

            # Atur sudut pandang isometrik awal menghadap +Z dari kuadran depan-kanan
            ctr = vis.get_view_control()
            ctr.set_front([0.55, -0.65, 0.52])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_lookat(aabb.get_center())
            ctr.set_zoom(0.75)

            def toggle_wireframe(v: o3d.visualization.Visualizer) -> bool:
                state["wireframe"] = not state["wireframe"]
                v.get_render_option().mesh_show_wireframe = state["wireframe"]
                logger.info("Preview Wireframe: %s", "ON" if state["wireframe"] else "OFF")
                return False

            def toggle_backface(v: o3d.visualization.Visualizer) -> bool:
                state["back_face"] = not state["back_face"]
                v.get_render_option().mesh_show_back_face = state["back_face"]
                logger.info("Preview Back-Face: %s", "ON" if state["back_face"] else "OFF")
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
                logger.info("Preview Diagnostic Guides: %s", "ON" if state["guides"] else "OFF")
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
        denoised = self.preprocessor.enhance_and_denoise(img_bgr)

        if self.cfg.mode == "cad":
            binary = self.preprocessor.binarize_high_precision(denoised)
            return self.cad_reconstructor.reconstruct(
                binary_mask=binary,
                denoised_gray=denoised,
                img_rgb=img_rgb,
                z_offset=z_offset,
            )
        else:
            pcd, z_grid, rgb_grid = self.surface_reconstructor.build_calibrated_point_cloud(
                img_rgb=img_rgb,
                denoised_gray=denoised,
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

        # Audit geometri & topologi sebelum/sesudah ekspor
        report = MeshValidator.validate(
            model_name=base_name,
            mesh=clean_tm,
            o3d_mesh=clean_o3d,
            unit=self.cfg.calibration.unit,
        )

        # Jika mode gerbang konfirmasi aktif, tampilkan pratinjau Open3D sebelum menulis file
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

        # Pratinjau interaktif standar setelah validasi & ekspor selesai
        if self.cfg.preview.enabled and not self.cfg.preview.confirm_before_export:
            InteractiveMeshPreviewer.show(
                o3d_mesh=clean_o3d,
                report=report,
                preview_cfg=self.cfg.preview,
                output_dir=self.cfg.output_dir,
            )

        return report


# ==============================================================================
# PARSING & ANTARMUKA CLI (COMMAND-LINE INTERFACE)
# ==============================================================================


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Konversi Presisi Tinggi Gambar 2D ke Model 3D dengan Kalibrasi Intrinsik, Homografi, & Open3D Preview",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Sub-Mode Utilitas: Kalibrasi Kamera Papan Catur ---
    calib_group = parser.add_argument_group("Kalibrasi Matriks Intrinsik Kamera (Tahap 0A)")
    calib_group.add_argument(
        "--calibrate-chessboard",
        nargs="+",
        help="Folder/daftar foto papan catur untuk menghitung matriks intrinsik kamera K & distorsi",
    )
    calib_group.add_argument(
        "--chessboard-size",
        type=str,
        default="9x6",
        help="Jumlah sudut dalam papan catur (KolomxBaris, mis. 9x6)",
    )
    calib_group.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="Ukuran sisi satu kotak papan catur dalam satuan nyata (mis. 25.0 mm)",
    )
    calib_group.add_argument(
        "--save-calib",
        type=Path,
        default=Path("./camera_intrinsics.json"),
        help="Path file JSON keluaran hasil kalibrasi kamera",
    )
    calib_group.add_argument(
        "--camera-calib",
        type=Path,
        default=None,
        help="Path file JSON kalibrasi intrinsik kamera untuk menghilangkan distorsi lensa",
    )
    calib_group.add_argument(
        "--intrinsics-manual",
        type=str,
        default=None,
        help="Parameter intrinsik manual 'fx,fy,cx,cy,k1,k2,p1,p2[,k3]' jika tidak memakai JSON",
    )

    # --- Sub-Mode Koreksi Perspektif Homografi 4-Titik ---
    homo_group = parser.add_argument_group("Koreksi Perspektif Homografi 4-Titik (Tahap 0B)")
    homo_group.add_argument(
        "--homography-points",
        type=str,
        default=None,
        help="4 titik sudut referensi piksel 'x1,y1;x2,y2;x3,y3;x4,y4' berurutan [TL, TR, BR, BL]",
    )
    homo_group.add_argument(
        "--pick-points",
        action="store_true",
        help="Pilih 4 titik sudut referensi secara interaktif dengan klik mouse pada gambar",
    )
    homo_group.add_argument(
        "--homography-size",
        type=str,
        default=None,
        help="Dimensi fisik nyata bidang 4 titik 'LebarxTinggi' dalam satuan --unit (mis. 210x297 untuk A4 mm)",
    )
    homo_group.add_argument(
        "--homography-crop",
        action="store_true",
        help="Potong output hanya di dalam area 4 titik (default: pertahankan seluruh kanvas)",
    )

    # --- Sub-Mode Pratinjau 3D Interaktif (Tahap 5) ---
    prev_group = parser.add_argument_group("Pratinjau Mesh 3D Interaktif Open3D (Tahap 5)")
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
        help="Tampilkan pratinjau terlebih dahulu dan minta konfirmasi terminal [Y/n] sebelum menyimpan file",
    )

    # --- Input & Rekonstruksi Utama ---
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        help="Path 1 file gambar, beberapa file gambar, folder, atau wildcard (mis. *.png)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./output_3d"),
        help="Direktori penyimpanan hasil ekspor 3D",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["cad", "surface"],
        default="cad",
        help="Mode rekonstruksi: 'cad' (Mode A: kontur & ekstrusi) atau 'surface' (Mode B: depth/relief)",
    )
    parser.add_argument(
        "--multi-strategy",
        choices=["batch", "stack", "merge"],
        default="batch",
        help="Strategi multi-file: 'batch' (terpisah), 'stack' (tumpuk layer Z di Mode A), 'merge' (ICP Mode B)",
    )
    parser.add_argument(
        "--ref-real-size",
        type=float,
        default=None,
        help="Ukuran fisik referensi nyata (wajib jika tidak menggunakan --homography-size)",
    )
    parser.add_argument(
        "--ref-pixels",
        type=float,
        default=None,
        help="Panjang objek referensi dalam piksel (wajib jika tidak menggunakan --homography-size)",
    )
    parser.add_argument(
        "--unit",
        choices=["mm", "cm", "m"],
        default="mm",
        help="Satuan fisik model 3D",
    )
    parser.add_argument(
        "--extrude-height",
        type=float,
        default=5.0,
        help="Tinggi ekstrusi Z (Mode A) atau amplitudo relief maksimum (Mode B) dalam satuan --unit",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Balik polaritas objek (objek terang di atas latar gelap)",
    )
    parser.add_argument("--rdp-epsilon", type=float, default=0.35, help="Toleransi Ramer-Douglas-Peucker (px)")
    parser.add_argument("--min-area", type=float, default=25.0, help="Luas kontur minimum (px^2)")
    parser.add_argument("--layer-gap", type=float, default=0.0, help="Celah antar layer pada mode stack")
    parser.add_argument(
        "--depth-backend",
        choices=["heightmap", "depth-anything", "midas"],
        default="heightmap",
        help="Backend estimasi kedalaman untuk Mode B",
    )
    parser.add_argument(
        "--surface-method",
        choices=["poisson", "alpha", "grid-solid"],
        default="poisson",
        help="Algoritma rekonstruksi permukaan untuk Mode B",
    )
    parser.add_argument("--poisson-depth", type=int, default=9, help="Kedalaman octree Poisson (8-11)")
    parser.add_argument("--no-solid-base", action="store_true", help="Jangan tutup alas bawah Z=0 pada Mode B")
    parser.add_argument("--taubin-iter", type=int, default=5, help="Iterasi Taubin smoothing pada Mode B")
    parser.add_argument("--decimate", type=float, default=0.0, help="Rasio reduksi segitiga (0.0 - 0.90)")

    args = parser.parse_args()

    try:
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
            raise CalibrationError("Argumen --input wajib diisi untuk menjalankan rekonstruksi 3D.")

        input_files = resolve_input_files(args.input)

        intrinsics_cfg: Optional[CameraIntrinsicsConfig] = None
        if args.camera_calib is not None:
            intrinsics_cfg = CameraIntrinsicsConfig.from_json(args.camera_calib)
        elif args.intrinsics_manual is not None:
            vals = [float(v.strip()) for v in args.intrinsics_manual.split(",") if v.strip()]
            if len(vals) < 8:
                raise CalibrationError(
                    "--intrinsics-manual membutuhkan minimal 8 nilai: fx,fy,cx,cy,k1,k2,p1,p2[,k3]."
                )
            intrinsics_cfg = CameraIntrinsicsConfig.from_manual_params(
                fx=vals[0], fy=vals[1], cx=vals[2], cy=vals[3], dist_coeffs=vals[4:]
            )

        homography_cfg: Optional[HomographyConfig] = None
        if args.homography_points is not None or args.pick_points:
            if args.homography_size is None:
                raise CalibrationError(
                    "--homography-size (mis. 210x297) wajib diisi saat menggunakan homografi 4-titik."
                )
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

        if homography_cfg is not None:
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
                    "Isi --ref-real-size dan --ref-pixels, ATAU aktifkan homografi 4-titik "
                    "(--homography-points / --pick-points + --homography-size)."
                )
            calib_cfg = CalibrationConfig(
                ref_real_size=args.ref_real_size,
                ref_pixels=args.ref_pixels,
                unit=args.unit,
                extrude_height=args.extrude_height,
                invert_polarity=args.invert,
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
            homography=homography_cfg,
            interactive_pick_points=args.pick_points,
            rdp_epsilon_px=args.rdp_epsilon,
            min_contour_area_px=args.min_area,
            layer_gap=args.layer_gap,
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
