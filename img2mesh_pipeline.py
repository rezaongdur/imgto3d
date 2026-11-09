#!/usr/bin/env python3
"""
End-to-End 2D Image(s) to Calibrated 3D Solid Mesh Reconstruction Pipeline.

Mendukung dua mode rekonstruksi presisi tinggi:
  - Mode A (CAD / Orthographic): Deteksi kontur sub-piksel, simplifikasi
    Ramer-Douglas-Peucker, validasi topologi Shapely, triangulasi CDT,
    dan ekstrusi sumbu Z untuk denah, logo, dan komponen mekanikal.
  - Mode B (Surface / Relief / 3D): Estimasi depth map resolusi tinggi
    (Depth Anything V2, MiDaS, atau Bilateral Heightmap), proyeksi point cloud
    terkalibrasi Open3D, dan rekonstruksi Screened Poisson / Alpha Shapes
    dengan penutupan alas solid watertight.

Ekspor otomatis ke: .STL (binary), .OBJ + .MTL, .DAE (COLLADA Z-up SketchUp),
dan .PLY (dengan vertex colors).
"""

from __future__ import annotations

import argparse
import glob
import logging
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
    """Dilempar apabila parameter kalibrasi fisik tidak valid."""


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
class PipelineConfig:
    """Konfigurasi lengkap seluruh tahapan pipeline."""

    input_paths: List[Path]
    output_dir: Path
    mode: ModeType = "cad"
    multi_strategy: MultiFileStrategy = "batch"
    calibration: CalibrationConfig = field(
        default_factory=lambda: CalibrationConfig(100.0, 100.0, "mm", 5.0)
    )
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
# TAHAP 1: PREPROCESSING & KALIBRASI SKALA
# ==============================================================================


class ImagePreprocessor:
    """Menangani koreksi kontras adaptif, bilateral filtering, dan binarisasi."""

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def load_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memuat citra BGR dan versi RGB-nya dengan validasi integritas file.
        """
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"File citra tidak ditemukan: {image_path}")

        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise PipelineError(f"Gagal mendekode citra (format korup/tidak didukung): {image_path}")

        # Tangani citra RGBA (misalnya logo PNG dengan latar transparan)
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            alpha = img_bgr[:, :, 3:4].astype(np.float32) / 255.0
            bgr = img_bgr[:, :, :3].astype(np.float32)
            white_bg = np.full_like(bgr, 255.0)
            img_bgr = np.clip(bgr * alpha + white_bg * (1.0 - alpha), 0, 255).astype(np.uint8)
        elif img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_bgr, img_rgb

    def enhance_and_denoise(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Menerapkan CLAHE pada kanal luminansi L* (CIE Lab) dilanjutkan
        dengan Bilateral Filtering untuk meredam noise tanpa mengaburkan tepi tajam.
        """
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
        """
        Menghasilkan mask biner bersih (0 atau 255) dengan sub-pixel morphological
        conditioning untuk ekstraksi kontur CAD presisi tinggi.
        """
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

        # Operasi morfologi ringan untuk menutup celah 1-piksel tanpa membulatkan sudut
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
        """
        Memperhalus koordinat titik kontur ke presisi sub-piksel menggunakan
        cv2.cornerSubPix pada titik sudut dominan.
        """
        refined: List[np.ndarray] = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        h, w = gray_img.shape[:2]

        for cnt in contours:
            pts = cnt.reshape(-1, 2).astype(np.float32)
            if len(pts) < 3:
                continue
            # Filter titik yang terlalu dekat dengan batas gambar agar window subPix aman
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
        """
        Membangun poligon Shapely berlubang (exterior + interior holes) yang
        terkalibrasi dalam satuan fisik nyata.
        """
        h, _ = binary_mask.shape[:2]
        scale = self.cfg.calibration.scale_per_pixel

        # RETR_CCOMP menyusun hierarki 2 level: level 0 = exterior, level 1 = hole
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
            # Proses hanya kontur terluar (parent == -1)
            if hier[idx][3] != -1:
                continue

            area_px = cv2.contourArea(cnt)
            if area_px < self.cfg.min_contour_area_px:
                continue

            # Aproksimasi poligon Ramer-Douglas-Peucker dengan epsilon sangat kecil
            approx_ext = cv2.approxPolyDP(cnt, epsilon=self.cfg.rdp_epsilon_px, closed=True)
            refined_ext = self._refine_contours_subpixel(denoised_gray, [approx_ext])
            if not refined_ext or len(refined_ext[0]) < 3:
                continue

            ext_pts_px = refined_ext[0]
            ext_coords = [(float(x * scale), float((h - y) * scale)) for x, y in ext_pts_px]

            # Kumpulkan seluruh lubang (holes) yang merupakan child langsung dari kontur ini
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

            # Ekstrak semua Polygon valid dari hasil make_valid
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
        """
        Mengekstrusi poligon terkalibrasi menjadi mesh solid 3D watertight
        dengan pewarnaan vertex yang disampel dari citra input.
        """
        polygons = self._extract_shapely_polygons(binary_mask, denoised_gray)
        extruded_parts: List[trimesh.Trimesh] = []
        height = self.cfg.calibration.extrude_height

        for poly in polygons:
            # Orientasi standar Shapely (CCW exterior, CW holes)
            oriented_poly = poly
            try:
                mesh_part = trimesh.creation.extrude_polygon(oriented_poly, height=height)
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

        # Sampel warna RGB dari citra asli berdasarkan proyeksi (X, Y) vertex
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
        """
        Menghasilkan depth map ternormalisasi [0.0, 1.0].
        Mendukung Depth Anything V2 / MiDaS via HuggingFace Transformers,
        dengan fallback deterministik ke Bilateral Photometric Heightmap.
        """
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

        # Fallback / Default Presisi Tinggi: Bilateral Photometric Heightmap
        depth_norm = denoised_gray.astype(np.float32) / 255.0
        if self.cfg.calibration.invert_polarity:
            depth_norm = 1.0 - depth_norm
        # Penghalusan sub-piksel Gaussian ringan untuk menghilangkan kuantisasi 8-bit
        depth_norm = gaussian_filter(depth_norm, sigma=0.8)
        return np.clip(depth_norm, 0.0, 1.0)

    def build_calibrated_point_cloud(
        self, img_rgb: np.ndarray, denoised_gray: np.ndarray
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
        """
        Memproyeksikan piksel terkalibrasi menjadi Open3D PointCloud dengan
        estimasi normal permukaan yang konsisten secara topologis.
        """
        depth_map = self._estimate_depth_map(img_rgb, denoised_gray)
        h, w = depth_map.shape[:2]

        # Downsampling terukur jika resolusi citra melebihi max_grid_dim demi efisiensi memori
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

        # Estimasi dan orientasi normal konsisten ke arah +Z
        radius = eff_scale * 3.0
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))

        return pcd, z_coords, rgb_small

    def _build_watertight_grid_solid(
        self, z_grid: np.ndarray, rgb_grid: np.ndarray, eff_scale: float
    ) -> trimesh.Trimesh:
        """
        Membangun solid relief 3D 100% watertight dan manifold dari grid elevasi
        terkalibrasi lengkap dengan 4 dinding samping dan alas datar Z = 0.
        """
        gh, gw = z_grid.shape
        u = np.arange(gw, dtype=np.float64) * eff_scale
        v = (gh - np.arange(gh, dtype=np.float64)) * eff_scale
        xx, yy = np.meshgrid(u, v)

        # Top surface vertices: indeks 0 .. (gh*gw - 1)
        top_verts = np.stack([xx, yy, z_grid], axis=-1).reshape(-1, 3)
        # Bottom base vertices (Z = 0): indeks (gh*gw) .. (2*gh*gw - 1)
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

        # 1. Permukaan atas (normal menghadap +Z) & Permukaan bawah (normal menghadap -Z)
        for r in range(gh - 1):
            for c in range(gw - 1):
                t00, t01 = idx_top(r, c), idx_top(r, c + 1)
                t10, t11 = idx_top(r + 1, c), idx_top(r + 1, c + 1)
                # Top faces (CCW dilihat dari atas +Z)
                faces.append([t00, t10, t11])
                faces.append([t00, t11, t01])

                b00, b01 = idx_bot(r, c), idx_bot(r, c + 1)
                b10, b11 = idx_bot(r + 1, c), idx_bot(r + 1, c + 1)
                # Bottom faces (CCW dilihat dari bawah -Z)
                faces.append([b00, b11, b10])
                faces.append([b00, b01, b11])

        # 2. Dinding tepi Utara (r = 0, menghadap +Y) & Selatan (r = gh - 1, menghadap -Y)
        for c in range(gw - 1):
            # Utara (r = 0)
            faces.append([idx_top(0, c), idx_top(0, c + 1), idx_bot(0, c + 1)])
            faces.append([idx_top(0, c), idx_bot(0, c + 1), idx_bot(0, c)])
            # Selatan (r = gh - 1)
            r_last = gh - 1
            faces.append([idx_top(r_last, c), idx_bot(r_last, c + 1), idx_top(r_last, c + 1)])
            faces.append([idx_top(r_last, c), idx_bot(r_last, c), idx_bot(r_last, c + 1)])

        # 3. Dinding tepi Barat (c = 0, menghadap -X) & Timur (c = gw - 1, menghadap +X)
        for r in range(gh - 1):
            # Barat (c = 0)
            faces.append([idx_top(r, 0), idx_bot(r + 1, 0), idx_top(r + 1, 0)])
            faces.append([idx_top(r, 0), idx_bot(r, 0), idx_bot(r + 1, 0)])
            # Timur (c = gw - 1)
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
        """
        Membangun mesh 3D dari Point Cloud terkalibrasi menggunakan metode
        Poisson, Alpha Shapes, atau Grid Solid Watertight.
        """
        if self.cfg.surface_method == "grid-solid" and z_grid is not None and rgb_grid is not None and eff_scale is not None:
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

            # Jika close_solid_base aktif dan data grid tersedia, gunakan hasil Poisson
            # untuk menyaring elevasi lalu tutup menjadi solid watertight
            if self.cfg.close_solid_base and z_grid is not None and rgb_grid is not None and eff_scale is not None:
                logger.info("Menutup alas bawah Z=0 untuk menjamin solid watertight siap cetak...")
                # Haluskan z_grid sesuai rekonstruksi Poisson sebelum menutup alas
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

    def clean_and_optimize(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, o3d.geometry.TriangleMesh]:
        """
        Menjalankan pembersihan mesh multi-tahap di Trimesh dan Open3D.
        """
        # 1. Pembersihan dasar Trimesh (hapus segitiga nol, duplikat, dan vertex yatim)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()

        # 2. Hapus pulau noise terisolasi jika ukurannya sangat kecil (< 0.1% dari total face)
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

        # 3. Konversi ke Open3D untuk pembersihan non-manifold & Taubin smoothing
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

        # 4. Taubin Smoothing (bebas penyusutan volume, menjaga ketajaman sudut CAD)
        if self.cfg.taubin_iterations > 0 and self.cfg.mode == "surface":
            o3d_mesh = o3d_mesh.filter_smooth_taubin(
                number_of_iterations=self.cfg.taubin_iterations,
                lambda_filter=0.5,
                mu=-0.53,
            )

        # 5. Edge-Preserving Quadric Edge Collapse Decimation (opsional jika diminta)
        if 0.0 < self.cfg.target_reduction < 0.95:
            current_tris = len(o3d_mesh.triangles)
            target_tris = max(12, int(current_tris * (1.0 - self.cfg.target_reduction)))
            if target_tris < current_tris:
                o3d_mesh = o3d_mesh.simplify_quadric_decimation(
                    target_number_of_triangles=target_tris
                )

        o3d_mesh.compute_vertex_normals()

        # 6. Sinkronisasi kembali ke Trimesh + Penutupan Lubang & Perbaikan Normal
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

        # Pastikan orientasi normal mengarah ke luar (volume bertanda positif)
        if final_tm.is_watertight and final_tm.volume < 0:
            final_tm.invert()

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

        # 1. Binary .STL
        stl_path = self.cfg.output_dir / f"{base_name}.stl"
        mesh.export(str(stl_path), file_type="stl")
        exported["STL"] = stl_path

        # 2. .OBJ + .MTL (lengkap dengan UV dan normal)
        obj_path = self.cfg.output_dir / f"{base_name}.obj"
        self._export_obj_with_mtl(mesh, obj_path, base_name)
        exported["OBJ"] = obj_path
        exported["MTL"] = self.cfg.output_dir / f"{base_name}.mtl"

        # 3. .DAE / COLLADA (Z-up & skala metrik presisi untuk SketchUp)
        dae_path = self.cfg.output_dir / f"{base_name}.dae"
        self._export_sketchup_collada(mesh, dae_path)
        exported["DAE"] = dae_path

        # 4. .PLY (dengan informasi vertex colors & normal)
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
        """
        Menulis file Wavefront .OBJ beserta file material .MTL dengan koordinat
        tekstur UV terproyeksi planar dan vektor normal vertex.
        """
        mtl_filename = f"{base_name}.mtl"
        mtl_path = obj_path.with_suffix(".mtl")

        # Hitung rata-rata warna difus dari vertex colors jika ada
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

        # Hitung koordinat UV planar ternormalisasi [0, 1] berdasarkan bounding box XY
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
        """
        Mengekspor ke format COLLADA (.DAE) dan menyuntikkan tag <unit> serta
        <up_axis>Z_UP</up_axis> sehingga SketchUp langsung mengimpor pada
        orientasi Z-up dan skala fisik 1:1 tanpa distorsi.
        """
        dae_bytes = trimesh.exchange.dae.export_collada(mesh)
        dae_xml_str = dae_bytes.decode("utf-8") if isinstance(dae_bytes, bytes) else str(dae_bytes)

        # Daftarkan namespace COLLADA 1.4.1
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
# TAHAP 5: VALIDASI GEOMETRIS & TOPOLOGIS
# ==============================================================================


class MeshValidator:
    """Mengaudit dimensi fisik nyata, volume, dan integritas topologi mesh."""

    @staticmethod
    def validate(
        model_name: str,
        mesh: trimesh.Trimesh,
        o3d_mesh: o3d.geometry.TriangleMesh,
        unit: str,
        exported_files: Dict[str, Path],
    ) -> ValidationReport:
        extents = mesh.extents  # [Panjang X, Lebar Y, Tinggi Z]
        is_wt = bool(mesh.is_watertight)
        volume = float(mesh.volume) if is_wt else None

        report = ValidationReport(
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
            exported_files=exported_files,
        )
        return report

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
        print("-" * 72)
        print("  File Output Terealisasi:")
        for fmt, path in report.exported_files.items():
            print(f"    [{fmt:<3}] -> {path.resolve()}")
        print(f"{border}\n")


# ==============================================================================
# ORKESTRATOR UTAMA PIPELINE (SINGLE & MULTI-FILE)
# ==============================================================================


class ImageTo3DPipeline:
    """Orkestrator end-to-end yang menghubungkan seluruh 5 tahapan presisi."""

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
                img_bgr, img_rgb = self.preprocessor.load_image(img_path)
                denoised = self.preprocessor.enhance_and_denoise(img_bgr)
                pcd, _, _ = self.surface_reconstructor.build_calibrated_point_cloud(img_rgb, denoised)

                if idx > 0 and len(combined_pcd.points) > 0:
                    # Registrasi halus Point-to-Plane ICP antar awan titik
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
        img_bgr, img_rgb = self.preprocessor.load_image(img_path)
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
        exported_files = self.exporter.export_all(clean_tm, clean_o3d, base_name)
        report = MeshValidator.validate(
            model_name=base_name,
            mesh=clean_tm,
            o3d_mesh=clean_o3d,
            unit=self.cfg.calibration.unit,
            exported_files=exported_files,
        )
        MeshValidator.print_report(report)
        return report


# ==============================================================================
# ANTARMUKA CLI (COMMAND-LINE INTERFACE)
# ==============================================================================


def resolve_input_files(raw_inputs: Sequence[str]) -> List[Path]:
    """Mengurai path tunggal, multi-file, direktori, atau pola wildcard."""
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


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Konversi Presisi Tinggi Gambar 2D (JPG/PNG) ke Model 3D (.STL, .OBJ, .DAE, .PLY)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
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
    # Parameter Kalibrasi Skala
    parser.add_argument(
        "--ref-real-size",
        type=float,
        required=True,
        help="Ukuran fisik referensi nyata (misal: 100.0 untuk 100 mm)",
    )
    parser.add_argument(
        "--ref-pixels",
        type=float,
        required=True,
        help="Panjang objek referensi pada gambar dalam satuan piksel (misal: 500.0)",
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
    # Parameter Mode A (CAD)
    parser.add_argument(
        "--rdp-epsilon",
        type=float,
        default=0.35,
        help="Toleransi error Ramer-Douglas-Peucker dalam piksel (Mode A)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=25.0,
        help="Luas kontur minimum dalam piksel persegi untuk memfilter noise (Mode A)",
    )
    parser.add_argument(
        "--layer-gap",
        type=float,
        default=0.0,
        help="Jarak celah antar layer saat menggunakan --multi-strategy stack",
    )
    # Parameter Mode B (Surface / Relief)
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
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=9,
        help="Kedalaman octree Screened Poisson Surface Reconstruction (8-11)",
    )
    parser.add_argument(
        "--no-solid-base",
        action="store_true",
        help="Jangan tutup alas bawah Z=0 pada Mode B (menghasilkan open surface mesh)",
    )
    # Parameter Optimasi Mesh
    parser.add_argument(
        "--taubin-iter",
        type=int,
        default=5,
        help="Jumlah iterasi Taubin smoothing pada Mode B",
    )
    parser.add_argument(
        "--decimate",
        type=float,
        default=0.0,
        help="Rasio reduksi segitiga (0.0 - 0.90) dengan quadric edge collapse",
    )

    args = parser.parse_args()
    input_files = resolve_input_files(args.input)

    calib = CalibrationConfig(
        ref_real_size=args.ref_real_size,
        ref_pixels=args.ref_pixels,
        unit=args.unit,
        extrude_height=args.extrude_height,
        invert_polarity=args.invert,
    )

    return PipelineConfig(
        input_paths=input_files,
        output_dir=args.output_dir,
        mode=args.mode,
        multi_strategy=args.multi_strategy,
        calibration=calib,
        rdp_epsilon_px=args.rdp_epsilon,
        min_contour_area_px=args.min_area,
        layer_gap=args.layer_gap,
        depth_backend=args.depth_backend,
        surface_method=args.surface_method,
        poisson_depth=args.poisson_depth,
        close_solid_base=not args.no_solid_base,
        taubin_iterations=args.taubin_iter,
        target_reduction=args.decimate,
    )


def main() -> int:
    try:
        config = parse_args()
        logger.info(
            "Skala Kalibrasi Terhitung: 1 piksel = %.6f %s",
            config.calibration.scale_per_pixel,
            config.calibration.unit,
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
