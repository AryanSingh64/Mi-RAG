import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image


class VisualFingerprintMatcher:
    """
    High-speed, multi-metric perceptual image matcher.
    Uses gradient difference hash (dHash), average hash (aHash), normalized
    cross-correlation (NCC), and color distribution correlation.
    Enables instant reverse-image matching against document figures on CPU in milliseconds
    without requiring heavy deep-learning model inference.
    """

    @staticmethod
    def compute_fingerprint(image_input: Union[str, Path, Image.Image]) -> Optional[Dict[str, np.ndarray]]:
        """
        Extracts structural, intensity, and color fingerprints from an image.
        """
        should_close = False
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                return None
            try:
                img = Image.open(p)
                should_close = True
            except Exception:
                return None
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            return None

        try:
            rgb = img.convert("RGB")
            
            # 1. dHash (64-bit gradient structure: 9x8 grayscale)
            gray_d = rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            arr_gray_d = np.array(gray_d, dtype=np.int16)
            dhash_bits = (arr_gray_d[:, 1:] > arr_gray_d[:, :-1]).flatten()

            # 2. aHash (64-bit average intensity: 8x8 grayscale)
            gray_a = rgb.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
            arr_gray_a = np.array(gray_a, dtype=np.float32)
            ahash_bits = (arr_gray_a > arr_gray_a.mean()).flatten()

            # 3. Normalized cross-correlation thumbnail (32x32 RGB)
            thumb = rgb.resize((32, 32), Image.Resampling.BILINEAR)
            thumb_arr = np.array(thumb, dtype=np.float32)
            std = thumb_arr.std()
            norm_thumb = (thumb_arr - thumb_arr.mean()) / (std if std > 1e-4 else 1.0)

            # 4. Color channel histogram (16 bins per channel)
            h_r, _ = np.histogram(thumb_arr[:, :, 0], bins=16, range=(0, 256), density=True)
            h_g, _ = np.histogram(thumb_arr[:, :, 1], bins=16, range=(0, 256), density=True)
            h_b, _ = np.histogram(thumb_arr[:, :, 2], bins=16, range=(0, 256), density=True)
            color_hist = np.concatenate([h_r, h_g, h_b])

            return {
                "dhash": dhash_bits,
                "ahash": ahash_bits,
                "norm_thumb": norm_thumb,
                "color_hist": color_hist
            }
        except Exception:
            return None
        finally:
            if should_close:
                try:
                    img.close()
                except Exception:
                    pass

    @classmethod
    def compare_fingerprints(cls, fp1: Dict[str, np.ndarray], fp2: Dict[str, np.ndarray]) -> float:
        """
        Calculates similarity between two fingerprints on a scale of 0.0 to 1.0.
        """
        if not fp1 or not fp2:
            return 0.0

        # 1. dHash similarity (gradient structure)
        diff_d = np.count_nonzero(fp1["dhash"] != fp2["dhash"])
        sim_d = 1.0 - (diff_d / len(fp1["dhash"]))

        # 2. aHash similarity (luminance distribution)
        diff_a = np.count_nonzero(fp1["ahash"] != fp2["ahash"])
        sim_a = 1.0 - (diff_a / len(fp1["ahash"]))

        # 3. Normalized Cross Correlation (spatial pixel layout)
        ncc = np.mean(fp1["norm_thumb"] * fp2["norm_thumb"])
        sim_ncc = max(0.0, min(1.0, float(ncc)))

        # 4. Color histogram correlation
        corr_matrix = np.corrcoef(fp1["color_hist"], fp2["color_hist"])
        col_corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        sim_col = max(0.0, min(1.0, (float(col_corr) + 1.0) / 2.0))

        # Weighted aggregate similarity
        combined = (0.35 * sim_d) + (0.35 * sim_ncc) + (0.15 * sim_a) + (0.15 * sim_col)
        return float(np.clip(combined, 0.0, 1.0))

    @classmethod
    def match_against_directory(
        cls,
        query_image: Union[str, Path, Image.Image],
        target_dir: Union[str, Path],
        min_threshold: float = 0.50,
        top_k: int = 3
    ) -> List[Tuple[Path, float]]:
        """
        Finds the closest matching document figures in target_dir for a given query image.
        Returns a sorted list of (image_path, similarity_score).
        """
        target_path = Path(target_dir)
        if not target_path.exists() or not target_path.is_dir():
            return []

        q_fp = cls.compute_fingerprint(query_image)
        if not q_fp:
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        candidates = [
            f for f in target_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        scored: List[Tuple[Path, float]] = []
        for cand in candidates:
            c_fp = cls.compute_fingerprint(cand)
            if c_fp:
                sim = cls.compare_fingerprints(q_fp, c_fp)
                if sim >= min_threshold:
                    scored.append((cand, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
