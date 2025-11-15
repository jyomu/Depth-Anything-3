#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VRAM効率的なシーン復元の実装例

このスクリプトは、12GB未満のVRAMで大規模シーンを処理する方法を示します。
主な機能:
- フレームチャンク処理
- VRAMモニタリング
- 中間結果の保存
- 自動的なメモリクリーンアップ
"""

import gc
import glob
import os
import sys
import time
from typing import List

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.specs import Prediction
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install the package: pip install -e .")
    sys.exit(1)


class VRAMEfficientReconstructor:
    """
    VRAM効率的なシーン復元クラス
    
    大規模シーンを小さなチャンクに分割して処理し、
    12GB未満のVRAMでも処理可能にします。
    """
    
    def __init__(
        self,
        model_name: str = "depth-anything/DA3-LARGE",
        device: str = "cuda",
        chunk_size: int = 8,
        process_res: int = 504,
        max_vram_gb: float = 11.0,
        verbose: bool = True,
    ):
        """
        Args:
            model_name: 使用するモデル名
            device: デバイス ('cuda' or 'cpu')
            chunk_size: 一度に処理するフレーム数
            process_res: 処理解像度
            max_vram_gb: VRAMの目標上限（GB）
            verbose: 詳細ログの出力
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.process_res = process_res
        self.max_vram_gb = max_vram_gb
        self.verbose = verbose
        
        self.model = None
        self.load_time = 0
        
    def load_model(self):
        """モデルをロード"""
        if self.model is not None:
            return
            
        if self.verbose:
            print(f"Loading model: {self.model_name}")
        
        start_time = time.time()
        self.model = DepthAnything3.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.load_time = time.time() - start_time
        
        if self.verbose:
            print(f"Model loaded in {self.load_time:.2f}s")
            if torch.cuda.is_available():
                self._print_vram_usage("After model loading")
    
    def _print_vram_usage(self, stage: str = ""):
        """VRAM使用量を表示"""
        if not torch.cuda.is_available():
            return
            
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
        
        status = f"VRAM {stage}: "
        status += f"Allocated={allocated:.2f}GB, "
        status += f"Reserved={reserved:.2f}GB, "
        status += f"Peak={max_allocated:.2f}GB"
        
        if reserved > self.max_vram_gb:
            status += f" ⚠️ EXCEEDS LIMIT ({self.max_vram_gb}GB)"
            
        print(status)
    
    def _cleanup_memory(self):
        """メモリをクリーンアップ"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def process_scene(
        self,
        image_paths: List[str],
        output_dir: str,
        export_format: str = "mini_npz",
        num_max_points: int = 500_000,
        conf_thresh_percentile: float = 50.0,
    ) -> List[Prediction]:
        """
        シーン全体をチャンク処理
        
        Args:
            image_paths: 入力画像パスのリスト
            output_dir: 出力ディレクトリ
            export_format: エクスポート形式
            num_max_points: 最大ポイント数
            conf_thresh_percentile: 信頼度閾値パーセンタイル
            
        Returns:
            全チャンクのPredictionリスト
        """
        self.load_model()
        os.makedirs(output_dir, exist_ok=True)
        
        num_chunks = (len(image_paths) + self.chunk_size - 1) // self.chunk_size
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing Scene")
            print(f"{'='*60}")
            print(f"Total images: {len(image_paths)}")
            print(f"Chunk size: {self.chunk_size}")
            print(f"Number of chunks: {num_chunks}")
            print(f"Process resolution: {self.process_res}")
            print(f"Target VRAM limit: {self.max_vram_gb}GB")
            print(f"{'='*60}\n")
        
        all_predictions = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(image_paths))
            chunk_images = image_paths[start_idx:end_idx]
            
            if self.verbose:
                print(f"\nChunk {chunk_idx + 1}/{num_chunks}")
                print(f"  Frames: {start_idx} to {end_idx-1} ({len(chunk_images)} images)")
            
            # メモリをクリーンアップ
            self._cleanup_memory()
            
            # VRAMをリセット
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            
            # チャンクの処理
            try:
                chunk_start_time = time.time()
                
                chunk_output_dir = os.path.join(output_dir, f"chunk_{chunk_idx:04d}")
                prediction = self.model.inference(
                    chunk_images,
                    process_res=self.process_res,
                    export_dir=chunk_output_dir,
                    export_format=export_format,
                    num_max_points=num_max_points,
                    conf_thresh_percentile=conf_thresh_percentile,
                )
                
                chunk_time = time.time() - chunk_start_time
                
                all_predictions.append(prediction)
                
                if self.verbose:
                    print(f"  Processing time: {chunk_time:.2f}s")
                    print(f"  Output: {chunk_output_dir}")
                    self._print_vram_usage(f"Chunk {chunk_idx + 1}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ OOM Error in chunk {chunk_idx}")
                    print(f"  Current chunk size: {self.chunk_size}")
                    print(f"  Suggested actions:")
                    print(f"    1. Reduce chunk_size (try {self.chunk_size // 2})")
                    print(f"    2. Reduce process_res (try {self.process_res - 56})")
                    print(f"    3. Use smaller model (e.g., DA3-BASE)")
                    self._cleanup_memory()
                raise
        
        # 最終統合
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing Complete")
            print(f"{'='*60}")
            print(f"Total chunks processed: {len(all_predictions)}")
            print(f"Total frames: {sum(p.depth.shape[0] for p in all_predictions)}")
        
        # チャンクを統合
        if len(all_predictions) > 1:
            merged_prediction = self._merge_predictions(all_predictions)
            
            # 統合結果を保存
            merged_output_path = os.path.join(output_dir, "merged_scene.npz")
            self._save_merged_prediction(merged_prediction, merged_output_path)
            
            if self.verbose:
                print(f"Merged scene saved to: {merged_output_path}")
        
        return all_predictions
    
    def _merge_predictions(self, predictions: List[Prediction]) -> Prediction:
        """
        複数のPredictionを統合
        
        Args:
            predictions: Predictionのリスト
            
        Returns:
            統合されたPrediction
        """
        merged = Prediction()
        
        # numpy配列を結合
        if all(p.depth is not None for p in predictions):
            merged.depth = np.concatenate([p.depth for p in predictions], axis=0)
        
        if all(p.conf is not None for p in predictions):
            merged.conf = np.concatenate([p.conf for p in predictions], axis=0)
        
        if all(p.extrinsics is not None for p in predictions):
            merged.extrinsics = np.concatenate([p.extrinsics for p in predictions], axis=0)
        
        if all(p.intrinsics is not None for p in predictions):
            merged.intrinsics = np.concatenate([p.intrinsics for p in predictions], axis=0)
        
        if all(p.processed_images is not None for p in predictions):
            merged.processed_images = np.concatenate(
                [p.processed_images for p in predictions], axis=0
            )
        
        return merged
    
    def _save_merged_prediction(self, prediction: Prediction, output_path: str):
        """
        統合されたPredictionをNPZファイルとして保存
        
        Args:
            prediction: 統合されたPrediction
            output_path: 出力ファイルパス
        """
        save_dict = {}
        
        if prediction.depth is not None:
            save_dict["depth"] = prediction.depth
        if prediction.conf is not None:
            save_dict["conf"] = prediction.conf
        if prediction.extrinsics is not None:
            save_dict["extrinsics"] = prediction.extrinsics
        if prediction.intrinsics is not None:
            save_dict["intrinsics"] = prediction.intrinsics
        if prediction.processed_images is not None:
            save_dict["processed_images"] = prediction.processed_images
        
        np.savez_compressed(output_path, **save_dict)


def main():
    """使用例"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VRAM効率的なシーン復元",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的な使用
  python vram_efficient_reconstruction.py /path/to/images --output ./output
  
  # チャンクサイズと解像度の指定
  python vram_efficient_reconstruction.py /path/to/images \\
      --output ./output \\
      --chunk-size 4 \\
      --process-res 448
  
  # より小さいモデルを使用
  python vram_efficient_reconstruction.py /path/to/images \\
      --output ./output \\
      --model depth-anything/DA3-BASE
        """
    )
    
    parser.add_argument("input", help="入力画像ディレクトリまたはパターン")
    parser.add_argument("--output", "-o", required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--model",
        default="depth-anything/DA3-LARGE",
        help="モデル名 (default: DA3-LARGE)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="チャンクサイズ（フレーム数） (default: 8)"
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="処理解像度 (default: 504)"
    )
    parser.add_argument(
        "--max-vram",
        type=float,
        default=11.0,
        help="目標VRAM上限（GB） (default: 11.0)"
    )
    parser.add_argument(
        "--export-format",
        default="mini_npz",
        help="エクスポート形式 (default: mini_npz)"
    )
    parser.add_argument(
        "--num-max-points",
        type=int,
        default=500_000,
        help="最大ポイント数 (default: 500,000)"
    )
    parser.add_argument(
        "--conf-thresh-percentile",
        type=float,
        default=50.0,
        help="信頼度閾値パーセンタイル (default: 50.0)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="デバイス (default: cuda)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="詳細ログを抑制"
    )
    
    args = parser.parse_args()
    
    # 入力画像の取得
    if os.path.isdir(args.input):
        # ディレクトリの場合
        image_paths = sorted(
            glob.glob(os.path.join(args.input, "*.png")) +
            glob.glob(os.path.join(args.input, "*.jpg")) +
            glob.glob(os.path.join(args.input, "*.jpeg"))
        )
    else:
        # パターンの場合
        image_paths = sorted(glob.glob(args.input))
    
    if not image_paths:
        print(f"❌ No images found in: {args.input}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Reconstructorの作成
    reconstructor = VRAMEfficientReconstructor(
        model_name=args.model,
        device=args.device,
        chunk_size=args.chunk_size,
        process_res=args.process_res,
        max_vram_gb=args.max_vram,
        verbose=not args.quiet,
    )
    
    # 処理実行
    try:
        predictions = reconstructor.process_scene(
            image_paths,
            args.output,
            export_format=args.export_format,
            num_max_points=args.num_max_points,
            conf_thresh_percentile=args.conf_thresh_percentile,
        )
        
        print(f"\n✅ Success! Output saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
