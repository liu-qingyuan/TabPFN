#!/usr/bin/env python3
"""
Figure combination utilities for Nature journal-style multi-panel figures.

This module provides functionality to combine individual PDF figures into 
Nature-standard multi-panel layouts with proper formatting and labeling.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import io
import warnings
from typing import Optional, List, Tuple

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

warnings.filterwarnings('ignore')

# Nature journal font settings
def apply_nature_style():
    """Apply Nature journal style settings to matplotlib."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'text.usetex': False,
        'mathtext.default': 'regular',
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight'
    })

class FigureCombiner:
    """Utility class for combining individual figures into multi-panel layouts."""
    
    def __init__(self):
        """Initialize the FigureCombiner with Nature journal settings."""
        apply_nature_style()
    
    def pdf_to_image(self, pdf_path: Path) -> Optional[Image.Image]:
        """Convert PDF to PIL Image."""
        if FITZ_AVAILABLE:
            try:
                # Open PDF and get first page
                doc = fitz.open(str(pdf_path))
                page = doc.load_page(0)
                
                # Render page to pixmap (image) at high DPI
                mat = fitz.Matrix(2.0, 2.0)  # 2x scaling for high quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                doc.close()
                
                return img
                
            except Exception as e:
                print(f"⚠️ Error converting {pdf_path} to image: {e}")
                return None
        else:
            # Fallback to pdf2image or ghostscript
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
                if images:
                    return images[0]
            except ImportError:
                pass
            
            # Alternative: Use ghostscript via subprocess
            try:
                import subprocess
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    cmd = [
                        'gs', '-dNOPAUSE', '-dBATCH', '-dSAFER', 
                        '-sDEVICE=png16m', '-r200',
                        f'-sOutputFile={tmp.name}',
                        str(pdf_path)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    img = Image.open(tmp.name)
                    Path(tmp.name).unlink()  # Clean up
                    return img
                    
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"⚠️ Ghostscript not available or failed for {pdf_path}")
            
            print(f"❌ No PDF conversion method available for {pdf_path}")
            return None

    def combine_analysis_figures(self, input_dir: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Combine ROC, Calibration, and Decision Curve Analysis figures into one Nature-style figure.
        
        Layout order as requested:
        - ROC curves (top row, 2 panels: a, b)
        - Calibration curves (middle row, 2 panels: c, d) 
        - Decision curve analysis (bottom row, 2 panels: e, f)
        
        Args:
            input_dir: Directory containing individual PDF figures
            output_path: Output path for combined figure (optional)
            
        Returns:
            Path to the saved combined figure or None if failed
        """
        # Find required files (try both PDF and PNG)
        roc_path = None
        calib_path = None
        decision_path = None
        
        # Prioritize PNG files for Linux compatibility (no PDF conversion tools needed)
        # Check for ROC curves in roc_curves subdirectory
        roc_subdir = input_dir / "roc_curves"
        if roc_subdir.exists():
            for ext in ['.png', '.pdf']:  # PNG first for Linux compatibility
                roc_candidate = roc_subdir / f"roc_comparison{ext}"
                if roc_candidate.exists():
                    roc_path = roc_candidate
                    break
        
        # Check for main figures
        for ext in ['.png', '.pdf']:  # PNG first for Linux compatibility
            if roc_path is None:
                roc_candidate = input_dir / f"roc_comparison{ext}"
                if roc_candidate.exists():
                    roc_path = roc_candidate
                    
            calib_candidate = input_dir / f"calibration_curves{ext}"
            decision_candidate = input_dir / f"decision_curve_analysis{ext}"
            
            if calib_candidate.exists():
                calib_path = calib_candidate
            if decision_candidate.exists():
                decision_path = decision_candidate
        
        # Check if all required files are found
        missing_files = []
        if roc_path is None:
            missing_files.append("roc_comparison")
        if calib_path is None:
            missing_files.append("calibration_curves")
        if decision_path is None:
            missing_files.append("decision_curve_analysis")
            
        if missing_files:
            print(f"⚠️ Required files not found: {', '.join(missing_files)}")
            return None
        
        # Load images (PNG preferred, PDF fallback)
        print(f"📊 Loading figures for combination...")
        print(f"   ROC curves: {roc_path}")
        print(f"   Calibration curves: {calib_path}")
        print(f"   Decision curve analysis: {decision_path}")
        
        # Load images - PNG files are loaded directly, PDF files need conversion
        if roc_path.suffix.lower() == '.png':
            roc_img = Image.open(roc_path)
        else:
            roc_img = self.pdf_to_image(roc_path)
            
        if calib_path.suffix.lower() == '.png':
            calib_img = Image.open(calib_path)
        else:
            calib_img = self.pdf_to_image(calib_path)
            
        if decision_path.suffix.lower() == '.png':
            decision_img = Image.open(decision_path)
        else:
            decision_img = self.pdf_to_image(decision_path)
        
        if roc_img is None or calib_img is None or decision_img is None:
            print("⚠️ Failed to convert some PDFs to images")
            return None
        
        print(f"   ROC image size: {roc_img.size}")
        print(f"   Calibration image size: {calib_img.size}")
        print(f"   Decision curve image size: {decision_img.size}")
        
        # Create figure with 3x2 layout (6 panels)
        fig = plt.figure(figsize=(16, 18))
        
        # Calculate subplot positions for Nature journal style
        left_margin = 0.08
        right_margin = 0.95
        bottom_margin = 0.05
        top_margin = 0.95
        
        subplot_width = (right_margin - left_margin - 0.03) / 2  # 0.03 for spacing between panels
        subplot_height = (top_margin - bottom_margin - 0.06) / 3  # 0.06 for spacing between rows
        
        # Split each figure (each should have 2 subplots side by side)
        # ROC curves
        roc_width, roc_height = roc_img.size
        roc_left = roc_img.crop((0, 0, roc_width//2, roc_height))
        roc_right = roc_img.crop((roc_width//2, 0, roc_width, roc_height))
        
        # Calibration curves  
        calib_width, calib_height = calib_img.size
        calib_left = calib_img.crop((0, 0, calib_width//2, calib_height))
        calib_right = calib_img.crop((calib_width//2, 0, calib_width, calib_height))
        
        # Decision curve analysis
        decision_width, decision_height = decision_img.size
        decision_left = decision_img.crop((0, 0, decision_width//2, decision_height))
        decision_right = decision_img.crop((decision_width//2, 0, decision_width, decision_height))
        
        # Position subplots (top to bottom: ROC, Calibration, Decision)
        positions = [
            # Top row (ROC curves) - a, b
            [left_margin, bottom_margin + 2*subplot_height + 0.06, subplot_width, subplot_height],  # a
            [left_margin + subplot_width + 0.03, bottom_margin + 2*subplot_height + 0.06, subplot_width, subplot_height],  # b
            # Middle row (Calibration curves) - c, d
            [left_margin, bottom_margin + subplot_height + 0.03, subplot_width, subplot_height],  # c
            [left_margin + subplot_width + 0.03, bottom_margin + subplot_height + 0.03, subplot_width, subplot_height],  # d
            # Bottom row (Decision curves) - e, f
            [left_margin, bottom_margin, subplot_width, subplot_height],  # e
            [left_margin + subplot_width + 0.03, bottom_margin, subplot_width, subplot_height]   # f
        ]
        
        images = [roc_left, roc_right, calib_left, calib_right, decision_left, decision_right]
        labels = ['a', 'b', 'c', 'd', 'e', 'f']
        
        for i, (pos, img, label) in enumerate(zip(positions, images, labels)):
            ax = fig.add_axes(pos)
            ax.imshow(img, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove all spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add Nature-style panel label
            ax.text(0.02, 0.98, label, transform=ax.transAxes, 
                    fontsize=20, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.9))
        
        if output_path is None:
            output_path = input_dir / "combined_analysis_figures.pdf"
        
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Combined analysis figures saved: {output_path}")
        return output_path
    
    def combine_heatmap_figures(self, input_dir: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Combine source_cv_heatmap and uda_methods_heatmap into one figure
        with 2 subplots (1x2) labeled a, b
        """
        # Prioritize PNG files for Linux compatibility
        source_path = None
        uda_path = None
        
        for ext in ['.png', '.pdf']:  # PNG first for Linux compatibility
            source_candidate = input_dir / f"source_cv_heatmap{ext}"
            uda_candidate = input_dir / f"uda_methods_heatmap{ext}"
            
            if source_candidate.exists() and uda_candidate.exists():
                source_path = source_candidate
                uda_path = uda_candidate
                break
        
        if source_path is None or uda_path is None:
            print(f"⚠️ Required heatmap files not found:")
            print(f"   source_cv_heatmap: {'✓' if source_path else '✗'}")
            print(f"   uda_methods_heatmap: {'✓' if uda_path else '✗'}")
            return None
        
        # Load images (PNG preferred, PDF fallback)
        if source_path.suffix.lower() == '.png':
            source_img = Image.open(source_path)
            uda_img = Image.open(uda_path)
        else:
            source_img = self.pdf_to_image(source_path)
            uda_img = self.pdf_to_image(uda_path)
        
        if source_img is None or uda_img is None:
            print("⚠️ Failed to convert heatmap PDFs to images")
            return None
        
        print(f"📊 Combining heatmap figures...")
        print(f"   Source CV heatmap size: {source_img.size}")
        print(f"   UDA methods heatmap size: {uda_img.size}")
        
        # Create figure with 1x2 layout
        fig = plt.figure(figsize=(16, 8))
        
        # Calculate subplot positions
        left_margin = 0.05
        right_margin = 0.95
        bottom_margin = 0.1
        top_margin = 0.9
        
        subplot_width = (right_margin - left_margin - 0.05) / 2
        subplot_height = top_margin - bottom_margin
        
        positions = [
            [left_margin, bottom_margin, subplot_width, subplot_height],  # left (a)
            [left_margin + subplot_width + 0.05, bottom_margin, subplot_width, subplot_height]   # right (b)
        ]
        
        images = [source_img, uda_img]
        labels = ['a', 'b']
        
        for i, (pos, img, label) in enumerate(zip(positions, images, labels)):
            ax = fig.add_axes(pos)
            ax.imshow(img, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove all spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add Nature-style panel label
            ax.text(0.02, 0.98, label, transform=ax.transAxes, 
                    fontsize=20, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
        
        if output_path is None:
            output_path = input_dir / "combined_heatmaps.pdf"
        
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Combined heatmap figure saved: {output_path}")
        return output_path

def create_combined_figures(input_dir: Path) -> List[Path]:
    """
    Create all combined figures for a given results directory.
    
    Args:
        input_dir: Directory containing individual analysis figures
        
    Returns:
        List of paths to created combined figures
    """
    combiner = FigureCombiner()
    created_files = []
    
    print(f"🔄 Creating combined figures for: {input_dir}")
    
    # Create main analysis figure (ROC + Calibration + DCA)
    main_figure = combiner.combine_analysis_figures(input_dir)
    if main_figure:
        created_files.append(main_figure)
    
    # Create heatmap combination if heatmaps exist
    heatmap_figure = combiner.combine_heatmap_figures(input_dir)
    if heatmap_figure:
        created_files.append(heatmap_figure)
    
    if created_files:
        print(f"\n✅ Combined figures created:")
        for file_path in created_files:
            print(f"   - {file_path}")
    else:
        print(f"⚠️ No combined figures were created")
    
    return created_files