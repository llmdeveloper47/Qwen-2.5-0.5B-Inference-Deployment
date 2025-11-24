#!/usr/bin/env python3
"""
Generate comprehensive PDF report from experiment results.

This script creates a professional PDF report summarizing:
- Experiment configuration
- Performance metrics
- Comparison tables
- Visualizations
- Recommendations
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, 
    Spacer, Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def create_title_page(elements, styles):
    """Create report title page."""
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#666666'),
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("Intent Classification Model", title_style))
    elements.append(Paragraph("Quantization & Latency Experiment Report", title_style))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(f"Model: codefactory4791/intent-classification-qwen", subtitle_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    elements.append(PageBreak())


def create_summary_section(elements, styles, summary_df):
    """Create summary section with comparison table."""
    heading_style = styles['Heading2']
    
    elements.append(Paragraph("1. Performance Summary", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Create table data
    table_data = [
        ['Quantization', 'Batch Size', 'Avg Latency (ms)', 'P95 (ms)', 'P99 (ms)', 'Throughput (samples/s)']
    ]
    
    for _, row in summary_df.iterrows():
        table_data.append([
            row['quantization'],
            str(int(row['batch_size'])),
            f"{row['avg_latency_ms']:.2f}",
            f"{row['p95_latency_ms']:.2f}",
            f"{row['p99_latency_ms']:.2f}",
            f"{row['throughput']:.2f}",
        ])
    
    # Create table
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.5*inch))


def create_visualizations_section(elements, styles, output_dir):
    """Add visualization images to report."""
    heading_style = styles['Heading2']
    
    elements.append(Paragraph("2. Performance Visualizations", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add latency comparison plot
    latency_plot = Path(output_dir) / "latency_comparison.png"
    if latency_plot.exists():
        img = Image(str(latency_plot), width=7*inch, height=5.25*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))
    
    elements.append(PageBreak())
    
    # Add efficiency analysis plot
    efficiency_plot = Path(output_dir) / "efficiency_analysis.png"
    if efficiency_plot.exists():
        elements.append(Paragraph("2.2 Efficiency Analysis", styles['Heading3']))
        elements.append(Spacer(1, 0.2*inch))
        img = Image(str(efficiency_plot), width=7*inch, height=2.62*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))


def create_recommendations_section(elements, styles, recommendations):
    """Create recommendations section."""
    heading_style = styles['Heading2']
    subheading_style = styles['Heading3']
    
    elements.append(PageBreak())
    elements.append(Paragraph("3. Recommendations", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Best for latency
    elements.append(Paragraph("3.1 Best for Low Latency", subheading_style))
    rec = recommendations['best_latency']
    text = f"""
    For applications requiring minimal latency:<br/>
    <b>Quantization:</b> {rec['quantization']}<br/>
    <b>Batch Size:</b> {rec['batch_size']}<br/>
    <b>P95 Latency:</b> {rec['p95_latency_ms']}ms<br/>
    <b>Throughput:</b> {rec['throughput']} samples/s
    """
    elements.append(Paragraph(text, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Best for throughput
    elements.append(Paragraph("3.2 Best for High Throughput", subheading_style))
    rec = recommendations['best_throughput']
    text = f"""
    For maximum throughput:<br/>
    <b>Quantization:</b> {rec['quantization']}<br/>
    <b>Batch Size:</b> {rec['batch_size']}<br/>
    <b>P95 Latency:</b> {rec['p95_latency_ms']}ms<br/>
    <b>Throughput:</b> {rec['throughput']} samples/s
    """
    elements.append(Paragraph(text, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Best balanced
    elements.append(Paragraph("3.3 Best Balanced Configuration", subheading_style))
    rec = recommendations['best_balance']
    text = f"""
    For optimal latency/throughput trade-off:<br/>
    <b>Quantization:</b> {rec['quantization']}<br/>
    <b>Batch Size:</b> {rec['batch_size']}<br/>
    <b>P95 Latency:</b> {rec['p95_latency_ms']}ms<br/>
    <b>Throughput:</b> {rec['throughput']} samples/s<br/>
    <b>Balance Score:</b> {rec['balance_score']}
    """
    elements.append(Paragraph(text, styles['Normal']))


def generate_pdf_report(results_dir: str, output_file: str):
    """Generate comprehensive PDF report."""
    print("\n" + "=" * 70)
    print("Generating PDF Report")
    print("=" * 70)
    
    # Load data
    analysis_dir = Path(results_dir) / "analysis"
    
    # Load summary table
    summary_file = analysis_dir / "comparison_table.csv"
    if not summary_file.exists():
        print(f"✗ Summary file not found: {summary_file}")
        print(f"  Run analyze_results.py first")
        sys.exit(1)
    
    summary_df = pd.read_csv(summary_file)
    
    # Load recommendations
    rec_file = analysis_dir / "recommendations.json"
    if rec_file.exists():
        with open(rec_file) as f:
            recommendations = json.load(f)
    else:
        recommendations = {}
    
    # Create PDF
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Build report sections
    create_title_page(elements, styles)
    create_summary_section(elements, styles, summary_df)
    create_visualizations_section(elements, styles, analysis_dir)
    
    if recommendations:
        create_recommendations_section(elements, styles, recommendations)
    
    # Build PDF
    doc.build(elements)
    
    print(f"\n✓ PDF report generated: {output_file}")
    print(f"  Pages: ~{len(elements) // 10}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF report from experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing analysis results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/experiment_report.pdf",
        help="Output PDF file path"
    )
    
    args = parser.parse_args()
    
    # Generate report
    generate_pdf_report(args.results_dir, args.output)
    
    print("\n" + "=" * 70)
    print("✓ Report generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

