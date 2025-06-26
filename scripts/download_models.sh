#!/bin/bash

# HADM Server Model Download Script
set -e

echo "📦 Downloading HADM pretrained models..."

# Create pretrained_models directory if it doesn't exist
mkdir -p pretrained_models

cd pretrained_models

# Download EVA-02-L base model
echo "⬇️ Downloading EVA-02-L base model..."
if [ ! -f "eva02_L_coco_det_sys_o365.pth" ]; then
    wget -O eva02_L_coco_det_sys_o365.pth \
        "https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/det/eva02_L_coco_det_sys_o365.pth"
    echo "✅ EVA-02-L base model downloaded"
else
    echo "✅ EVA-02-L base model already exists"
fi

# Download HADM-L model
echo "⬇️ Downloading HADM-L model..."
if [ ! -f "HADM-L_0249999.pth" ]; then
    # Note: Replace with actual download link when available
    echo "❌ HADM-L model download link needed"
    echo "Please download manually from: https://www.dropbox.com/scl/fi/zwasvod906x1akzinnj3i/HADM-L_0249999.pth"
else
    echo "✅ HADM-L model already exists"
fi

# Download HADM-G model
echo "⬇️ Downloading HADM-G model..."
if [ ! -f "HADM-G_0249999.pth" ]; then
    # Note: Replace with actual download link when available
    echo "❌ HADM-G model download link needed"
    echo "Please download manually from: https://www.dropbox.com/scl/fi/bzj1m8p4cvm2vg4mai6uj/HADM-G_0249999.pth"
else
    echo "✅ HADM-G model already exists"
fi

cd ..

# Verify model files
echo "🔍 Verifying model files..."
ls -la pretrained_models/

echo ""
echo "📝 Model download status:"
echo "- EVA-02-L: $([ -f "pretrained_models/eva02_L_coco_det_sys_o365.pth" ] && echo "✅ Downloaded" || echo "❌ Missing")"
echo "- HADM-L: $([ -f "pretrained_models/HADM-L_0249999.pth" ] && echo "✅ Downloaded" || echo "❌ Missing")"
echo "- HADM-G: $([ -f "pretrained_models/HADM-G_0249999.pth" ] && echo "✅ Downloaded" || echo "❌ Missing")"
echo ""
echo "🚀 Ready to start the HADM server!" 