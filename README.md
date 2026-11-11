# -CV-HW1-ImageEnhancement
1st homework Computer Vision course 2026/2027 student ID. VR565141  
NOTE: the images contained in the folders are exported as a JPEG file and compressed to upload in the repository. To have the original images, run the code.
## Info
Python version: 3.10  
numpy >= 1.24.0    
py-opencv>=4.8,<4.10  
## Commands to run the code
conda create -n cv_hw --override-channels -c conda-forge python=3.10 numpy -y  
conda activate cv_hw  
conda install --override-channels -c conda-forge "py-opencv>=4.8,<4.10" opencv -y  
python pipeline.py  
## File structure
```text
├── data/
│   ├── original/           # Original evaluation images
│   ├── degraded/           # Synthetically degraded versions
│   └── enhanced/           # Processed results for each image and method
├── results/
│   ├── figures/            # Side-by-side comparison images [Original | Degraded | Result]
│   └── metrics.csv
├── pipeline.py
├── README.md
└── Report.pdf
```


