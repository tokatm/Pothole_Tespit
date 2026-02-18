```python
from google.colab import drive
drive.mount('/content/drive')

# Projeyi /content/Pothole_Tespit altina kopyala
# !cp -r /content/drive/MyDrive/pothole-dfine /content/Pothole_Tespit

%cd /content/Pothole_Tespit
!python tools/colab_setup.py
!bash scripts/colab_bootstrap.sh
```
