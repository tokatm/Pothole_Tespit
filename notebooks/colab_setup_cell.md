```python
from google.colab import drive
drive.mount('/content/drive')

# Projeyi /content/pothole-dfine altina kopyala
# !cp -r /content/drive/MyDrive/pothole-dfine /content/pothole-dfine

%cd /content/pothole-dfine
!python tools/colab_setup.py
!bash scripts/colab_bootstrap.sh
```
