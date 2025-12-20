1. Removed grid search to find the best params (clip limit, title)
   Cause:
   Result: Image A might get Clip=1.0, Tile=2, and Image B might get Clip=5.0, Tile=16.
   Impact on AI: If you are training a neural network ("LightXrayNet"), this inconsistent preprocessing confuses the model. The network will struggle to distinguish between actual biological features and the varying artifacts introduced by random preprocessing parameters.
   Solution:
   clipLimit=2.0, tileGridSize=(8, 8)

2. Removed horizontal flip from augmentation
   Cause:Changes the heart position
   Solution: removed horizontal flip, added
   ROTATION_RANGE = (-5.0, 5.0) # Degrees
   ZOOM_RANGE = (1.05, 1.15) # Zoom in
   TRANSLATION_FACTOR = 0.05 # Max shift 5%
   BRIGHTNESS_RANGE = (0.8, 1.2) # 80-120%
   CONTRAST_RANGE = (0.8, 1.2) # 80-120%
   NOISE_SIGMA = 5 # Gaussian noise level

3.switched to cv2.INTER_AREA from INTER LANCOSZ4 in resize
