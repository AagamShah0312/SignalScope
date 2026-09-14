+-------------------+
|    Input Image    |
+---------+---------+
          |
          v
+-------------------+
|   Preprocessing   | (Resizing, Normalization, Augmentation)
+---------+---------+
          |
          v
+-------------------+
| EfficientNet-B0   | (Feature Extraction Backbone)
+---------+---------+
          |
          v
+-------------------+
| Raw Probability   | (Binary Output Score)
+---------+---------+
          |
          v
+-------------------+
|   Calibration     | (Confidence Calibration Layer)
+---------+---------+
          |
          v
+-------------------+
|  Verdict Output   | ("Likely AI-Generated" vs "Likely Real")
+---------+---------+
          |
          +-----------------------------------+
          |                                   |
          v                                   v
+-------------------+               +-------------------+
|     Grad-CAM      |               | Text Explanation  |
| Visual Heat-Map   |               | Visual Cue Report |
+-------------------+               +-------------------+