import tensorflow as tf
import json
from pathlib import Path
from typing import Dict
import numpy as np
from app.core.config import MLPaths, ConfidenceThresholds

class ModelService:
    def __init__(self):
        self.models: Dict[str, tf.keras.Model] = {}
        self.metrics: Dict[str, dict] = {}
        self.load_all_models()
    
    # ----------------------
    # LOAD METRICS & MODELS
    # ----------------------
    @staticmethod
    def load_json(path: Path):
        return json.loads(path.read_text())
        
    
    def load_all_models(self):
        try:
            self.models = {
                'anatomy': tf.keras.models.load_model(str(MLPaths.MODELS_DIR / 'anatomical_classifier.keras')),
                'pneumonia': tf.keras.models.load_model(str(MLPaths.MODELS_DIR / 'pneumonia_classifier.keras')),
                'osteoarthritis': tf.keras.models.load_model(str(MLPaths.MODELS_DIR / 'osteo_efficientnetb0.keras')),
                'covid': tf.keras.models.load_model(str(MLPaths.MODELS_DIR / 'covid19_model.keras')),
                'scan_type': tf.keras.models.load_model(str(MLPaths.MODELS_DIR / 'medical_scan_type_classifier.keras'))
            }
            
            self.load_metrics()
            print("All models and metrics loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
        
    
    def load_metrics(self):
        # Pneumonia metrics
        pneu_info      = self.load_json(MLPaths.PNEU_METRICS / 'dataset_info.json') 
        pneu_norm      = self.load_json(MLPaths.PNEU_METRICS / 'normalization_stats.json')
        line_pneu      = next(l for l in (MLPaths.PNEU_METRICS / 'thresholds.txt').read_text().splitlines() if 'opt_threshold' in l)
        pneu_thresh    = float(line_pneu.split('=')[1])
        pneu_last_conv = pneu_info.get('last_conv_layer', 'conv2d_2')
        pneu_size      = (pneu_info['resize_to']['height'], pneu_info['resize_to']['width'])

        # COVID metrics
        covid_info   = self.load_json(MLPaths.COVID_METRICS / 'dataset_info.json')
        covid_norm   = self.load_json(MLPaths.COVID_METRICS / 'normalization_stats.json')
        try:
            covid_thresh_data = self.load_json(MLPaths.COVID_METRICS / 'optimal_threshold.json')
            covid_thresh = float(covid_thresh_data.get('optimal_threshold', 0.592))
        except Exception as e:
            print(f"Warning: Could not load COVID threshold, using default: {e}")
            covid_thresh = 0.592
        covid_last_conv = covid_info.get('last_conv_layer', 'conv2d_2')
        covid_size   = (covid_info['resize_to']['height'], covid_info['resize_to']['width'])

        # Medical scan type metrics
        med_scan_info   = self.load_json(MLPaths.MEDICAL_SCAN_TYPE_METRICS / 'dataset_info.json')
        med_scan_norm   = self.load_json(MLPaths.MEDICAL_SCAN_TYPE_METRICS / 'normalization_stats.json')
        line_med_scan   = next(l for l in (MLPaths.MEDICAL_SCAN_TYPE_METRICS / 'thresholds.txt').read_text().splitlines() if 'opt_threshold' in l)
        med_scan_thresh = float(line_med_scan.split('=')[1])
        med_scan_last_conv = med_scan_info.get('last_conv_layer', 'conv2d_2')
        med_scan_size   = (med_scan_info['resize_to']['height'], med_scan_info['resize_to']['width'])

        # Anatomy metrics (also reused as osteoarthritis threshold)
        anat_info   = self.load_json(MLPaths.ANAT_METRICS / 'dataset_info.json')
        anat_norm   = self.load_json(MLPaths.ANAT_METRICS / 'normalization_stats.json')
        line_anat   = next(l for l in (MLPaths.ANAT_METRICS / 'thresholds.txt').read_text().splitlines() if 'opt_threshold' in l)
        anat_thresh = float(line_anat.split('=')[1])
        anat_size   = (anat_info['resize_to']['height'], anat_info['resize_to']['width'])

        # ✅ Assign to self.metrics dictionary
        self.metrics = {
            'pneumonia': {
                'norm': pneu_norm,
                'threshold': pneu_thresh,
                'last_conv': pneu_last_conv,
                'size': pneu_size
            },
            'covid': {
                'norm': covid_norm,
                'threshold': covid_thresh,
                'last_conv': covid_last_conv,
                'size': covid_size
            },
            'scan_type': {
                'norm': med_scan_norm,
                'threshold': med_scan_thresh,
                'last_conv': med_scan_last_conv,
                'size': med_scan_size
            },
            'anatomy': {
                'norm': anat_norm,
                'threshold': anat_thresh,
                'size': anat_size
            }
        }
