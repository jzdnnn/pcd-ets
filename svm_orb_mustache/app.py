"""
Main CLI application for SVM+ORB Mustache & Beard Try-On.

Usage:
    python app.py train --pos_dir data/faces --neg_dir data/non_faces
    python app.py eval --report reports/test_metrics.json
    python app.py infer --image input.jpg --out output.jpg
    python app.py webcam --camera 0 --mustache assets/mustaches/handlebar.png
"""

import argparse
import logging
import sys
import json
import cv2
from pathlib import Path
import numpy as np

from pipelines.dataset import FaceDataset
from pipelines.features import FeaturePipeline
from pipelines.train import SVMTrainer
from pipelines.infer import InferencePipeline
from pipelines.overlay import MustacheOverlay, MustacheGallery
from pipelines.utils import setup_logging, save_config, load_config


def setup_args():
    """Setup command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SVM+ORB Mustache & Beard Try-On System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train SVM classifier')
    train_parser.add_argument('--pos_dir', type=str, default='data/faces',
                            help='Directory with positive face samples')
    train_parser.add_argument('--neg_dir', type=str, default='data/non_faces',
                            help='Directory with negative samples')
    train_parser.add_argument('--cascade', type=str, 
                            default='assets/cascades/haarcascade_frontalface_default.xml',
                            help='Path to face cascade XML')
    train_parser.add_argument('--k', type=int, default=256,
                            help='Number of visual words for BoVW')
    train_parser.add_argument('--max_desc', type=int, default=200000,
                            help='Maximum descriptors for codebook')
    train_parser.add_argument('--orb_features', type=int, default=500,
                            help='Number of ORB features per image')
    train_parser.add_argument('--svm', type=str, default='linear', choices=['linear', 'rbf'],
                            help='SVM kernel type')
    train_parser.add_argument('--C', type=float, default=1.0,
                            help='SVM regularization parameter')
    train_parser.add_argument('--test_size', type=float, default=0.15,
                            help='Test set fraction')
    train_parser.add_argument('--val_size', type=float, default=0.15,
                            help='Validation set fraction')
    train_parser.add_argument('--auto_roi', action='store_true',
                            help='Auto-generate ROIs from full images')
    train_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--report', type=str, default='reports/test_metrics.json',
                           help='Path to save metrics report')
    eval_parser.add_argument('--plot_dir', type=str, default='reports',
                           help='Directory to save plots')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference on image')
    infer_parser.add_argument('--image', type=str, required=True,
                            help='Input image path')
    infer_parser.add_argument('--out', type=str, required=True,
                            help='Output image path')
    infer_parser.add_argument('--mustache', type=str, default='assets/mustaches/handlebar.png',
                            help='Mustache PNG path')
    infer_parser.add_argument('--cascade', type=str,
                            default='assets/cascades/haarcascade_frontalface_default.xml',
                            help='Face cascade path')
    infer_parser.add_argument('--eye_cascade', type=str,
                            default='assets/cascades/haarcascade_eye.xml',
                            help='Eye cascade path')
    infer_parser.add_argument('--scale', type=float, default=1.3,
                            help='Mustache scale factor')
    infer_parser.add_argument('--y_offset', type=float, default=0.55,
                            help='Mustache vertical offset ratio')
    infer_parser.add_argument('--show', action='store_true',
                            help='Display result')
    infer_parser.add_argument('--boxes', action='store_true',
                            help='Draw bounding boxes')
    
    # Webcam command
    webcam_parser = subparsers.add_parser('webcam', help='Run live webcam mode')
    webcam_parser.add_argument('--camera', type=int, default=0,
                             help='Camera device ID')
    webcam_parser.add_argument('--mustache', type=str, default='assets/mustaches/handlebar.png',
                             help='Mustache PNG path')
    webcam_parser.add_argument('--mustache_dir', type=str, default='assets/mustaches',
                             help='Directory with multiple mustache styles')
    webcam_parser.add_argument('--cascade', type=str,
                             default='assets/cascades/haarcascade_frontalface_default.xml',
                             help='Face cascade path')
    webcam_parser.add_argument('--eye_cascade', type=str,
                             default='assets/cascades/haarcascade_eye.xml',
                             help='Eye cascade path')
    webcam_parser.add_argument('--show', action='store_true', default=True,
                             help='Show video window')
    webcam_parser.add_argument('--save_screenshots', action='store_true',
                             help='Enable screenshot saving')
    
    return parser


def train_command(args):
    """Execute training pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = FaceDataset(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        face_cascade_path=args.cascade,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    stats = dataset.load_and_split(
        roi_size=(128, 128),
        auto_generate_rois=args.auto_roi
    )
    
    if stats.get('total', 0) == 0:
        logger.error("No data loaded! Please check your data directories.")
        return
    
    # Save split info
    dataset.save_split_info('models/dataset_split.json')
    
    # Get data
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()
    
    # Feature extraction
    logger.info("Building feature pipeline...")
    feature_pipeline = FeaturePipeline(
        orb_n_features=args.orb_features,
        bovw_k=args.k,
        random_state=args.seed
    )
    
    logger.info("Extracting training features...")
    X_train_features = feature_pipeline.fit_transform(X_train, max_descriptors=args.max_desc)
    
    logger.info("Extracting validation features...")
    X_val_features = feature_pipeline.transform(X_val)
    
    logger.info("Extracting test features...")
    X_test_features = feature_pipeline.transform(X_test)
    
    # Save codebook
    feature_pipeline.save_codebook('models/codebook.pkl')
    
    # Train SVM
    logger.info("Training SVM classifier...")
    svm_trainer = SVMTrainer(
        kernel=args.svm,
        random_state=args.seed
    )
    
    train_results = svm_trainer.train(
        X_train_features, y_train,
        X_val_features, y_val,
        cv_folds=5
    )
    
    # Save model
    svm_trainer.save('models/svm.pkl', 'models/scaler.pkl')
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = svm_trainer.evaluate(
        X_test_features, y_test,
        output_dir='reports'
    )
    
    # Save configuration
    config = {
        'orb_features': args.orb_features,
        'bovw_k': args.k,
        'svm_kernel': args.svm,
        'best_params': train_results['best_params'],
        'dataset_stats': stats,
        'train_results': train_results,
        'test_results': test_results,
        'random_seed': args.seed
    }
    
    save_config(config, 'models/config.json')
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {test_results['f1']:.4f}")
    logger.info(f"Test ROC-AUC: {test_results.get('roc_auc', 0):.4f}")
    logger.info("=" * 60)


def eval_command(args):
    """Execute evaluation on saved model."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    # Load configuration
    if not Path('models/config.json').exists():
        logger.error("No trained model found. Run 'train' first.")
        return
    
    config = load_config('models/config.json')
    
    # Load dataset split
    dataset = FaceDataset()
    # ... load from saved split
    
    logger.info("Evaluation results:")
    logger.info(json.dumps(config['test_results'], indent=2))
    
    # Save report
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w') as f:
        json.dump(config['test_results'], f, indent=2)
    
    logger.info(f"Report saved to {args.report}")


def infer_command(args):
    """Execute inference on single image."""
    logger = logging.getLogger(__name__)
    logger.info("Running inference...")
    
    # Check if model exists
    if not Path('models/svm.pkl').exists():
        logger.error("No trained model found. Run 'train' first.")
        return
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"Failed to load image: {args.image}")
        return
    
    logger.info(f"Image loaded: {image.shape}")
    
    # Load configuration
    config = load_config('models/config.json')
    
    # Setup feature pipeline
    feature_pipeline = FeaturePipeline(
        orb_n_features=config['orb_features'],
        bovw_k=config['bovw_k'],
        random_state=config['random_seed']
    )
    feature_pipeline.load_codebook('models/codebook.pkl')
    
    # Load SVM
    svm_trainer = SVMTrainer(
        kernel=config['svm_kernel'],
        random_state=config['random_seed']
    )
    svm_trainer.load('models/svm.pkl', 'models/scaler.pkl')
    
    # Setup mustache overlay
    mustache_overlay = None
    if Path(args.mustache).exists():
        mustache_overlay = MustacheOverlay(
            mustache_path=args.mustache,
            eye_cascade_path=args.eye_cascade if Path(args.eye_cascade).exists() else None
        )
    
    # Setup inference pipeline
    pipeline = InferencePipeline(
        face_cascade_path=args.cascade,
        feature_pipeline=feature_pipeline,
        svm_trainer=svm_trainer,
        mustache_overlay=mustache_overlay
    )
    
    # Process image
    output, faces, num_faces = pipeline.process_image(
        image,
        apply_overlay=(mustache_overlay is not None),
        draw_boxes=args.boxes
    )
    
    logger.info(f"Detected {num_faces} face(s)")
    
    # Save output
    cv2.imwrite(args.out, output)
    logger.info(f"Output saved to {args.out}")
    
    # Show if requested
    if args.show:
        cv2.imshow('Result', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print timing
    timing = pipeline.get_timing_report()
    logger.info(f"Timing: {timing}")


def webcam_command(args):
    """Execute live webcam mode."""
    logger = logging.getLogger(__name__)
    logger.info("Starting webcam mode...")
    
    # Check if model exists
    if not Path('models/svm.pkl').exists():
        logger.error("No trained model found. Run 'train' first.")
        return
    
    # Load configuration
    config = load_config('models/config.json')
    
    # Setup feature pipeline
    feature_pipeline = FeaturePipeline(
        orb_n_features=config['orb_features'],
        bovw_k=config['bovw_k'],
        random_state=config['random_seed']
    )
    feature_pipeline.load_codebook('models/codebook.pkl')
    
    # Load SVM
    svm_trainer = SVMTrainer(
        kernel=config['svm_kernel'],
        random_state=config['random_seed']
    )
    svm_trainer.load('models/svm.pkl', 'models/scaler.pkl')
    
    # Setup mustache overlay
    mustache_overlay = None
    if Path(args.mustache).exists():
        mustache_overlay = MustacheOverlay(
            mustache_path=args.mustache,
            eye_cascade_path=args.eye_cascade if Path(args.eye_cascade).exists() else None
        )
    
    # Setup inference pipeline
    pipeline = InferencePipeline(
        face_cascade_path=args.cascade,
        feature_pipeline=feature_pipeline,
        svm_trainer=svm_trainer,
        mustache_overlay=mustache_overlay
    )
    
    # Run webcam
    pipeline.process_webcam(
        camera_id=args.camera,
        apply_overlay=(mustache_overlay is not None),
        save_screenshots=args.save_screenshots
    )


def main():
    """Main entry point."""
    parser = setup_args()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'eval':
            eval_command(args)
        elif args.command == 'infer':
            infer_command(args)
        elif args.command == 'webcam':
            webcam_command(args)
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
