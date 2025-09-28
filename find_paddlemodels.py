from paddleocr import PaddleOCR

# Ultra-lightweight mobile models with CPU optimization
ocr = PaddleOCR(
    # Model selection (NEW parameter names in 3.x)
    text_detection_model_name="PP-OCRv5_server_det",  # Ultra-lightweight detection
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",  # Ultra-lightweight recognition
    
    # Device specification (NEW in 3.x - replaces use_gpu)
    device="cpu",  # Critical: Force CPU-only mode
    
    # CPU optimization parameters
    enable_mkldnn=True,  # Enable Intel MKL-DNN acceleration
    cpu_threads=8,       # Adjust based on your CPU cores
    mkldnn_cache_capacity=10,  # MKL-DNN cache optimization
    
    # Feature toggles (NEW parameter names)
    use_doc_orientation_classify=False,  # Replaces use_angle_cls
    use_doc_unwarping=False,  # NEW feature - disable for speed
    use_textline_orientation=False,  # More specific than old use_angle_cls
    
    # Detection parameters (renamed in 3.x)
    text_det_limit_side_len=736,  # Replaces det_limit_side_len
    text_det_limit_type="min",    # Replaces det_limit_type
    text_det_thresh=0.3,          # Replaces det_db_thresh
    text_det_box_thresh=0.6,      # Replaces det_db_box_thresh
    text_det_unclip_ratio=1.5,    # Replaces det_db_unclip_ratio
    
    # Recognition parameters
    text_recognition_batch_size=6,  # Replaces rec_batch_num
    text_rec_score_thresh=0.0
)

# Process image
result = ocr.predict("test_screen.png")
for res in result:
    res.print()  # Display results
    res.save_to_img("output")  # Save annotated image
    res.save_to_json("output")  # Save JSON results
