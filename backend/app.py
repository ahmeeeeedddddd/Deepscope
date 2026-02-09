from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import json
from typing import Dict, Any, Optional
import logging

# Import custom modules
from model_handler import ModelHandler
from preprocessing import preprocess_image


# ═══════════════════════════════════════════════════════════
# FLASK APP CONFIGURATION
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Create upload directory if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# GLOBAL INSTANCES
# ═══════════════════════════════════════════════════════════

# Initialize model handler
model_handler = None


llm_client = None

# Load configuration
with open('../data/classification_config.json', 'r') as f:
    config = json.load(f)


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def validate_clinical_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate clinical data from frontend.
    
    Args:
        data: Dictionary containing patient clinical information
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['age', 'sex']
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate age
    try:
        age = int(data['age'])
        if age < 0 or age > 120:
            return False, "Age must be between 0 and 120"
    except (ValueError, TypeError):
        return False, "Age must be a valid number"
    
    # Validate sex
    if data['sex'] not in ['Male', 'Female', 'Other']:
        return False, "Sex must be Male, Female, or Other"
    
    # Additional info is optional - just check it exists in data structure
    if 'additional_info' in data and not isinstance(data['additional_info'], str):
        return False, "Additional info must be a string"
    
    return True, None


# ═══════════════════════════════════════════════════════════
# TASK 5.5 - TREATMENT PLAN GENERATION
# ═══════════════════════════════════════════════════════════

def generate_template_treatment_plan(
    diagnosis: str,
    confidence: float,
    tissue_type: str,
    patient_data: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate a template-based treatment plan (fallback when LLM is unavailable).
    
    This provides structured recommendations based on diagnosis and patient data.
    When LLM is available, it will generate more personalized plans.
    
    Args:
        diagnosis: Disease classification (e.g., 'Malignant', 'Benign')
        confidence: Model confidence (0-1)
        tissue_type: Tissue classification (TUM, NORM, etc.)
        patient_data: Patient clinical information
        
    Returns:
        Dictionary with treatment plan sections
    """
    age = patient_data.get('age', 'N/A')
    sex = patient_data.get('sex', 'N/A')
    comorbidities = patient_data.get('comorbidities', [])
    activity_level = patient_data.get('activity_level', 'moderate')
    smoking_status = patient_data.get('smoking_status', 'non-smoker')
    family_history = patient_data.get('family_history', False)
    additional_info = patient_data.get('additional_info', '')
    
    # Determine severity
    is_malignant = 'malignant' in diagnosis.lower() or tissue_type == 'TUM'
    is_high_confidence = confidence >= 0.85
    
    # Build treatment plan
    treatment_plan = {}
    
    # Include additional information if provided
    additional_context = ""
    if additional_info:
        additional_context = f"\n\nAdditional Patient Information: {additional_info}\n"

    # Medical Treatment
# Medical Treatment
    if is_malignant:
        treatment_plan['medical_treatment'] = (
            f"⚠️ URGENT: The AI model detected potential malignant tissue (confidence: {confidence:.1%}). "
            f"This requires immediate medical evaluation.{additional_context}\n\n"
            f"Recommended Next Steps:\n"
            f"1. Schedule consultation with an oncologist within 1-2 weeks\n"
            f"2. Additional diagnostic testing (colonoscopy, biopsy, CT scan)\n"
            f"3. Staging assessment to determine extent\n"
            f"4. Discuss treatment options: surgery, chemotherapy, radiation therapy\n\n"
            f"Treatment will depend on stage, location, and overall health status."
        )
    else:
        treatment_plan['medical_treatment'] = (
            f"✓ The AI model detected benign tissue (confidence: {confidence:.1%}).{additional_context}\n\n"
            f"Recommended Next Steps:\n"
            f"1. Follow-up with gastroenterologist for confirmation\n"
            f"2. Regular screening colonoscopy as per guidelines\n"
            f"3. Monitor for any changes in symptoms\n"
            f"4. Continue routine preventive care"
        )
    
    # Lifestyle Modifications (personalized by age and activity)
    lifestyle_recs = []
    
    if age < 50:
        lifestyle_recs.append("- Establish healthy habits early to reduce long-term risk")
    elif age >= 65:
        lifestyle_recs.append("- Age-appropriate exercise with medical clearance")
        lifestyle_recs.append("- Fall prevention measures")
    
    if activity_level == 'sedentary':
        lifestyle_recs.append("- Gradually increase physical activity (start with 10-15 min/day)")
        lifestyle_recs.append("- Reduce prolonged sitting time")
    
    if smoking_status in ['current_smoker', 'smoker']:
        lifestyle_recs.append("- ⚠️ CRITICAL: Smoking cessation program - significantly increases cancer risk")
        lifestyle_recs.append("- Consider nicotine replacement therapy or counseling")
    
    if comorbidities:
        lifestyle_recs.append(f"- Manage existing conditions: {', '.join(comorbidities)}")
    
    if family_history:
        lifestyle_recs.append("- Increased screening frequency due to family history")
        lifestyle_recs.append("- Consider genetic counseling")
    
    treatment_plan['lifestyle_modifications'] = "\\n".join(lifestyle_recs) if lifestyle_recs else "- Maintain current healthy lifestyle"
    
    # Diet Recommendations
    diet_recs = [
        "Colorectal Cancer Prevention Diet:",
        "- HIGH FIBER: 25-35g daily (whole grains, vegetables, fruits, legumes)",
        "- LIMIT RED MEAT: <500g/week, avoid processed meats",
        "- INCREASE: Leafy greens, cruciferous vegetables (broccoli, cauliflower)",
        "- HYDRATION: 8-10 glasses of water daily",
        "- LIMIT ALCOHOL: Maximum 1 drink/day (women), 2 drinks/day (men)",
        "- AVOID: Highly processed foods, excessive sugar"
    ]
    
    if 'diabetes' in [c.lower() for c in comorbidities]:
        diet_recs.append("- DIABETES: Focus on low glycemic index foods, portion control")
    
    if 'hypertension' in [c.lower() for c in comorbidities]:
        diet_recs.append("- HYPERTENSION: Low sodium diet (<2300mg/day)")
    
    treatment_plan['diet_recommendations'] = "\\n".join(diet_recs)
    
    # Exercise Guidance
    if activity_level == 'sedentary':
        exercise_plan = (
            "Beginner Exercise Plan:\\n"
            "- Week 1-2: 10-15 minutes walking daily\\n"
            "- Week 3-4: 20-30 minutes walking, 5 days/week\\n"
            "- Month 2+: 150 minutes moderate activity/week\\n"
            "- Add strength training 2x/week"
        )
    elif activity_level == 'active':
        exercise_plan = (
            "Maintain Current Activity:\\n"
            "- Continue 150+ minutes moderate-to-vigorous activity/week\\n"
            "- Include variety: cardio, strength, flexibility\\n"
            "- Listen to your body, avoid overtraining"
        )
    else:
        exercise_plan = (
            "Recommended Physical Activity:\\n"
            "- 150 minutes moderate aerobic activity per week\\n"
            "- Or 75 minutes vigorous activity per week\\n"
            "- Strength training 2+ days/week\\n"
            "- Examples: brisk walking, cycling, swimming, yoga"
        )
    
    if age >= 65 or comorbidities:
        exercise_plan += "\\n\\n⚠️ Consult physician before starting new exercise program"
    
    treatment_plan['exercise_guidance'] = exercise_plan
    
    # Next Steps
    next_steps = [
        f"1. IMMEDIATE: Consult with gastroenterologist to discuss this AI-assisted diagnosis",
        f"2. Bring this report and original pathology images to your appointment",
        f"3. Schedule appropriate diagnostic procedures (colonoscopy, imaging)",
    ]
    
    if is_malignant:
        next_steps.append("4. ⚠️ URGENT: Seek specialist consultation within 1-2 weeks")
        next_steps.append("5. Consider getting a second opinion from a cancer center")
    else:
        next_steps.append("4. Follow screening guidelines for your age and risk factors")
        next_steps.append("5. Report any new symptoms immediately (bleeding, pain, weight loss)")
    
    treatment_plan['next_steps'] = "\\n".join(next_steps)
    
    # Follow-up Schedule
    if is_malignant:
        followup = (
            "URGENT Follow-up Required:\\n"
            "- Oncologist consultation: Within 1-2 weeks\\n"
            "- Diagnostic imaging: Within 2-3 weeks\\n"
            "- Treatment planning: Within 3-4 weeks\\n"
            "- Post-treatment monitoring: Every 3-6 months for 5 years"
        )
    else:
        followup = (
            "Standard Follow-up Schedule:\\n"
            "- Gastroenterologist visit: Within 4-6 weeks\\n"
            "- Screening colonoscopy: Per physician recommendation\\n"
            "- Annual check-up: Monitor overall health\\n"
            "- Report any symptoms immediately"
        )
    
    if family_history or age >= 50:
        followup += "\\n- Consider more frequent screening due to risk factors"
    
    treatment_plan['follow_up'] = followup
    
    # Add disclaimer
    treatment_plan['disclaimer'] = (
        "⚠️ IMPORTANT DISCLAIMER:\\n"
        "This treatment plan is generated by AI based on the diagnostic results and patient data provided. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of qualified health providers with any questions regarding a medical condition. "
        f"AI Confidence: {confidence:.1%} - Further clinical validation required."
    )
    
    return treatment_plan


# ═══════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_handler is not None,
        'llm_available': llm_client is not None
    })


@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """
    Endpoint for image diagnosis using ViT model.
    
    Expected: multipart/form-data with 'image' file
    
    Returns:
        JSON with diagnosis results:
        {
            'success': bool,
            'diagnosis': str,
            'confidence': float,
            'class_probabilities': dict,
            'tissue_type': str
        }
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        # Check if file has a filename
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {app.config["ALLOWED_EXTENSIONS"]}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Image uploaded: {filename}")
        
        # Run model inference
        if model_handler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please initialize the server.'
            }), 500
        
        # Get prediction from model
        prediction = model_handler.predict(filepath)
        
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        
        logger.info(f"Diagnosis complete: {prediction['diagnosis']}")
        
        return jsonify({
            'success': True,
            **prediction
        })
    
    except Exception as e:
        logger.error(f"Error in diagnosis endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    

@app.route('/api/classify', methods=['POST'])
def classify():
    """
    Endpoint for image classification (alias for diagnose).
    This matches the frontend's expected endpoint.
    """
    return diagnose()  # Just call the diagnose function    


@app.route('/api/treatment', methods=['POST'])
def get_treatment_recommendation():
    """
    Endpoint for generating treatment recommendations using LLM.
    
    Expected JSON:
    {
        'diagnosis': str,
        'confidence': float,
        'tissue_type': str,
        'patient_data': {
            'age': int,
            'sex': str,
            'comorbidities': list,
            'activity_level': str,
            'smoking_status': str,
            'family_history': bool,
            ...
        }
    }
    
    Returns:
        JSON with treatment recommendations:
        {
            'success': bool,
            'treatment_plan': {
                'medical_treatment': str,
                'lifestyle_modifications': str,
                'diet_recommendations': str,
                'exercise_guidance': str,
                'next_steps': str,
                'follow_up': str
            }
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['diagnosis', 'confidence', 'tissue_type', 'patient_data']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate patient data
        is_valid, error_msg = validate_clinical_data(data['patient_data'])
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Invalid patient data: {error_msg}'
            }), 400
        
        logger.info(f"Generating treatment recommendation for: {data['diagnosis']}")
        
        # ═══════════════════════════════════════════════════════════
        # TASK 5.5 - GENERATE PERSONALIZED TREATMENT PLANS
        # ═══════════════════════════════════════════════════════════
        
        # Check if LLM client is initialized
        if llm_client is None:
            # Fallback: Return structured response without LLM
            logger.warning("LLM client not initialized, returning template response")
            treatment_plan = generate_template_treatment_plan(
                diagnosis=data['diagnosis'],
                confidence=data['confidence'],
                tissue_type=data['tissue_type'],
                patient_data=data['patient_data']
            )
        else:
            # Use LLM to generate personalized treatment plan
            try:
                treatment_plan = llm_client.generate_treatment_plan(
                    diagnosis=data['diagnosis'],
                    confidence=data['confidence'],
                    tissue_type=data['tissue_type'],
                    patient_data=data['patient_data']
                )
            except Exception as llm_error:
                logger.error(f"LLM generation failed: {str(llm_error)}, using template")
                treatment_plan = generate_template_treatment_plan(
                    diagnosis=data['diagnosis'],
                    confidence=data['confidence'],
                    tissue_type=data['tissue_type'],
                    patient_data=data['patient_data']
                )
        
        return jsonify({
            'success': True,
            'treatment_plan': treatment_plan
        })
        # ═══════════════════════════════════════════════════════════
    
    except Exception as e:
        logger.error(f"Error in treatment endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/complete-diagnosis', methods=['POST'])
def complete_diagnosis_and_treatment():
    """
    Combined endpoint: Upload image + clinical data → Get diagnosis + treatment.
    
    Expected: multipart/form-data with:
    - 'image': file
    - 'patient_data': JSON string
    
    Returns:
        JSON with both diagnosis and treatment recommendations
    """
    try:
        # Step 1: Get diagnosis from image
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        # Get and validate patient data
        patient_data_str = request.form.get('patient_data')
        if not patient_data_str:
            return jsonify({
                'success': False,
                'error': 'No patient data provided'
            }), 400
        
        try:
            patient_data = json.loads(patient_data_str)
        except json.JSONDecodeError:
            return jsonify({
                'success': False,
                'error': 'Invalid patient data JSON'
            }), 400
        
        # Validate patient data
        is_valid, error_msg = validate_clinical_data(patient_data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Invalid patient data: {error_msg}'
            }), 400
        
        # Process image
        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {app.config["ALLOWED_EXTENSIONS"]}'
            }), 400
        
        # Save and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get diagnosis
        if model_handler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        diagnosis_result = model_handler.predict(filepath)
        
        logger.info(f"Complete diagnosis request: {diagnosis_result['diagnosis']}")
        
        # ═══════════════════════════════════════════════════════════
        # TASK 5.5 - GENERATE TREATMENT PLAN
        # ═══════════════════════════════════════════════════════════
        
        # Generate treatment plan
        if llm_client is None:
            treatment_plan = generate_template_treatment_plan(
                diagnosis=diagnosis_result['diagnosis'],
                confidence=diagnosis_result['confidence'],
                tissue_type=diagnosis_result['tissue_type'],
                patient_data=patient_data
            )
        else:
            try:
                treatment_plan = llm_client.generate_treatment_plan(
                    diagnosis=diagnosis_result['diagnosis'],
                    confidence=diagnosis_result['confidence'],
                    tissue_type=diagnosis_result['tissue_type'],
                    patient_data=patient_data
                )
            except Exception as llm_error:
                logger.error(f"LLM failed, using template: {str(llm_error)}")
                treatment_plan = generate_template_treatment_plan(
                    diagnosis=diagnosis_result['diagnosis'],
                    confidence=diagnosis_result['confidence'],
                    tissue_type=diagnosis_result['tissue_type'],
                    patient_data=patient_data
                )
        # ═══════════════════════════════════════════════════════════
        
        return jsonify({
            'success': True,
            'diagnosis': diagnosis_result,
            'treatment_plan': treatment_plan
        })
    
    except Exception as e:
        logger.error(f"Error in complete diagnosis endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ═══════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════

def initialize_model():
    """Initialize the ViT model handler."""
    global model_handler
    
    try:
        logger.info("Loading ViT model...")
        model_handler = ModelHandler(
            model_path='../models/vit_best.pth',
            config_path='../data/classification_config.json',
            device='cuda',  # or 'cpu'
            reference_image_path='D:/NCT-CRC-HE-100K/TUM/TUM-EDPETKWQ.tif',  # ← Add this!
            use_stain_norm=True  # ← Add this!
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def initialize_llm():
    """Initialize the LLM client."""
    global llm_client
    
    try:
        logger.info("Initializing LLM client...")
        from llm_client import LLMClient
        llm_client = LLMClient(
            api_key='AIzaSyBORWZtS3vg_7nkRUfKp5knFfLLKqEN_PY',  # Replace with your key
            model_name='gemini-3-flash-preview',
            mock_mode=False,  # Use real API
            debug=False  # Set True for debugging
        )
        logger.info("✓ LLM client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        logger.info("Falling back to template-based treatment plans")


# ═══════════════════════════════════════════════════════════
# TASK 5.6 - TEST WITH VARIOUS PATIENT SCENARIOS
# ═══════════════════════════════════════════════════════════

def run_test_scenarios():
    """
    Test the API with various patient scenarios.
    
    This function tests:
    1. Different age groups (young, middle-aged, elderly)
    2. Various comorbidities
    3. Different activity levels
    4. Malignant vs benign diagnoses
    5. Edge cases
    """
    logger.info("\\n" + "="*80)
    logger.info("TASK 5.6 - TESTING WITH VARIOUS PATIENT SCENARIOS")
    logger.info("="*80)
    
    # Test scenarios
    test_scenarios = [
        {
            'name': "Scenario 1: Young healthy patient with benign diagnosis",
            'diagnosis': 'Benign',
            'confidence': 0.92,
            'tissue_type': 'NORM',
            'patient_data': {
                'age': 30,
                'sex': 'Male',
                'comorbidities': [],
                'activity_level': 'active',
                'smoking_status': 'non-smoker',
                'family_history': False
            }
        },
        {
            'name': "Scenario 2: Elderly patient with diabetes and malignant diagnosis",
            'diagnosis': 'Malignant colorectal adenocarcinoma',
            'confidence': 0.89,
            'tissue_type': 'TUM',
            'patient_data': {
                'age': 75,
                'sex': 'Female',
                'comorbidities': ['diabetes', 'hypertension'],
                'activity_level': 'sedentary',
                'smoking_status': 'former_smoker',
                'family_history': True
            }
        },
        {
            'name': "Scenario 3: Middle-aged smoker with family history",
            'diagnosis': 'Adenomatous polyp',
            'confidence': 0.85,
            'tissue_type': 'STR',
            'patient_data': {
                'age': 50,
                'sex': 'Male',
                'comorbidities': [],
                'activity_level': 'moderate',
                'smoking_status': 'current_smoker',
                'family_history': True
            }
        },
        {
            'name': "Scenario 4: Active lifestyle, no risk factors",
            'diagnosis': 'Normal mucosa',
            'confidence': 0.95,
            'tissue_type': 'NORM',
            'patient_data': {
                'age': 45,
                'sex': 'Female',
                'comorbidities': [],
                'activity_level': 'active',
                'smoking_status': 'non-smoker',
                'family_history': False
            }
        },
        {
            'name': "Scenario 5: Edge case - low confidence malignant",
            'diagnosis': 'Suspected malignancy',
            'confidence': 0.72,
            'tissue_type': 'TUM',
            'patient_data': {
                'age': 65,
                'sex': 'Male',
                'comorbidities': ['hypertension'],
                'activity_level': 'moderate',
                'smoking_status': 'former_smoker',
                'family_history': False
            }
        }
    ]
    
    # Run each test scenario
    for idx, scenario in enumerate(test_scenarios, 1):
        logger.info(f"\\n{'-'*80}")
        logger.info(f"TEST {idx}: {scenario['name']}")
        logger.info(f"{'-'*80}")
        
        try:
            # Generate treatment plan
            treatment_plan = generate_template_treatment_plan(
                diagnosis=scenario['diagnosis'],
                confidence=scenario['confidence'],
                tissue_type=scenario['tissue_type'],
                patient_data=scenario['patient_data']
            )
            
            # Log results
            logger.info(f"✓ Treatment plan generated successfully")
            logger.info(f"  Patient: Age {scenario['patient_data']['age']}, "
                       f"{scenario['patient_data']['sex']}")
            logger.info(f"  Diagnosis: {scenario['diagnosis']} "
                       f"(Confidence: {scenario['confidence']:.1%})")
            logger.info(f"  Activity Level: {scenario['patient_data']['activity_level']}")
            logger.info(f"  Comorbidities: {scenario['patient_data']['comorbidities'] or 'None'}")
            logger.info(f"\\n  Treatment Plan Sections:")
            for key in treatment_plan.keys():
                logger.info(f"    - {key}")
            
            # Verify all required sections exist
            required_sections = [
                'medical_treatment', 'lifestyle_modifications',
                'diet_recommendations', 'exercise_guidance',
                'next_steps', 'follow_up', 'disclaimer'
            ]
            
            for section in required_sections:
                if section not in treatment_plan:
                    logger.error(f"  ✗ Missing section: {section}")
                else:
                    logger.info(f"  ✓ {section}: {len(treatment_plan[section])} characters")
            
        except Exception as e:
            logger.error(f"✗ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\\n{'='*80}")
    logger.info("✓ TASK 5.6 - All test scenarios completed")
    logger.info(f"{'='*80}\\n")


# ═══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Initialize components
    try:
        initialize_model()
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        logger.info("Continuing without model (API will return errors for diagnosis endpoints)")
    
    try:
        initialize_llm()
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        logger.info("Continuing with template-based treatment plans")
    
    # ═══════════════════════════════════════════════════════════
    # TASK 5.6 - RUN TEST SCENARIOS
    # ═══════════════════════════════════════════════════════════
    
    # Run test scenarios before starting server
    #logger.info("Running pre-launch tests...")
    #run_test_scenarios()
    
    # ═══════════════════════════════════════════════════════════
    
    # Run Flask server
    logger.info("\\nStarting Flask server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
