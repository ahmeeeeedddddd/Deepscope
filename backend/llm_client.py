# backend/llm_client.py

"""
LLM Client for Treatment Recommendation Generation
Uses Google Gemini API for personalized colorectal cancer treatment plans.
Tasks 5.3 and 5.4 implemented.
"""

from google import genai
from google.genai import types
import json
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client for generating personalized colorectal cancer treatment recommendations.
    Combines ViT diagnosis results with patient clinical data using Google Gemini.
    """
    
    def __init__(
        self, 
        api_key: str = "AIzaSyBORWZtS3vg_7nkRUfKp5knFfLLKqEN_PY",
        model_name: str = "gemini-3-flash-preview",
        mock_mode: bool = False,
        debug: bool = False
    ):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
            mock_mode: If True, use mock responses instead of real API calls
            debug: If True, print raw API responses
        """
        self.mock_mode = mock_mode
        self.model_name = model_name
        self.debug = debug
        
        if not mock_mode:
            try:
                # Initialize the client with new API
                self.client = genai.Client(api_key=api_key)
                
                logger.info(f"✓ Gemini client initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {str(e)}")
                logger.info("Falling back to mock mode")
                self.mock_mode = True
        else:
            logger.info("LLM client running in MOCK MODE")
    
    def generate_treatment_plan(
        self,
        diagnosis: str,
        confidence: float,
        tissue_type: str,
        patient_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate personalized treatment recommendations.
        
        Args:
            diagnosis: Disease classification (e.g., "Malignant", "Benign")
            confidence: Model confidence (0-1)
            tissue_type: Tissue classification (TUM, NORM, etc.)
            patient_data: Dictionary containing patient clinical information
                
        Returns:
            Dictionary with treatment plan sections
        """
        
        if self.mock_mode:
            return self._generate_mock_response(diagnosis, confidence, tissue_type, patient_data)
        
        try:
            prompt = self._build_prompt(diagnosis, confidence, tissue_type, patient_data)
            
            # Generate response using new API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Lower for consistent medical advice
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=4096,  # ✅ Increased to capture full response
                )
            )
            
            # Parse the response
            response_text = response.text
            
            if self.debug:
                print("\n" + "="*80)
                print("RAW GEMINI RESPONSE:")
                print("="*80)
                print(response_text)
                print("="*80 + "\n")
            
            treatment_plan = self._parse_response(response_text)
            
            logger.info("✓ Treatment plan generated successfully via Gemini")
            
            return treatment_plan
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            logger.info("Falling back to mock response")
            return self._generate_mock_response(diagnosis, confidence, tissue_type, patient_data)
    
    def _build_prompt(
        self,
        diagnosis: str,
        confidence: float,
        tissue_type: str,
        patient_data: Dict[str, Any]
    ) -> str:
        """
        Build the prompt for Gemini with diagnosis and clinical information.
        """
        
        # Map tissue types to full names
        tissue_names = {
            'TUM': 'Tumor tissue (malignant)',
            'NORM': 'Normal colon mucosa',
            'ADI': 'Adipose (fat) tissue',
            'BACK': 'Background (non-tissue)',
            'DEB': 'Debris',
            'LYM': 'Lymphocytes (immune cells)',
            'MUC': 'Mucus',
            'MUS': 'Smooth muscle',
            'STR': 'Stroma (connective tissue)'
        }
        
        tissue_full_name = tissue_names.get(tissue_type, tissue_type)
        
        # Format diagnosis information
        diagnosis_info = f"""
**AI VISION TRANSFORMER DIAGNOSIS:**
- Primary Classification: {diagnosis}
- Tissue Type Detected: {tissue_full_name}
- AI Confidence Score: {confidence:.2%}
- Analysis Method: Lunit DINO Vision Transformer (ViT) on H&E histopathology
"""
        
        # Format clinical data
        age = patient_data.get('age', 'Not provided')
        sex = patient_data.get('sex', 'Not provided')
        comorbidities = patient_data.get('comorbidities', [])
        activity_level = patient_data.get('activity_level', 'Not provided')
        smoking_status = patient_data.get('smoking_status', 'Not provided')
        family_history = patient_data.get('family_history', 'Not provided')
        additional_info = patient_data.get('additional_info', '')
        bmi = patient_data.get('bmi', 'Not provided')
        
        clinical_info = f"""
**PATIENT CLINICAL PROFILE:**
- Age: {age} years
- Sex: {sex}
- Body Mass Index: {bmi}
- Comorbidities: {', '.join(comorbidities) if comorbidities else 'None reported'}
- Physical Activity Level: {activity_level}
- Smoking Status: {smoking_status}
- Family History of Colorectal Cancer: {family_history}
"""
        if additional_info:
            clinical_info += f"\n**ADDITIONAL PATIENT INFORMATION:**\n{additional_info}\n"
        
        prompt = f"""You are an expert oncology medical advisor AI assistant. Based on the AI diagnostic results and patient clinical data provided below, generate a comprehensive, personalized treatment and care plan for colorectal cancer management.

{diagnosis_info}

{clinical_info}

Generate a detailed treatment plan with EXACTLY these six sections (use these exact headers):

**1. MEDICAL TREATMENT**
- Immediate clinical actions required
- Recommended diagnostic confirmations (colonoscopy, biopsy, imaging)
- Potential treatment pathways (surgery, chemotherapy, radiation, immunotherapy)
- Specialist referrals needed
- Expected timeline for interventions

**2. LIFESTYLE MODIFICATIONS**
- Specific lifestyle changes based on patient's current activity level
- Daily routine adjustments
- Sleep and stress management
- Environmental factors to consider

**3. DIET RECOMMENDATIONS**
- Detailed dietary guidelines for colorectal cancer prevention/management
- Foods to emphasize (high fiber, antioxidants)
- Foods to limit or avoid (red meat, processed foods)
- Meal planning suggestions
- Hydration requirements
- Specific adjustments based on patient's comorbidities

**4. EXERCISE GUIDANCE**
- Personalized exercise prescription based on age and current activity level
- Types of exercises (aerobic, strength, flexibility)
- Frequency, intensity, duration recommendations
- Safety precautions and contraindications
- Progressive exercise plan (beginner to advanced)

**5. NEXT STEPS**
- Immediate actions (within 1-2 weeks)
- Short-term follow-up (1-3 months)
- Long-term monitoring plan
- When to seek emergency care
- Questions to ask healthcare providers

**6. FOLLOW-UP SCHEDULE**
- Recommended consultation timeline
- Screening and surveillance schedule
- Monitoring parameters
- Long-term prognosis and management

**CRITICAL GUIDELINES:**
- Use clear, compassionate, patient-friendly language
- Be specific and actionable in all recommendations
- Consider the patient's age, sex, comorbidities, and lifestyle when making suggestions
- Address smoking cessation urgently if applicable
- Tailor exercise recommendations to current fitness level
- If diagnosis indicates malignancy, stress urgency while remaining compassionate
- If diagnosis is benign, focus on prevention and monitoring
- Keep each section concise but complete (aim for 3-5 key points per section)

**IMPORTANT DISCLAIMER:**
End with a clear statement that this is AI-generated guidance to support medical decision-making, NOT a replacement for professional medical diagnosis or treatment. Emphasize the need for consultation with qualified healthcare providers.

Generate the complete treatment plan now, ensuring ALL SIX SECTIONS are thoroughly addressed."""

        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse the Gemini response into structured sections.
        Improved version with better section detection.
        """
        sections = {
            'medical_treatment': '',
            'lifestyle_modifications': '',
            'diet_recommendations': '',
            'exercise_guidance': '',
            'next_steps': '',
            'follow_up': '',
            'disclaimer': ''
        }
        
        # Section mapping with multiple possible headers
        section_keywords = {
            'medical_treatment': ['1. MEDICAL TREATMENT', 'MEDICAL TREATMENT', '**1. MEDICAL'],
            'lifestyle_modifications': ['2. LIFESTYLE MODIFICATIONS', 'LIFESTYLE MODIFICATIONS', '**2. LIFESTYLE'],
            'diet_recommendations': ['3. DIET RECOMMENDATIONS', 'DIET RECOMMENDATIONS', '**3. DIET'],
            'exercise_guidance': ['4. EXERCISE GUIDANCE', 'EXERCISE GUIDANCE', '**4. EXERCISE'],
            'next_steps': ['5. NEXT STEPS', 'NEXT STEPS', '**5. NEXT'],
            'follow_up': ['6. FOLLOW-UP SCHEDULE', 'FOLLOW-UP SCHEDULE', '**6. FOLLOW', 'FOLLOW UP'],
            'disclaimer': ['IMPORTANT DISCLAIMER', 'DISCLAIMER', 'EXECUTIVE SUMMARY']
        }
        
        current_section = None
        current_content = []
        
        lines = response_text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            # Check if this line is a section header
            matched_section = None
            
            for section_name, keywords in section_keywords.items():
                for keyword in keywords:
                    if keyword in line_upper:
                        # Check if it looks like a header (starts with ** or ###, or is numbered)
                        if (line_stripped.startswith('**') or 
                            line_stripped.startswith('###') or
                            line_stripped.startswith('#') or
                            any(line_stripped.startswith(f'{i}.') for i in range(1, 10))):
                            matched_section = section_name
                            break
                if matched_section:
                    break
            
            if matched_section:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = matched_section
                current_content = []
            elif current_section:
                # Add line to current section (skip empty header lines)
                if line_stripped and not (line_stripped.startswith('**') and line_stripped.endswith('**') and len(line_stripped) < 100):
                    current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Fallback: if parsing failed completely
        if not any(sections.values()):
            sections['medical_treatment'] = response_text
        
        # Ensure disclaimer exists
        if not sections['disclaimer']:
            sections['disclaimer'] = (
                "⚠️ IMPORTANT DISCLAIMER: This treatment plan is AI-generated based on "
                "the diagnostic results and patient data provided. It is NOT a substitute "
                "for professional medical advice, diagnosis, or treatment. Always seek the "
                "advice of your physician or other qualified health provider with any questions "
                "regarding a medical condition."
            )
        
        return sections
    
    def _generate_mock_response(
        self,
        diagnosis: str,
        confidence: float,
        tissue_type: str,
        patient_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate mock response for testing without API calls.
        """
        
        is_malignant = 'malignant' in diagnosis.lower() or tissue_type == 'TUM'
        age = patient_data.get('age', 50)
        comorbidities = patient_data.get('comorbidities', [])
        activity_level = patient_data.get('activity_level', 'moderate')
        
        mock_plan = {
            'medical_treatment': f"""
[MOCK RESPONSE - Gemini API not called]

Based on the AI diagnosis of {diagnosis} (confidence: {confidence:.1%}):

{'⚠️ URGENT ACTION REQUIRED:' if is_malignant else 'Recommended Actions:'}
1. Schedule consultation with gastroenterologist {'IMMEDIATELY (within 1-2 weeks)' if is_malignant else 'within 4-6 weeks'}
2. Confirmatory diagnostic procedures:
   - Colonoscopy with biopsy
   - {'CT scan and staging workup' if is_malignant else 'Follow-up imaging if needed'}
3. {'Oncology referral for treatment planning' if is_malignant else 'Continue routine surveillance'}
4. Discuss treatment options with medical team

{'This appears to be a malignant finding requiring prompt medical attention.' if is_malignant else 'This appears to be a benign finding, but medical confirmation is essential.'}
""",
            
            'lifestyle_modifications': f"""
Personalized for: Age {age}, Activity Level: {activity_level}

1. Stress Management:
   - Practice relaxation techniques daily
   - Ensure 7-8 hours quality sleep
   - Consider counseling support

2. {'Gradual increase in physical activity' if activity_level == 'sedentary' else 'Maintain current activity level'}
3. Avoid prolonged sitting
4. Smoking cessation {'(CRITICAL)' if patient_data.get('smoking_status') in ['current_smoker', 'smoker'] else '(maintain non-smoking status)'}
5. Limit alcohol consumption
""",
            
            'diet_recommendations': f"""
Colorectal Cancer Prevention Diet:

INCREASE:
- Fiber: 25-35g daily (whole grains, legumes, vegetables)
- Cruciferous vegetables: broccoli, cauliflower, Brussels sprouts
- Leafy greens: spinach, kale, collards
- Colorful fruits and vegetables
- Water: 8-10 glasses daily

LIMIT/AVOID:
- Red meat: <500g per week
- Processed meats: minimize or eliminate
- Highly processed foods
- Excessive sugar and refined carbohydrates
- Alcohol: limit to 1 drink/day

{'SPECIAL CONSIDERATIONS for comorbidities: ' + ', '.join(comorbidities) if comorbidities else 'No dietary restrictions due to comorbidities'}
""",
            
            'exercise_guidance': f"""
Personalized Exercise Plan (Age {age}, Current Level: {activity_level}):

{'BEGINNER PLAN (Sedentary to Active):' if activity_level == 'sedentary' else 'ACTIVE MAINTENANCE PLAN:'}
- {'Week 1-2: 10-15 min walking daily' if activity_level == 'sedentary' else 'Continue 150+ minutes moderate activity/week'}
- {'Week 3-4: 20-30 min, 5 days/week' if activity_level == 'sedentary' else 'Mix of cardio and strength training'}
- {'Month 2+: 150 min moderate activity/week' if activity_level == 'sedentary' else 'Include flexibility and balance exercises'}
- Strength training: 2 sessions/week

Recommended Activities:
- Brisk walking, swimming, cycling
- Yoga or tai chi for flexibility
- Light resistance training

{'⚠️ Consult physician before starting new exercise program' if age >= 60 or comorbidities else 'Listen to your body and progress gradually'}
""",
            
            'next_steps': f"""
IMMEDIATE (1-2 weeks):
1. {'URGENT: Schedule oncology consultation' if is_malignant else 'Schedule gastroenterology appointment'}
2. Bring this AI report to your appointment
3. {'Request expedited diagnostic workup' if is_malignant else 'Discuss surveillance plan'}

SHORT-TERM (1-3 months):
4. Complete recommended diagnostic procedures
5. {'Begin treatment planning' if is_malignant else 'Establish monitoring schedule'}
6. Implement lifestyle and dietary changes

LONG-TERM:
7. Regular follow-up appointments
8. Ongoing surveillance colonoscopies
9. Maintain healthy lifestyle modifications
10. {'Consider second opinion from cancer center' if is_malignant else 'Annual wellness checks'}
""",
            
            'follow_up': f"""
{'URGENT FOLLOW-UP SCHEDULE (Malignant Finding):' if is_malignant else 'STANDARD FOLLOW-UP SCHEDULE:'}

- Initial consultation: {'Within 1-2 weeks (URGENT)' if is_malignant else 'Within 4-6 weeks'}
- Diagnostic completion: {'Within 2-3 weeks' if is_malignant else 'Within 2-3 months'}
- {'Treatment initiation: Within 3-4 weeks' if is_malignant else 'Follow-up colonoscopy: Per physician recommendation'}
- {'Post-treatment monitoring: Every 3-6 months for 5 years' if is_malignant else 'Annual wellness visits'}

SURVEILLANCE:
- {f'Colonoscopy: More frequent due to {"family history" if patient_data.get("family_history") else "diagnosis"}' if is_malignant or patient_data.get('family_history') else 'Colonoscopy: Every 10 years (average risk) or as recommended'}
- Blood work and imaging as directed
- Report any new symptoms immediately

WARNING SIGNS requiring immediate medical attention:
- Rectal bleeding or blood in stool
- Unexplained weight loss
- Severe abdominal pain
- Persistent change in bowel habits
""",
            
            'disclaimer': f"""
⚠️ IMPORTANT DISCLAIMER - MOCK MODE ACTIVE

This is a MOCK RESPONSE generated for testing purposes. In production, this would be generated by Google Gemini AI.

This treatment plan is AI-generated based on the diagnostic results (AI confidence: {confidence:.1%}) and patient data provided. 

CRITICAL REMINDERS:
- This is NOT a medical diagnosis
- This is NOT a substitute for professional medical care
- AI diagnostic confidence of {confidence:.1%} requires clinical validation
- Always consult qualified healthcare providers for medical decisions
- Never delay seeking medical care based on AI recommendations
- All treatment decisions should be made in consultation with your medical team

For medical emergencies, call emergency services immediately.
"""
        }
        
        logger.info("✓ Mock treatment plan generated (API not called)")
        return mock_plan


# ═══════════════════════════════════════════════════════════
# TESTING FUNCTION
# ═══════════════════════════════════════════════════════════

def test_llm_client():
    """
    Test the LLM client with sample scenarios.
    This implements Task 5.6 - Testing with various patient scenarios.
    """
    
    print("\n" + "="*80)
    print("TESTING LLM CLIENT - TASK 5.6")
    print("="*80)
    
    # Initialize client
    client = LLMClient(
        api_key="AIzaSyBORWZtS3vg_7nkRUfKp5knFfLLKqEN_PY",  # Replace with your actual key
        model_name="gemini-3-flash-preview",
        mock_mode=False,  # Set to True to test without API calls
        debug=False  # Set to True to see raw responses
    )
    
    # Test scenarios
    test_scenarios = [
        {
            'name': "Young healthy patient - Benign",
            'diagnosis': 'Benign tissue',
            'confidence': 0.92,
            'tissue_type': 'NORM',
            'patient_data': {
                'age': 30,
                'sex': 'Male',
                'comorbidities': [],
                'activity_level': 'active',
                'smoking_status': 'non-smoker',
                'family_history': False,
                'bmi': 22.5
            }
        },
        {
            'name': "Elderly patient - Malignant with comorbidities",
            'diagnosis': 'Malignant colorectal adenocarcinoma',
            'confidence': 0.89,
            'tissue_type': 'TUM',
            'patient_data': {
                'age': 72,
                'sex': 'Female',
                'comorbidities': ['Type 2 Diabetes', 'Hypertension', 'Osteoarthritis'],
                'activity_level': 'sedentary',
                'smoking_status': 'former_smoker',
                'family_history': True,
                'bmi': 29.3
            }
        }
    ]
    
    # Run tests
    for idx, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'-'*80}")
        print(f"TEST SCENARIO {idx}: {scenario['name']}")
        print(f"{'-'*80}")
        print(f"Diagnosis: {scenario['diagnosis']}")
        print(f"Confidence: {scenario['confidence']:.1%}")
        print(f"Patient: Age {scenario['patient_data']['age']}, {scenario['patient_data']['sex']}")
        print(f"Activity: {scenario['patient_data']['activity_level']}")
        print(f"Comorbidities: {scenario['patient_data']['comorbidities'] or 'None'}")
        print(f"\nGenerating treatment plan...")
        
        try:
            result = client.generate_treatment_plan(
                diagnosis=scenario['diagnosis'],
                confidence=scenario['confidence'],
                tissue_type=scenario['tissue_type'],
                patient_data=scenario['patient_data']
            )
            
            print(f"\n✓ Treatment plan generated successfully!")
            print(f"\nSections generated:")
            for section_name, content in result.items():
                content_preview = content[:100].replace('\n', ' ') + '...' if len(content) > 100 else content
                print(f"  - {section_name}: {len(content)} chars | {content_preview}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✓ Testing complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run tests
    test_llm_client()