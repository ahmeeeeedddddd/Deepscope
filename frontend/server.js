import express from 'express';
import multer from 'multer';
import axios from 'axios';
import path from 'path';
import fs from 'fs';
import FormData from 'form-data';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ========================================
// APP CONFIGURATION
// ========================================

const app = express();
const PORT = 3000;
const BACKEND_URL = 'http://localhost:5000';

// In-memory storage for latest diagnosis (replace with database in production)
let latestDiagnosis = null;

// Set view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, '..', 'backend', 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

// ========================================
// FILE UPLOAD CONFIGURATION
// ========================================

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadsDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, 'histology-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const fileFilter = (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|tif|tiff/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);

    if (mimetype && extname) {
        return cb(null, true);
    } else {
        cb(new Error('Only histopathology images (.jpg, .png, .tif) are allowed!'));
    }
};

const upload = multer({
    storage: storage,
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB max file size
    },
    fileFilter: fileFilter
});

// ========================================
// UTILITY FUNCTIONS
// ========================================

/**
 * Send image to backend for classification
 */
async function classifyImage(filePath) {
    try {
        const formData = new FormData();
        formData.append('image', fs.createReadStream(filePath));

        const response = await axios.post(`${BACKEND_URL}/api/classify`, formData, {
            headers: formData.getHeaders(),
            timeout: 30000
        });

        return response.data;
    } catch (error) {
        console.error('Classification error:', error.message);
        throw new Error('Failed to classify image. Backend may be unavailable.');
    }
}

/**
 * Send diagnosis + clinical data to backend for treatment recommendations
 */
async function getTreatmentRecommendations(diagnosisData, clinicalData) {
    try {
        const response = await axios.post(`${BACKEND_URL}/api/treatment`, {
            diagnosis: diagnosisData.diagnosis,        // ← Extract just the diagnosis string
            confidence: diagnosisData.confidence,       // ← Extract confidence
            tissue_type: diagnosisData.tissue_type,     // ← Extract tissue_type
            patient_data: clinicalData                  // ← Changed from clinical_info to patient_data
        }, {
            timeout: 180000  // 3 minutes timeout for LLM processing
        });

        return response.data;
    } catch (error) {
        console.error('Treatment recommendation error:', error.message);
        if (error.response) {
            console.error('Backend response:', error.response.data);
        }
        throw new Error('Failed to generate treatment recommendations.');
    }
}

/**
 * Parse clinical data from form
 */
function parseClinicalData(body) {
    const clinicalData = {
        age: parseInt(body.age),
        sex: body.sex
    };

    // Add optional fields if provided
    if (body.comorbidities && body.comorbidities.trim()) {
        clinicalData.comorbidities = body.comorbidities
            .split(',')
            .map(c => c.trim())
            .filter(c => c);
    }

    if (body.activity_level) {
        clinicalData.activity_level = body.activity_level;
    }

    if (body.smoking_status) {
        clinicalData.smoking_status = body.smoking_status;
    }

    if (body.family_history) {
        clinicalData.family_history = body.family_history === 'true';
    }

    if (body.bmi) {
        clinicalData.bmi = parseFloat(body.bmi);
    }

    // NEW: Handle additional information
    if (body.additional_info && body.additional_info.trim()) {
        clinicalData.additional_info = body.additional_info.trim();
    }

    return clinicalData;
}

/**
 * Format treatment plan sections into a single readable string
 */
function formatTreatmentPlan(treatmentPlan) {
    if (!treatmentPlan) return 'No treatment plan available';
    
    const sections = [];
    
    if (treatmentPlan.medical_treatment) {
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push('MEDICAL TREATMENT');
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push(treatmentPlan.medical_treatment);
        sections.push('');
    }
    
    if (treatmentPlan.lifestyle_modifications) {
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push('LIFESTYLE MODIFICATIONS');
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push(treatmentPlan.lifestyle_modifications);
        sections.push('');
    }
    
    if (treatmentPlan.diet_recommendations) {
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push('DIET RECOMMENDATIONS');
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push(treatmentPlan.diet_recommendations);
        sections.push('');
    }
    
    if (treatmentPlan.exercise_guidance) {
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push('EXERCISE GUIDANCE');
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push(treatmentPlan.exercise_guidance);
        sections.push('');
    }
    
    if (treatmentPlan.next_steps) {
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push('NEXT STEPS');
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push(treatmentPlan.next_steps);
        sections.push('');
    }
    
    if (treatmentPlan.follow_up) {
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push('FOLLOW-UP SCHEDULE');
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push(treatmentPlan.follow_up);
        sections.push('');
    }
    
    if (treatmentPlan.disclaimer) {
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push('DISCLAIMER');
        sections.push('═══════════════════════════════════════════════════════════');
        sections.push(treatmentPlan.disclaimer);
    }
    
    return sections.join('\n');
}

/**
 * Clean up old uploaded files (older than 1 hour)
 */
function cleanupOldFiles() {
    const files = fs.readdirSync(uploadsDir);
    const now = Date.now();
    const oneHour = 60 * 60 * 1000;

    files.forEach(file => {
        const filePath = path.join(uploadsDir, file);
        const stats = fs.statSync(filePath);
        
        if (now - stats.mtimeMs > oneHour) {
            fs.unlinkSync(filePath);
            console.log(`Cleaned up old file: ${file}`);
        }
    });
}

// Run cleanup every 30 minutes
setInterval(cleanupOldFiles, 30 * 60 * 1000);

// ========================================
// ROUTES
// ========================================

/**
 * GET / - Home page with combined upload and clinical form
 */
app.get('/', (req, res) => {
    res.render('index', {
        title: 'New Diagnosis - DeepScope',
        error: null
    });
});

/**
 * GET /dashboard - Dashboard page with latest diagnosis
 */
app.get('/dashboard', (req, res) => {
    res.render('dashboard', {
        title: 'Dashboard - DeepScope',
        diagnosis: latestDiagnosis ? latestDiagnosis.diagnosis : null,
        treatment: latestDiagnosis ? latestDiagnosis.treatment : null,
        patient: latestDiagnosis ? latestDiagnosis.patient : {}
    });
});

/**
 * GET /history - History page (placeholder)
 */
app.get('/history', (req, res) => {
    res.send('<h1>History Page - Coming Soon</h1><p><a href="/">Back to Home</a></p>');
});

/**
 * POST /upload - Handle combined image upload, classification, and treatment generation
 */
app.post('/upload', upload.single('histology_image'), async (req, res) => {
    try {
        // Check if file was uploaded
        if (!req.file) {
            return res.status(400).render('index', {
                title: 'New Diagnosis - DeepScope',
                error: 'Please upload a histopathology image'
            });
        }

        console.log(`Image uploaded: ${req.file.filename}`);
        console.log(`File size: ${(req.file.size / 1024).toFixed(2)} KB`);

        // Parse clinical data from form
        const clinicalData = parseClinicalData(req.body);
        console.log('Clinical data:', clinicalData);

        // Step 1: Classify the image
        console.log('Sending image to backend for classification...');
        const classificationResult = await classifyImage(req.file.path);
        console.log('Classification successful:', classificationResult);

        // Step 2: Generate treatment recommendations
        console.log('Generating treatment recommendations...');
        const treatmentResult = await getTreatmentRecommendations(
            classificationResult,
            clinicalData
        );
        console.log('Treatment recommendations generated successfully');

        // Format treatment plan as a single string for display
        const formattedTreatment = formatTreatmentPlan(treatmentResult.treatment_plan);

        // Store latest diagnosis in memory for dashboard
        latestDiagnosis = {
            diagnosis: {
                predicted_class: classificationResult.diagnosis || classificationResult.tissue_description,
                tissue_type: classificationResult.tissue_type,
                confidence: classificationResult.confidence,
                is_malignant: classificationResult.is_malignant,
                clinical_group: classificationResult.clinical_group
            },
            treatment: treatmentResult.treatment_plan,
            patient: {
                age: clinicalData.age,
                sex: clinicalData.sex,
                comorbidities: clinicalData.comorbidities || [],
                activity_level: clinicalData.activity_level || 'Not specified',
                smoking_status: clinicalData.smoking_status || 'Not specified',
                family_history: clinicalData.family_history || false,
                bmi: clinicalData.bmi || 'Not specified',
                additional_info: clinicalData.additional_info || 'None'
            }
        };

        // Render results page with both diagnosis and treatment
        
        // Extract heatmap filename if available
        let heatmapFilename = null;
        if (classificationResult.heatmap_path) {
            heatmapFilename = path.basename(classificationResult.heatmap_path);
            console.log(`Grad-CAM heatmap available: ${heatmapFilename}`);
        }

        // Render results page with both diagnosis and treatment
        res.render('results', {
            title: 'Diagnosis Analysis',
            image_filename: classificationResult.preview_image || req.file.filename,
            heatmap_filename: heatmapFilename,
            segmentation_filename: classificationResult.segmentation_filename || null, 
            diagnosis: classificationResult,
            treatment: {
                recommendation: formattedTreatment,
                raw: treatmentResult.treatment_plan
            },
            patient: latestDiagnosis.patient
        });

    } catch (error) {
        console.error('Upload error:', error);
        
        // Clean up uploaded file if processing failed
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }

        res.status(500).render('index', {
            title: 'New Diagnosis - DeepScope',
            error: error.message
        });
    }
});

/**
 * POST /treatment - Alternative endpoint for separate treatment generation
 */
app.post('/treatment', express.json(), async (req, res) => {
    try {
        const { diagnosis, clinical_info } = req.body;

        if (!diagnosis || !clinical_info) {
            return res.status(400).json({
                error: 'Missing diagnosis or clinical information'
            });
        }

        console.log('Generating treatment recommendations...');
        const treatment = await getTreatmentRecommendations(diagnosis, clinical_info);

        res.json({
            success: true,
            treatment: treatment
        });

    } catch (error) {
        console.error('Treatment generation error:', error);
        res.status(500).json({
            error: error.message
        });
    }
});

/**
 * GET /api/health - Health check endpoint
 */
app.get('/api/health', async (req, res) => {
    try {
        const backendResponse = await axios.get(`${BACKEND_URL}/api/health`, {
            timeout: 5000
        });

        res.json({
            status: 'healthy',
            frontend: 'running',
            backend: backendResponse.data.status || 'running',
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(503).json({
            status: 'unhealthy',
            frontend: 'running',
            backend: 'unreachable',
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

/**
 * GET /uploaded/:filename - Serve uploaded images
 */
app.get('/uploaded/:filename', (req, res) => {
    const filePath = path.join(uploadsDir, req.params.filename);
    
    if (!fs.existsSync(filePath)) {
        return res.status(404).send('Image not found');
    }

    res.sendFile(filePath);
});

app.get('/heatmap/:filename', (req, res) => {
    const filePath = path.join(uploadsDir, req.params.filename);
    
    if (!fs.existsSync(filePath)) {
        return res.status(404).send('Heatmap not found');
    }

    res.sendFile(filePath);
});


app.get('/segmentation/:filename', (req, res) => {
    const filePath = path.join(uploadsDir, req.params.filename);
    
    if (!fs.existsSync(filePath)) {
        return res.status(404).send('Segmentation not found');
    }

    res.sendFile(filePath);
});

// ========================================
// ERROR HANDLING
// ========================================

/**
 * Handle multer errors
 */
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).render('index', {
                title: 'New Diagnosis - DeepScope',
                error: 'File too large. Maximum size is 10MB.'
            });
        }
        return res.status(400).render('index', {
            title: 'New Diagnosis - DeepScope',
            error: `Upload error: ${error.message}`
        });
    }
    
    if (error) {
        return res.status(400).render('index', {
            title: 'New Diagnosis - DeepScope',
            error: error.message
        });
    }
    
    next();
});

/**
 * 404 handler
 */
app.use((req, res) => {
    res.status(404).send('404 - Page Not Found');
});

/**
 * General error handler
 */
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).send('500 - Internal Server Error');
});

// ========================================
// SERVER STARTUP
// ========================================

app.listen(PORT, () => {
    console.log('═══════════════════════════════════════════════════════════');
    console.log('  DEEPSCOPE - COLORECTAL CANCER DIAGNOSIS SYSTEM');
    console.log('═══════════════════════════════════════════════════════════');
    console.log(`  Frontend URL:  http://localhost:${PORT}`);
    console.log(`  Backend URL:   ${BACKEND_URL}`);
    console.log(`  Environment:   development`);
    console.log('═══════════════════════════════════════════════════════════');
    console.log('  Available Routes:');
    console.log('    GET  /              - New diagnosis (upload + clinical)');
    console.log('    GET  /dashboard     - Dashboard view');
    console.log('    POST /upload        - Process diagnosis');
    console.log('    POST /treatment     - Generate treatment (API)');
    console.log('    GET  /api/health    - Health check');
    console.log('═══════════════════════════════════════════════════════════');
    console.log('  Server is running and ready to accept requests');
    console.log('  Press Ctrl+C to stop the server');
    console.log('═══════════════════════════════════════════════════════════\n');
});

// ========================================
// GRACEFUL SHUTDOWN
// ========================================

process.on('SIGTERM', () => {
    console.log('SIGTERM signal received: closing HTTP server');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('\nSIGINT signal received: closing HTTP server');
    process.exit(0);
});

// ========================================
// EXPORT FOR TESTING
// ========================================

export default app;