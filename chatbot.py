# Complete Enhanced Mental Health Chatbot Application - Enhanced with Ollama Llama 3
# This is the COMPLETE version maintaining ALL original functionality

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import logging
import traceback
from datetime import datetime, timedelta
import json
import time
import threading
from typing import Dict, Any, Optional
import numpy as np
import atexit
# (In chatbot.py, after the import statements)
import re


# In chatbot.py, replace the clean_ai_response function

def clean_ai_response(text: str) -> str:
    """
    Cleans instructional artifacts from the AI's response while preserving Markdown.
    """
    if not isinstance(text, str):
        return text

    # Replace escaped newlines with actual newlines
    cleaned_text = text.replace('\\n', '\n').strip()

    # Remove role markers first
    cleaned_text = cleaned_text.replace("SAHARA:", "").strip()

    # (The rest of the function for filtering instructions can remain)
    lines = cleaned_text.splitlines()
    instructional_phrases = [
        "your task is to", "your response must be only", "based on all the context",
        "generate only the direct, empathetic response", "do not repeat, reference, or explain"
    ]

    filtered_lines = [
        line for line in lines 
        if not any(phrase.lower() in line.lower() for phrase in instructional_phrases)
    ]

    return "\n".join(filtered_lines).strip()
# Import enhanced database models
from enhanced_database_models import (
    db, User, ConversationTurn, MethodFeedback,
    CounselorInteraction, CrisisEvent, SystemMetrics,
    UserSession, init_database, get_user_statistics
)

# Import enhanced AI components with Ollama integration
from nlu_processor import ProgressiveNLUProcessor
from ko import ProgressiveResponseGenerator
from conversation_memory import ProgressiveConversationMemory
from optimized_crisis_detector import OptimizedCrisisDetector

# Import Ollama integration
from api_ollama_integration import ollama_llama3

# Configure comprehensive logging with multiple handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handlers
file_handler = logging.FileHandler('chatbot.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

error_handler = logging.FileHandler('system_errors.log', mode='a', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Console handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(error_handler)
logger.addHandler(stream_handler)

# Get module logger
logger = logging.getLogger(__name__)

# Initialize Flask application with enhanced configuration
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/v1/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
    }
})

# Enhanced security configuration
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24)
)

# Enhanced database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
models_path = os.path.join(basedir, 'models')
logs_path = os.path.join(basedir, 'logs')

# Ensure all directories exist
for path in [instance_path, models_path, logs_path]:
    os.makedirs(path, exist_ok=True)

# --- THIS IS THE CORRECTED CODE BLOCK ---

# First, determine the correct database URI
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith('postgres://'):
    # This path is for Render (PostgreSQL)
    db_uri = database_url.replace('postgres://', 'postgresql://', 1)
else:
    # This path is for your local computer (SQLite)
    db_uri = f'sqlite:///{os.path.join(instance_path, "enhanced_chatbot.db")}'

# Now, update the app configuration
app.config.update({
    'SQLALCHEMY_DATABASE_URI': db_uri,
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'SQLALCHEMY_ENGINE_OPTIONS': {
        'pool_timeout': 30,
        'pool_recycle': 300,
        'pool_pre_ping': True,
        'echo': False
    }
})

# Initialize database
db.init_app(app)

# Global system state tracking
system_state = {
    'startup_time': datetime.now(),
    'total_requests': 0,
    'successful_responses': 0,
    'error_count': 0,
    'crisis_interventions': 0,
    'methods_suggested': 0,
    'professional_referrals': 0,
    'llama_responses': 0,
    'fallback_responses': 0
}

# Thread lock for system state updates
state_lock = threading.Lock()

# Initialize enhanced AI components with comprehensive error handling
def initialize_ai_components():
    """Initialize all AI components with Ollama integration and proper error handling"""
    global nlu_processor, response_generator, conversation_memory, crisis_detector
    global system_status
    
    logger.info("🚀 Initializing Enhanced Mental Health Chatbot System with Ollama Llama 3...")
    
    # Model file paths
    nlu_model_path = os.path.join(models_path, 'progressive_nlu_model.pkl')
    memory_model_path = os.path.join(models_path, 'progressive_memory.pkl')
    crisis_model_path = os.path.join(models_path, 'optimized_crisis_detector.pkl')
    
    system_status = {
        'nlu_processor': False,
        'response_generator': False,
        'conversation_memory': False,
        'crisis_detector': False,
        'database': False,
        'ollama_llama3': False
    }
    
    try:
        # Check Ollama Llama 3 availability
        logger.info("🦙 Checking Ollama Llama 3 availability...")
        system_status['ollama_llama3'] = ollama_llama3.is_available
        
        if ollama_llama3.is_available:
            logger.info("✅ Ollama Llama 3 is available and ready for AI-enhanced responses")
        else:
            logger.info("⚠️ Ollama Llama 3 not available - using rule-based responses with fallback")
        
        # Initialize NLU Processor
        logger.info("🧠 Initializing Progressive NLU Processor with Ollama integration...")
        nlu_processor = ProgressiveNLUProcessor(model_path=nlu_model_path)
        system_status['nlu_processor'] = True
        logger.info("✅ NLU Processor initialized successfully")
        
        # Initialize Response Generator
        logger.info("💬 Initializing Progressive Response Generator with Ollama integration...")
        response_generator = ProgressiveResponseGenerator()
        system_status['response_generator'] = True
        logger.info("✅ Response Generator initialized successfully")
        
        # Initialize Conversation Memory
        logger.info("🧠 Initializing Progressive Conversation Memory...")
        conversation_memory = ProgressiveConversationMemory(save_path=memory_model_path)
        system_status['conversation_memory'] = True
        logger.info("✅ Conversation Memory initialized successfully")
        
        # Initialize Crisis Detector
        logger.info("🆘 Initializing Optimized Crisis Detector...")
        crisis_detector = OptimizedCrisisDetector(model_path=crisis_model_path, load_semantic_model=False)
        system_status['crisis_detector'] = True
        logger.info("✅ Crisis Detector initialized successfully")
        
        logger.info("✅ All enhanced AI components initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Critical error initializing AI components: {e}")
        logger.error(traceback.format_exc())
        
        # Initialize minimal fallback components
        try:
            logger.info("🔄 Attempting to initialize fallback components...")
            nlu_processor = ProgressiveNLUProcessor()
            response_generator = ProgressiveResponseGenerator()
            conversation_memory = ProgressiveConversationMemory()
            crisis_detector = OptimizedCrisisDetector()
            logger.info("⚠️ Fallback components initialized (limited functionality)")
            return False
        except Exception as fallback_error:
            logger.error(f"❌ Failed to initialize even fallback components: {fallback_error}")
            nlu_processor = None
            response_generator = None
            conversation_memory = None
            crisis_detector = None
            return False

# Initialize system
ai_initialized = initialize_ai_components()

# Database initialization with app context
with app.app_context():
    try:
        init_database(app)
        system_status['database'] = True
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        system_status['database'] = False

# Utility functions for system management
def update_system_state(operation: str, success: bool = True, **kwargs):
    """Thread-safe system state updates"""
    with state_lock:
        system_state['total_requests'] += 1
        if success:
            system_state['successful_responses'] += 1
        else:
            system_state['error_count'] += 1
        
        # Update specific counters
        for key, value in kwargs.items():
            if key in system_state:
                system_state[key] += value

def save_all_models():
    """Save all AI models with comprehensive error handling"""
    try:
        if nlu_processor and system_status['nlu_processor']:
            nlu_processor.save_nlu_model(os.path.join(models_path, 'progressive_nlu_model.pkl'))
            logger.info("✅ NLU model saved")
        
        if conversation_memory and system_status['conversation_memory']:
            conversation_memory.save_memory()
            logger.info("✅ Conversation memory saved")
        
        if crisis_detector and system_status['crisis_detector']:
            crisis_detector.save_detector(os.path.join(models_path, 'optimized_crisis_detector.pkl'))
            logger.info("✅ Crisis detector saved")
        
        logger.info("✅ All models saved successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving models: {e}")
        return False

def track_system_metrics():
    """Track and update system-wide metrics"""
    try:
        today = datetime.now().date()
        
        # Check if metrics already exist for today
        existing_metrics = SystemMetrics.query.filter_by(metrics_date=today).first()
        
        if not existing_metrics:
            # Calculate metrics for today
            total_users = User.query.filter_by(is_active=True).count()
            new_users = User.query.filter(
                User.created_at >= datetime.combine(today, datetime.min.time())
            ).count()
            
            total_conversations = ConversationTurn.query.filter(
                ConversationTurn.timestamp >= datetime.combine(today, datetime.min.time())
            ).count()
            
            crisis_events = CrisisEvent.query.filter(
                CrisisEvent.crisis_detected_date >= datetime.combine(today, datetime.min.time())
            ).count()
            
            methods_suggested = ConversationTurn.query.filter(
                ConversationTurn.timestamp >= datetime.combine(today, datetime.min.time()),
                ConversationTurn.method_suggested.isnot(None)
            ).count()
            
            counselor_referrals = CounselorInteraction.query.filter(
                CounselorInteraction.referral_date >= datetime.combine(today, datetime.min.time())
            ).count()
            
            # Calculate method success rate
            method_feedback = MethodFeedback.query.filter(
                MethodFeedback.feedback_date >= datetime.combine(today, datetime.min.time())
            ).all()
            
            if method_feedback:
                effective_methods = sum(1 for feedback in method_feedback 
                                     if feedback.effectiveness_rating == 'effective')
                method_success_rate = effective_methods / len(method_feedback)
            else:
                method_success_rate = 0.0
            
            # Create metrics record
            metrics = SystemMetrics(
                metrics_date=today,
                total_active_users=total_users,
                new_users_registered=new_users,
                total_conversations=total_conversations,
                crisis_events_detected=crisis_events,
                methods_suggested_total=methods_suggested,
                methods_marked_effective=sum(1 for feedback in method_feedback 
                                           if feedback.effectiveness_rating == 'effective'),
                overall_method_success_rate=method_success_rate,
                counselor_referrals_made=counselor_referrals
            )
            
            db.session.add(metrics)
            db.session.commit()
            logger.info(f"✅ System metrics updated for {today}")
            
    except Exception as e:
        logger.error(f"❌ Error tracking system metrics: {e}")

def get_current_user():
    """Security helper to get current authenticated user"""
    user_id = session.get('user_id')
    if user_id:
        try:
            return User.query.get(user_id)
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            return None
    return None

def create_user_session(user: User, request_info: dict):
    """Create and track user session"""
    try:
        user_session = UserSession(
            user_id=user.id,
            ip_address=request_info.get('remote_addr', '')[:45],
            user_agent=request_info.get('user_agent', '')[:500],
            device_type=determine_device_type(request_info.get('user_agent', ''))
        )
        
        db.session.add(user_session)
        db.session.commit()
        
        # Store session ID for later reference
        session['session_record_id'] = user_session.id
        
    except Exception as e:
        logger.error(f"Error creating user session: {e}")

def determine_device_type(user_agent: str) -> str:
    """Determine device type from user agent"""
    user_agent = user_agent.lower()
    if any(mobile in user_agent for mobile in ['mobile', 'android', 'iphone']):
        return 'mobile'
    elif 'tablet' in user_agent or 'ipad' in user_agent:
        return 'tablet'
    else:
        return 'desktop'

def end_user_session():
    """End current user session"""
    try:
        session_id = session.get('session_record_id')
        if session_id:
            user_session = UserSession.query.get(session_id)
            if user_session:
                user_session.end_session()
                db.session.commit()
    except Exception as e:
        logger.error(f"Error ending user session: {e}")

def convert_numpy_types(obj):
    """Recursively converts numpy types to native Python types in a dictionary."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Enhanced API Routes with comprehensive functionality
@app.route("/v1/register", methods=["POST"])
def register():
    """Enhanced user registration with comprehensive validation and tracking"""
    try:
        update_system_state('register')
        data = request.get_json()
        
        # Enhanced input validation
        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided."
            }), 400
        
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        full_name = data.get("fullName", "").strip()
        student_id = data.get("studentId", "").strip()
        college = data.get("college", "").strip()
        
        # Comprehensive validation
        validation_errors = []
        
        if not email:
            validation_errors.append("Email is required")
        elif "@" not in email or "." not in email:
            validation_errors.append("Please provide a valid email address")
        
        if not password:
            validation_errors.append("Password is required")
        elif len(password) < 8:
            validation_errors.append("Password must be at least 8 characters long")
        
        if not full_name:
            validation_errors.append("Full name is required")
        elif len(full_name) < 2:
            validation_errors.append("Please provide your full name")
        
        if validation_errors:
            return jsonify({
                "success": False,
                "message": "Please correct the following errors:",
                "errors": validation_errors
            }), 400
        
        # Check if user already exists
        existing_user = User.query.filter(
            (User.email == email) | 
            (User.student_id == student_id if student_id else False)
        ).first()
        
        if existing_user:
            return jsonify({
                "success": False,
                "message": "An account with this email or student ID already exists."
            }), 409
        
        # Generate student ID if not provided
        if not student_id:
            last_user = User.query.order_by(User.id.desc()).first()
            if last_user:
                try:
                    last_id_num = int(last_user.student_id.replace("STU", ""))
                    new_id_num = last_id_num + 1
                except (ValueError, AttributeError):
                    new_id_num = last_user.id + 1
            else:
                new_id_num = 1
            student_id = "STU" + str(new_id_num).zfill(6)
        
        # Create new user with enhanced profile
        new_user = User(
            student_id=student_id,
            email=email,
            full_name=full_name,
            college=college
        )
        new_user.set_password(password)
        
        # Set initial user state
        new_user.current_conversation_stage = 'initial_contact'
        new_user.current_risk_level = 'low'
        new_user.primary_concerns = '[]'
        
        db.session.add(new_user)
        db.session.commit()
        
        # Create user session
        create_user_session(new_user, {
            'remote_addr': request.environ.get('REMOTE_ADDR'),
            'user_agent': request.environ.get('HTTP_USER_AGENT')
        })
        
        # Initialize user in conversation memory
        if conversation_memory:
            conversation_memory.create_or_get_user(student_id)
        
        logger.info(f"✅ New user registered: {student_id} ({email})")
        
        return jsonify({
            "success": True,
            "studentId": student_id,
            "message": f"Welcome {full_name}! Your account has been created successfully.",
            "user": {
                "studentId": student_id,
                "fullName": full_name,
                "email": email,
                "memberSince": new_user.created_at.isoformat()
            }
        }), 201
        
    except Exception as e:
        logger.error(f"❌ Registration error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('register', success=False)
        
        return jsonify({
            "success": False,
            "message": "Registration failed due to a server error. Please try again."
        }), 500

@app.route("/v1/login", methods=["POST"])
def login():
    """Enhanced user login with session management and security tracking"""
    try:
        update_system_state('login')
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "No login data provided."
            }), 400
        
        student_id = data.get("studentId", "").strip()
        password = data.get("password", "")
        
        if not student_id or not password:
            return jsonify({
                "success": False,
                "message": "Student ID and password are required."
            }), 400
        
        # Find and authenticate user
        user = User.query.filter_by(student_id=student_id, is_active=True).first()
        
        if user and user.check_password(password):
            # Update user login tracking
            user.update_last_login()
            db.session.commit()
            
            # Set session with enhanced security
            session.permanent = True
            session['user_id'] = user.id
            session['student_id'] = user.student_id
            session['login_time'] = datetime.now().isoformat()
            
            # Create user session tracking
            create_user_session(user, {
                'remote_addr': request.environ.get('REMOTE_ADDR'),
                'user_agent': request.environ.get('HTTP_USER_AGENT')
            })
            
            # Get user statistics
            user_stats = get_user_statistics(user.id) if user.id else {}
            
            logger.info(f"✅ User logged in: {student_id}")
            
            response_data = {
                "success": True,
                "message": f"Welcome back, {user.full_name}!",
                "user": {
                    "studentId": user.student_id,
                    "fullName": user.full_name,
                    "email": user.email,
                    "riskLevel": user.current_risk_level,
                    "conversationStage": user.current_conversation_stage,
                    "totalConversations": user.total_conversations,
                    "lastLogin": user.last_login.isoformat() if user.last_login else None,
                    "memberSince": user.created_at.isoformat(),
                    "methodSuccessRate": user.calculate_method_success_rate(),
                    "improvementTrend": user.improvement_trend
                }
            }
            
            # Add user statistics if available
            if user_stats.get('user_info'):
                response_data["statistics"] = user_stats
            
            return jsonify(response_data)
            
        else:
            logger.warning(f"❌ Failed login attempt: {student_id}")
            update_system_state('login', success=False)
            
            return jsonify({
                "success": False,
                "message": "Invalid Student ID or password. Please check your credentials and try again."
            }), 401
            
    except Exception as e:
        logger.error(f"❌ Login error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('login', success=False)
        
        return jsonify({
            "success": False,
            "message": "Login failed due to a server error. Please try again."
        }), 500

@app.route("/v1/logout", methods=["POST"])
def logout():
    """Enhanced user logout with session cleanup"""
    try:
        # End user session tracking
        end_user_session()
        
        # Clear session
        session.clear()
        
        logger.info("✅ User logged out successfully")
        
        return jsonify({
            "success": True,
            "message": "You have been logged out successfully."
        })
        
    except Exception as e:
        logger.error(f"❌ Logout error: {e}")
        return jsonify({
            "success": True,  # Always succeed logout for security
            "message": "Logged out successfully."
        })

@app.route("/v1/predict", methods=["POST"])
def predict():
    """
    Enhanced conversation prediction with Ollama Llama 3 integration
    Main endpoint that handles all user interactions with AI enhancement
    """
    # Check if AI components are available
    if not all([nlu_processor, response_generator, conversation_memory, crisis_detector]):
        return jsonify({
            "response": "I'm experiencing technical difficulties right now. Please try again in a moment, or contact emergency services at 911 if this is urgent.",
            "error": True,
            "system_status": system_status,
            "fallback_resources": {
                "crisis_hotline": "988",
                "emergency": "911",
                "crisis_text": "Text HOME to 741741"
            }
        }), 503
    
    data = {}
    try:
        start_time = time.time()
        update_system_state('predict')
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided in request."}), 400
        
        user_message = data.get("message", "").strip()
        user_id_str = data.get("userId", "").strip()
        language_code = data.get("language", "en").strip()
        context = data.get("context", {})
        
        if not user_id_str or not user_message:
            return jsonify({"error": "Both userId and message are required."}), 400
        
        if len(user_message) > 2000:
            return jsonify({"error": "Message too long. Please keep messages under 2000 characters."}), 400
        
        # Find user in database
        with app.app_context():
            current_user = User.query.filter_by(student_id=user_id_str, is_active=True).first()
            if not current_user:
                return jsonify({
                    "error": "User not found. Please log in again.",
                    "login_required": True
                }), 401
        
        logger.info(f"🔄 Processing message from {user_id_str}: '{user_message[:50]}...'")
        
       # --- FIX START: Gather and pass conversation history to NLU ---
        
        # Get the last 4 turns of history from conversation memory for context
        history_turns = conversation_memory.conversation_history.get(current_user.student_id, [])
        
        # Format the history with role and content for the NLU processor
        nlu_history = []
        for turn in history_turns[-4:]: # Use the last 4 turns (2 user, 2 bot)
            nlu_history.append({'role': 'user', 'content': turn.user_message, 'intent': turn.intent})
            nlu_history.append({'role': 'assistant', 'content': turn.bot_response, 'intent': turn.intent})

        # Step 1: Enhanced NLU Processing with Full Context
        nlu_understanding = nlu_processor.understand_user_intent(
            user_message, 
            conversation_history=nlu_history
        )
        
        # --- FIX END ---
        
        # Step 2: Crisis Detection
        recent_conversation = [turn.user_message for turn in 
                             ConversationTurn.query.filter_by(user_id=current_user.id)
                             .order_by(ConversationTurn.timestamp.desc()).limit(10).all()]
        
        crisis_analysis = crisis_detector.detect_crisis_with_context(
            user_message, conversation_history=recent_conversation
        )
        
        logger.info(f"🔍 Crisis Analysis: {crisis_analysis['is_crisis']} "
                   f"(level: {crisis_analysis['risk_level']}, score: {crisis_analysis['crisis_score']:.2f})")
        
        # Step 3: Handle Crisis Situations
        if crisis_analysis['is_crisis'] and crisis_analysis['risk_level'] in ['high', 'crisis']:
            crisis_response = response_generator._generate_crisis_response(nlu_understanding, user_id_str)
            serializable_response = convert_numpy_types(crisis_response)
            return jsonify(serializable_response)
        
        # Step 4: Enhanced Response Generation with Ollama
        response_data = response_generator.generate_response(
            nlu_understanding=nlu_understanding,
            user_message=user_message,
            user_id=user_id_str,
            language=language_code, # <-- ADD THIS
            conversation_memory=conversation_memory,
            response_options=context,
            ollama_client=ollama_llama3
        )
        # --- ADD THIS CLEANUP SAFETY NET ---
        if 'response' in response_data and response_data['response']:
            response_data['response'] = clean_ai_response(response_data['response'])
        # ------------------------------------
        
        # Step 5: Intelligent Method Suggestions
        logger.debug("Step 5: Method Suggestions...")
        method_suggested = None
        method_info = None
        method_suggestion = None
        
        if (response_data.get('provides_concrete_help') and 
            nlu_understanding['primary_intent'] not in ['general_support', 'help_seeking'] and 
            nlu_understanding['severity_score'] > 0.3):
            
            method_suggestion = conversation_memory.suggest_method(
                user_id_str,
                nlu_understanding['primary_intent'],
                nlu_understanding['severity_score'],
                context=context
            )
            
            if method_suggestion:
                method_suggested = method_suggestion['method_id']
                method_info = method_suggestion
                logger.info(f"🛠️ Method suggested: {method_suggested}")
                
                # Add method to response if appropriate stage
                if response_data['conversation_stage'] in ['method_suggestion', 'gentle_help_offering']:
                    method_instructions = method_suggestion['method_data']
                    response_data['method_suggested'] = method_suggested
                    response_data['method_instructions'] = method_instructions
        
        # Step 6: Professional Help Assessment and Referral
        logger.debug("Step 6: Professional Help Assessment...")
        counselor_referral = None

        # Determine if we should suggest a counselor (only when no referral exists yet)
        should_refer_counselor = (
            (
                nlu_understanding['severity_score'] > 0.6 and
                current_user.total_conversations > 3 and
                current_user.counselor_referral_status == 'none'
            ) or (
                current_user.counselor_referral_status == 'none' and
                current_user.effective_methods_count < current_user.methods_tried_count and
                current_user.methods_tried_count > 2
            )
        )

        # Only suggest a counselor at an appropriate conversational moment
        appropriate_stages_for_referral = ['gentle_help_offering', 'ongoing_support', 'professional_referral']

        # If the generator already produced a referral OR we decided to suggest one here, record it once and avoid duplicating text
        if (
            (response_data.get('professional_referral') and current_user.counselor_referral_status == 'none') or
            (should_refer_counselor and not method_suggestion and response_data['conversation_stage'] in appropriate_stages_for_referral)
        ):
            logger.info(f"🏥 Counselor referral flow for {user_id_str}")

            # If no referral text from generator, create one structured field instead of appending to main response
            if not response_data.get('professional_referral'):
                counselor_recommendation = conversation_memory.get_counselor_recommendation(user_id_str)
                counselor_info = counselor_recommendation['counselor']
                referral_info = counselor_recommendation['referral_info']
                referral_text = (
                    f"**Professional Support Recommendation**\n\n"
                    f"Based on our conversations, I think working with **{counselor_info['name']}** could provide you with additional support and specialized techniques.\n\n"
                    f"**Why I recommend them:** {referral_info['why_recommended']}\n\n"
                    f"**What you can expect:** {referral_info['what_to_expect']}\n\n"
                    f"**Next steps:** {', '.join(referral_info['next_steps'][:3])}\n\n"
                    f"Would you like me to help you book a session?"
                )
                response_data['professional_referral'] = {
                    'text': referral_text,
                    'is_recommendation': True
                }
            else:
                # Extract counselor info from memory for DB logging
                counselor_recommendation = conversation_memory.get_counselor_recommendation(user_id_str)
                counselor_info = counselor_recommendation['counselor']

            # Track referral in database once
            counselor_interaction = CounselorInteraction(
                user_id=current_user.id,
                counselor_id=counselor_info['id'],
                counselor_name=counselor_info['name'],
                referral_reason=f"Severity: {nlu_understanding['severity_score']:.2f}, {current_user.total_conversations} conversations",
                referral_urgency=counselor_recommendation['referral_urgency']
            )
            counselor_interaction.set_specialties(counselor_info['specialties'])
            counselor_interaction.set_matched_concerns(nlu_understanding.get('user_needs', []))
            db.session.add(counselor_interaction)

            # Update user referral status only if previously none
            if current_user.counselor_referral_status == 'none':
                current_user.counselor_referral_status = 'suggested'
                current_user.referral_date = datetime.now()
                # Keep conversation memory in sync to avoid repeated recommendations
                try:
                    conversation_memory.update_counselor_referral_status(
                        user_id_str,
                        'suggested',
                        {'counselor_id': counselor_info['id']}
                    )
                except Exception as _e:
                    logger.warning(f"Could not update memory referral status for {user_id_str}: {_e}")

            counselor_referral = counselor_recommendation
            update_system_state('predict', professional_referrals=1)
        
        # Step 7: Save Comprehensive Conversation Turn to Database
        logger.debug("Step 7: Saving Conversation Turn...")
        
        conversation_turn = ConversationTurn(
            user_id=current_user.id,
            user_message=user_message,
            bot_response=response_data['response'],
            detected_intent=nlu_understanding['primary_intent'],
            intent_confidence=nlu_understanding['confidence'],
            severity_score=nlu_understanding['severity_score'],
            emotional_state=nlu_understanding['emotional_state'].get('primary_emotion', 'neutral'),
            emotional_intensity=nlu_understanding['emotional_state'].get('intensity', 0.0),
            is_crisis=crisis_analysis['is_crisis'],
            crisis_level=crisis_analysis['crisis_score'],
            crisis_risk_level=crisis_analysis['risk_level'],
            conversation_stage=response_data['conversation_stage'],
            response_type=response_data.get('intent_addressed'),
            provides_concrete_help=response_data.get('provides_concrete_help', False),
            method_suggested=method_suggested,
            is_follow_up=response_data.get('is_follow_up', False),
            urgency_level=nlu_understanding['urgency_level'],
            response_time_ms=int((time.time() - start_time) * 1000)
        )
        
        conversation_turn.set_context_entities(nlu_understanding['context_entities'])
        conversation_turn.set_user_needs(nlu_understanding['user_needs'])
        
        db.session.add(conversation_turn)
        
        # Step 8: Update User Profile Comprehensively
        logger.debug("Step 8: Updating User Profile...")
        
        current_user.total_conversations += 1
        current_user.current_conversation_stage = response_data['conversation_stage']
        current_user.current_risk_level = (crisis_analysis['risk_level'] if crisis_analysis['is_crisis'] 
                                         else current_user.current_risk_level)
        
        # Update severity tracking
        if not current_user.first_severity_score:
            current_user.first_severity_score = nlu_understanding['severity_score']
        current_user.latest_severity_score = nlu_understanding['severity_score']
        
        # Calculate and update average severity
        all_turns = ConversationTurn.query.filter_by(user_id=current_user.id).all()
        if all_turns:
            current_user.average_severity_score = sum(turn.severity_score for turn in all_turns) / len(all_turns)
        
        # Update primary concerns
        concerns = current_user.get_primary_concerns()
        intent = nlu_understanding['primary_intent']
        if intent not in concerns and intent not in ['general_support', 'help_seeking']:
            concerns.append(intent)
            current_user.set_primary_concerns(concerns[-5:])  # Keep top 5
        
        # Update improvement trend
        if len(all_turns) > 5:
            recent_scores = [turn.severity_score for turn in all_turns[-5:]]
            early_scores = [turn.severity_score for turn in all_turns[:5]]
            
            if sum(recent_scores) < sum(early_scores):
                current_user.improvement_trend = 'improving'
            elif sum(recent_scores) > sum(early_scores):
                current_user.improvement_trend = 'declining'
            else:
                current_user.improvement_trend = 'stable'
        
        # Update method tracking
        if method_suggested:
            current_user.methods_tried_count += 1
            current_user.current_method = method_suggested
            
            # Set method check due date
            method_data = conversation_memory.method_library.get(method_suggested, {})
            check_in_days = method_data.get('check_in_days', 7)
            current_user.method_check_due = datetime.now() + timedelta(days=check_in_days)
        
        db.session.commit()
        
        # Step 9: Add to Memory System
        logger.debug("Step 9: Adding to Memory System...")
        
        conversation_memory.add_conversation_turn(
            user_id=user_id_str,
            user_message=user_message,
            bot_response=response_data['response'],
            nlu_data=nlu_understanding,
            response_data=response_data
        )
        conversation_memory.save_memory()
        
        # Step 10: Prepare Comprehensive Enhanced Response
        with app.app_context():
            current_user = User.query.filter_by(student_id=user_id_str, is_active=True).first()
            
            enhanced_response = {
                **response_data,
                'analysis': {
                    'intent': nlu_understanding['primary_intent'],
                    'confidence': nlu_understanding['confidence'],
                    'severity': nlu_understanding['severity_score'],
                    'emotional_state': nlu_understanding.get('emotional_state', {}).get('primary_emotion', 'neutral'),
                    'emotional_intensity': nlu_understanding.get('emotional_state', {}).get('intensity', 0.0),
                    'urgency': nlu_understanding['urgency_level'],
                    'crisis_detected': crisis_analysis['is_crisis'],
                    'crisis_level': crisis_analysis['risk_level'],
                    'crisis_score': crisis_analysis['crisis_score'],
                    'in_scope': nlu_understanding.get('in_scope', True),
                    'llama_analysis_used': nlu_understanding.get('ollama_analysis_used', False)
                },
                'user_progress': {
                    'total_conversations': current_user.total_conversations
                },
                'system_info': {
                    'response_time_ms': int((time.time() - start_time) * 1000),
                    'ollama_available': ollama_llama3.is_available,
                }
            }
            
            serializable_response = convert_numpy_types(enhanced_response)
            return jsonify(serializable_response)
    
    except Exception as e:
        logger.error(f"❌ Error in predict for user {data.get('userId', 'unknown')}: {str(e)}")
        logger.error(traceback.format_exc())
        update_system_state('predict', success=False)
        
        error_response = {
            "response": "I apologize, but I'm having some technical difficulties right now. Please try again in a moment, or contact emergency services at 911 if this is urgent.",
            "error": True,
            "error_type": "processing_error",
            "fallback_resources": {
                "crisis_hotline": "988",
                "emergency": "911",
                "crisis_text": "Text HOME to 741741"
            }
        }
        
        serializable_error = convert_numpy_types(error_response)
        return jsonify(serializable_error), 500

@app.route("/v1/method-feedback", methods=["POST"])
def method_feedback():
    """Process comprehensive user feedback on suggested methods"""
    try:
        update_system_state('method_feedback')
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_id_str = data.get("userId", "").strip()
        feedback_text = data.get("feedback", "").strip()
        method_id = data.get("methodId", "").strip()
        effectiveness_rating = data.get("effectivenessRating")
        
        if not user_id_str or not feedback_text:
            return jsonify({"error": "User ID and feedback are required"}), 400
        
        # Find user
        current_user = User.query.filter_by(student_id=user_id_str, is_active=True).first()
        if not current_user:
            return jsonify({"error": "User not found"}), 401
        
        # Process feedback through memory system
        if not conversation_memory:
            return jsonify({"error": "Memory system unavailable"}), 503
        
        feedback_result = conversation_memory.process_method_feedback(
            user_id_str,
            feedback_text,
            method_id
        )
        
        if feedback_result['processed']:
            # Update or create database feedback record
            method_feedback_record = MethodFeedback.query.filter_by(
                user_id=current_user.id,
                method_id=feedback_result['method_id']
            ).first()
            
            if not method_feedback_record:
                # Get method info from memory
                method_data = conversation_memory.method_library.get(
                    feedback_result['method_id'], {}
                )
                
                method_feedback_record = MethodFeedback(
                    user_id=current_user.id,
                    method_id=feedback_result['method_id'],
                    method_name=method_data.get('name', feedback_result['method_id']),
                    method_category=method_data.get('category', 'unknown'),
                    severity_when_suggested=current_user.latest_severity_score or 0.5,
                    intent_when_suggested=current_user.get_primary_concerns()[-1] if current_user.get_primary_concerns() else 'unknown'
                )
                
                db.session.add(method_feedback_record)
            
            # Update feedback
            effectiveness_score = feedback_result['effectiveness_score']
            effectiveness_rating = feedback_result['feedback_category']
            
            method_feedback_record.update_effectiveness(
                effectiveness_rating,
                effectiveness_score,
                feedback_text
            )
            
            method_feedback_record.mark_as_used()
            
            # Update user method counts
            if effectiveness_score > 0.6:
                current_user.effective_methods_count += 1
                method_feedback_record.improvement_noted = True
            
            # Clear current method from user
            current_user.current_method = None
            current_user.method_check_due = None
            
            db.session.commit()
            
            logger.info(f"✅ Method feedback processed for {user_id_str}: "
                       f"{feedback_result['method_id']} -> {effectiveness_rating}")
            
            return jsonify({
                "success": True,
                "feedback_processed": True,
                "method_id": feedback_result['method_id'],
                "effectiveness_score": effectiveness_score,
                "effectiveness_category": effectiveness_rating,
                "detailed_analysis": feedback_result['detailed_analysis'],
                "next_steps": feedback_result['next_steps'],
                "method_summary": feedback_result['method_summary'],
                "user_progress": {
                    "methods_tried": current_user.methods_tried_count,
                    "effective_methods": current_user.effective_methods_count,
                    "success_rate": current_user.calculate_method_success_rate()
                }
            })
        
        return jsonify({
            "success": False,
            "message": "Unable to process feedback",
            "reason": feedback_result.get('reason', 'Unknown error')
        }), 400
    
    except Exception as e:
        logger.error(f"❌ Method feedback error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('method_feedback', success=False)
        
        return jsonify({
            "error": "Failed to process feedback due to server error",
            "message": "Please try again later"
        }), 500

@app.route("/v1/book-counselor", methods=["POST"])
def book_counselor():
    """Handle comprehensive counselor booking requests"""
    try:
        update_system_state('book_counselor')
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_id_str = data.get("userId", "").strip()
        counselor_id = data.get("counselorId", "")
        interest_level = data.get("interestLevel", "interested")
        
        if not user_id_str:
            return jsonify({"error": "User ID is required"}), 400
        
        current_user = User.query.filter_by(student_id=user_id_str, is_active=True).first()
        if not current_user:
            return jsonify({"error": "User not found"}), 401
        
        if not conversation_memory:
            return jsonify({"error": "Memory system unavailable"}), 503
        
        # Get counselor recommendation
        recommendation = conversation_memory.get_counselor_recommendation(user_id_str)
        counselor_info = recommendation['counselor']
        
        # If specific counselor requested, use that one
        if counselor_id:
            # Find specific counselor
            specific_counselor = next(
                (c for c in conversation_memory.counselor_database if c['id'] == counselor_id),
                None
            )
            if specific_counselor:
                counselor_info = specific_counselor
        
        # Update user referral status based on interest level
        status_mapping = {
            'interested': 'interested',
            'very_interested': 'very_interested',
            'ready_to_book': 'ready_to_book',
            'booked': 'booked',
            'attending': 'attending'
        }
        
        current_user.counselor_referral_status = status_mapping.get(interest_level, 'interested')
        
        # Update counselor interaction record
        interaction = CounselorInteraction.query.filter_by(
            user_id=current_user.id,
            counselor_id=counselor_info['id']
        ).order_by(CounselorInteraction.referral_date.desc()).first()
        
        if interaction:
            interaction.update_status(current_user.counselor_referral_status)
        else:
            # Create new interaction record
            interaction = CounselorInteraction(
                user_id=current_user.id,
                counselor_id=counselor_info['id'],
                counselor_name=counselor_info['name'],
                referral_reason="User initiated booking request",
                referral_urgency=recommendation.get('referral_urgency', 'standard')
            )
            
            interaction.set_specialties(counselor_info['specialties'])
            db.session.add(interaction)
        
        # Update memory system
        conversation_memory.update_counselor_referral_status(
            user_id_str,
            current_user.counselor_referral_status,
            {'counselor_id': counselor_info['id'], 'interest_level': interest_level}
        )
        
        db.session.commit()
        
        # Prepare comprehensive response
        logistics = counselor_info.get('logistics', {})
        contact = counselor_info.get('contact', {})
        
        booking_response = {
            "success": True,
            "counselor": {
                "id": counselor_info['id'],
                "name": counselor_info['name'],
                "title": counselor_info['title'],
                "specialties": counselor_info['specialties'],
                "description": counselor_info.get('description', ''),
                "approaches": counselor_info.get('approaches', [])
            },
            "logistics": {
                "wait_time": f"{counselor_info.get('availability', {}).get('wait_time_weeks', 'Unknown')} weeks",
                "session_types": logistics.get('session_types', ['in_person']),
                "insurance": logistics.get('insurance', ['Contact for details']),
                "location": logistics.get('location', 'Contact for details')
            },
            "contact": {
                "phone": contact.get('phone', 'Contact information not available'),
                "website": contact.get('website', ''),
                "booking_method": contact.get('booking_method', 'Call for appointment')
            },
            "next_steps": recommendation['preparation_suggestions'],
            "booking_status": current_user.counselor_referral_status,
            "message": f"Great! I've noted your interest in working with {counselor_info['name']}. Here's everything you need to know to get started."
        }
        
        logger.info(f"✅ Counselor booking processed for {user_id_str}: "
                   f"{counselor_info['name']} (Status: {current_user.counselor_referral_status})")
        
        return jsonify(booking_response)
    
    except Exception as e:
        logger.error(f"❌ Counselor booking error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('book_counselor', success=False)
        
        return jsonify({
            "error": "Failed to process booking request",
            "message": "Please try again later"
        }), 500

@app.route("/v1/history", methods=["POST"])
def get_history():
    """Get comprehensive conversation history with analytics"""
    try:
        update_system_state('get_history')
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_id_str = data.get("userId", "")
        limit = min(data.get("limit", 50), 100)
        include_analysis = data.get("includeAnalysis", False)
        
        if not user_id_str:
            return jsonify({"error": "User ID is required"}), 400
        
        current_user = User.query.filter_by(student_id=user_id_str, is_active=True).first()
        if not current_user:
            return jsonify({"error": "User not found"}), 401
        
        # Get conversation turns with ordering
        turns = ConversationTurn.query.filter_by(user_id=current_user.id)\
                .order_by(ConversationTurn.timestamp.asc())\
                .limit(limit).all()
        
        # Format conversation history
        chat_log = []
        for turn in turns:
            # User message
            user_entry = {
                "role": "user",
                "content": turn.user_message,
                "timestamp": turn.timestamp.isoformat(),
                "turn_id": turn.id
            }
            
            # Assistant response with optional analysis
            assistant_entry = {
                "role": "assistant", 
                "content": turn.bot_response,
                "timestamp": turn.timestamp.isoformat(),
                "turn_id": turn.id
            }
            
            if include_analysis:
                assistant_entry["analysis"] = {
                    "intent": turn.detected_intent,
                    "confidence": turn.intent_confidence,
                    "severity": turn.severity_score,
                    "emotional_state": turn.emotional_state,
                    "emotional_intensity": turn.emotional_intensity,
                    "stage": turn.conversation_stage,
                    "crisis_detected": turn.is_crisis,
                    "crisis_level": turn.crisis_risk_level,
                    "method_suggested": turn.method_suggested,
                    "response_time": turn.response_time_ms,
                    "urgency": turn.urgency_level
                }
            
            chat_log.extend([user_entry, assistant_entry])
        
        # Get user progress summary
        user_summary = conversation_memory.get_user_summary(user_id_str) if conversation_memory else {}
        
        response_data = {
            "success": True,
            "history": chat_log,
            "summary": {
                "total_conversations": current_user.total_conversations,
                "current_stage": current_user.current_conversation_stage,
                "risk_level": current_user.current_risk_level,
                "improvement_trend": current_user.improvement_trend,
                "methods_tried": current_user.methods_tried_count,
                "effective_methods": current_user.effective_methods_count,
                "method_success_rate": current_user.calculate_method_success_rate(),
                "member_since": current_user.created_at.isoformat(),
                "last_interaction": current_user.last_login.isoformat() if current_user.last_login else None
            }
        }
        
        # Add detailed progress if available
        if user_summary.get('exists'):
            response_data["progress_analytics"] = user_summary.get('progress_metrics', {})
            response_data["method_analytics"] = user_summary.get('method_effectiveness', {})
            response_data["risk_assessment"] = user_summary.get('risk_assessment', {})
        
        logger.info(f"✅ History retrieved for {user_id_str}: {len(turns)} turns")
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"❌ History retrieval error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('get_history', success=False)
        
        return jsonify({
            "error": "Failed to retrieve conversation history",
            "message": "Please try again later"
        }), 500

@app.route("/v1/user-stats", methods=["POST"])
def get_user_stats():
    """Get comprehensive user statistics and analytics"""
    try:
        update_system_state('get_user_stats')
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_id_str = data.get("userId", "")
        
        if not user_id_str:
            return jsonify({"error": "User ID is required"}), 400
        
        current_user = User.query.filter_by(student_id=user_id_str, is_active=True).first()
        if not current_user:
            return jsonify({"error": "User not found"}), 401
        
        # Get comprehensive statistics from database
        db_stats = get_user_statistics(current_user.id)
        
        # Get memory system statistics if available
        memory_stats = {}
        if conversation_memory:
            memory_stats = conversation_memory.get_user_summary(user_id_str)
        
        # Combine statistics
        comprehensive_stats = {
            "success": True,
            "user_info": {
                "student_id": current_user.student_id,
                "full_name": current_user.full_name,
                "email": current_user.email,
                "member_since": current_user.created_at.isoformat(),
                "last_active": current_user.last_login.isoformat() if current_user.last_login else None,
                "total_conversations": current_user.total_conversations,
                "current_stage": current_user.current_conversation_stage,
                "risk_level": current_user.current_risk_level,
                "improvement_trend": current_user.improvement_trend
            },
            "conversation_analytics": {
                "total_interactions": current_user.total_conversations,
                "first_severity": current_user.first_severity_score,
                "latest_severity": current_user.latest_severity_score,
                "average_severity": current_user.average_severity_score,
                "primary_concerns": current_user.get_primary_concerns()
            },
            "method_effectiveness": {
                "methods_tried": current_user.methods_tried_count,
                "effective_methods": current_user.effective_methods_count,
                "success_rate": current_user.calculate_method_success_rate(),
                "current_method": current_user.current_method,
                "next_check_in": current_user.method_check_due.isoformat() if current_user.method_check_due else None
            },
            "professional_help": {
                "referral_status": current_user.counselor_referral_status,
                "referral_date": current_user.referral_date.isoformat() if current_user.referral_date else None,
                "crisis_count": current_user.crisis_count,
                "last_crisis": current_user.last_crisis_date.isoformat() if current_user.last_crisis_date else None
            }
        }
        
        # Add database statistics if available
        if db_stats:
            comprehensive_stats["detailed_analytics"] = db_stats
        
        # Add memory system analytics if available
        if memory_stats.get('exists'):
            comprehensive_stats["advanced_analytics"] = {
                "progress_metrics": memory_stats.get('progress_metrics', {}),
                "method_analytics": memory_stats.get('method_effectiveness', {}),
                "risk_assessment": memory_stats.get('risk_assessment', {}),
                "personalization": memory_stats.get('personalization', {})
            }
        
        logger.info(f"✅ User stats retrieved for {user_id_str}")
        
        return jsonify(comprehensive_stats)
    
    except Exception as e:
        logger.error(f"❌ User stats error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('get_user_stats', success=False)
        
        return jsonify({
            "error": "Failed to retrieve user statistics",
            "message": "Please try again later"
        }), 500

@app.route("/v1/health", methods=["GET"])
def health_check():
    """Comprehensive system health check with Ollama status"""
    try:
        # System health overview
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": round((datetime.now() - system_state['startup_time']).total_seconds() / 3600, 2),
            "components": system_status,
            "api_ollama_integration": {
                "available": ollama_llama3.is_available,
                "status": "connected" if ollama_llama3.is_available else "fallback_mode"
            },
            "system_metrics": {
                "total_requests": system_state['total_requests'],
                "successful_responses": system_state['successful_responses'],
                "error_count": system_state['error_count'],
                "success_rate": system_state['successful_responses'] / max(system_state['total_requests'], 1),
                "crisis_interventions": system_state['crisis_interventions'],
                "methods_suggested": system_state['methods_suggested'],
                "professional_referrals": system_state['professional_referrals'],
                "llama_responses": system_state.get('llama_responses', 0),
                "fallback_responses": system_state.get('fallback_responses', 0)
            }
        }
        
        # Database health check
        try:
            with app.app_context():
                total_users = User.query.count()
                total_conversations = ConversationTurn.query.count()
                health_status["database"] = {
                    "status": "connected",
                    "total_users": total_users,
                    "total_conversations": total_conversations
                }
        except Exception as e:
            health_status["database"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Overall health assessment
        critical_components = ['nlu_processor', 'response_generator', 'conversation_memory', 'crisis_detector', 'database']
        if not all(system_status.get(comp, False) for comp in critical_components):
            health_status["status"] = "degraded"
        
        return jsonify(health_status)
    
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route("/v1/admin/metrics", methods=["GET"])
def get_system_metrics():
    """Get system-wide metrics and analytics"""
    try:
        # Get recent metrics from database
        recent_metrics = SystemMetrics.query.order_by(SystemMetrics.metrics_date.desc()).limit(30).all()
        
        metrics_data = []
        for metric in recent_metrics:
            metrics_data.append({
                "date": metric.metrics_date.isoformat(),
                "total_users": metric.total_active_users,
                "new_users": metric.new_users_registered,
                "conversations": metric.total_conversations,
                "crisis_events": metric.crisis_events_detected,
                "methods_suggested": metric.methods_suggested_total,
                "method_success_rate": metric.overall_method_success_rate,
                "counselor_referrals": metric.counselor_referrals_made
            })
        
        # Get system analytics from memory if available
        system_analytics = {}
        if conversation_memory:
            try:
                system_analytics = conversation_memory.get_system_analytics()
            except Exception as e:
                logger.error(f"Error getting system analytics: {e}")
        
        # Current system state
        current_state = {
            **system_state,
            'component_health': system_status,
            'current_time': datetime.now().isoformat()
        }
        
        return jsonify({
            "success": True,
            "historical_metrics": metrics_data,
            "system_analytics": system_analytics,
            "current_state": current_state,
            "summary": {
                "total_metrics_days": len(metrics_data),
                "latest_date": metrics_data[0]["date"] if metrics_data else None,
                "system_health": "healthy" if all(system_status.values()) else "degraded"
            }
        })
    
    except Exception as e:
        logger.error(f"❌ Metrics retrieval error: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "error": "Failed to retrieve system metrics",
            "message": "Check system logs for details"
        }), 500

@app.route("/v1/save-models", methods=["POST"])
def save_models_endpoint():
    """Manually trigger comprehensive model saving"""
    try:
        # Authentication check (basic - enhance in production)
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('ADMIN_API_KEY', 'admin_key_123'):
            return jsonify({"error": "Unauthorized"}), 401
        
        success = save_all_models()
        
        if success:
            return jsonify({
                "success": True,
                "message": "All AI models saved successfully",
                "timestamp": datetime.now().isoformat(),
                "models_saved": {
                    "nlu_processor": system_status['nlu_processor'],
                    "conversation_memory": system_status['conversation_memory'],
                    "crisis_detector": system_status['crisis_detector']
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": "Some models failed to save - check logs for details",
                "timestamp": datetime.now().isoformat()
            }), 500
    
    except Exception as e:
        logger.error(f"❌ Model saving endpoint error: {e}")
        return jsonify({
            "success": False,
            "message": f"Error saving models: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/v1/system/status", methods=["GET"])
def system_status_endpoint():
    """Get detailed system status information"""
    return jsonify({
        "system_info": {
            "application_name": "Enhanced Mental Health Chatbot with Ollama Llama 3",
            "version": "2.1.0",
            "startup_time": system_state['startup_time'].isoformat(),
            "current_time": datetime.now().isoformat(),
            "uptime_hours": round((datetime.now() - system_state['startup_time']).total_seconds() / 3600, 2)
        },
        "components": system_status,
        "api_ollama_integration": {
            "available": ollama_llama3.is_available,
            "model": ollama_llama3.client.model if ollama_llama3.is_available else "N/A",
            "base_url": ollama_llama3.client.base_url if ollama_llama3.is_available else "N/A"
        },
        "features": {
            "progressive_conversation_stages": True,
            "method_effectiveness_tracking": True,
            "optimized_crisis_detection": True,
            "professional_counselor_integration": True,
            "comprehensive_user_analytics": True,
            "persistent_conversation_memory": True,
            "enhanced_emotional_analysis": True,
            "real_time_system_monitoring": True,
            "ollama_ai_enhancement": ollama_llama3.is_available
        },
        "performance": {
            "total_requests": system_state['total_requests'],
            "successful_responses": system_state['successful_responses'],
            "error_count": system_state['error_count'],
            "success_rate": system_state['successful_responses'] / max(system_state['total_requests'], 1),
            "crisis_interventions": system_state['crisis_interventions'],
            "methods_suggested": system_state['methods_suggested'],
            "professional_referrals": system_state['professional_referrals'],
            "llama_responses": system_state.get('llama_responses', 0),
            "fallback_responses": system_state.get('fallback_responses', 0)
        }
    })

# Ollama-specific endpoints
@app.route("/v1/ollama/status", methods=["GET"])
def ollama_status():
    """Get Ollama integration status"""
    return jsonify({
        "ollama_available": ollama_llama3.is_available,
        "client_info": {
            "base_url": ollama_llama3.client.base_url,
            "model": ollama_llama3.client.model
        },
        "integration_working": system_status.get('ollama_llama3', False),
        "responses_generated": system_state.get('llama_responses', 0),
        "fallback_responses": system_state.get('fallback_responses', 0)
    })

@app.route("/v1/ollama/test", methods=["POST"])
def test_ollama():
    """Test Ollama integration"""
    try:
        data = request.get_json()
        test_message = data.get("message", "Hello, how are you?")
        
        if ollama_llama3.is_available:
            response = ollama_llama3.generate_mental_health_response(
                user_message=test_message,
                user_intent="general_support",
                conversation_stage="initial_contact",
                severity_score=0.3
            )
            
            if response:
                return jsonify({
                    "success": True,
                    "ollama_response": response,
                    "message": "Ollama integration working correctly"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Ollama available but failed to generate response"
                }), 500
        else:
            return jsonify({
                "success": False,
                "message": "Ollama not available",
                "fallback_active": True
            }), 503
    
    except Exception as e:
        logger.error(f"Ollama test error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Scheduled tasks and cleanup
def cleanup_on_exit():
    """Cleanup tasks on application shutdown"""
    logger.info("🔄 Application shutdown - performing cleanup...")
    try:
        with app.app_context():
            # Save all models
            save_all_models()
            
            # Update final metrics
            track_system_metrics()
            
            # Close database connections
            db.session.close()
        
        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_on_exit)

# Periodic tasks (every 24 hours)
def run_periodic_tasks():
    """Run periodic maintenance tasks"""
    try:
        logger.info("🔄 Running periodic maintenance tasks...")
        
        # Track daily metrics
        track_system_metrics()
        
        # Clean up old data (keep 90 days)
        if conversation_memory:
            conversation_memory.cleanup_old_data(days_to_keep=90)
        
        # Save models
        save_all_models()
        
        logger.info("✅ Periodic maintenance completed")
    except Exception as e:
        logger.error(f"❌ Error in periodic tasks: {e}")

def schedule_periodic_tasks():
    """Schedule periodic tasks to run every 24 hours"""
    def task_scheduler():
        while True:
            time.sleep(24 * 60 * 60)  # 24 hours
            run_periodic_tasks()
    
    scheduler_thread = threading.Thread(target=task_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("✅ Periodic task scheduler started")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested resource does not exist",
        "available_endpoints": [
            "/v1/health", "/v1/register", "/v1/login", "/v1/logout",
            "/v1/predict", "/v1/method-feedback", "/v1/book-counselor",
            "/v1/history", "/v1/user-stats", "/v1/ollama/status", "/v1/ollama/test"
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The requested HTTP method is not supported for this endpoint"
    }), 405

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "Request too large",
        "message": "The request payload is too large"
    }), 413

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later."
    }), 429

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later.",
        "support_resources": {
            "crisis_hotline": "988",
            "emergency": "911",
            "crisis_text": "Text HOME to 741741"
        }
    }), 500

# Enhanced startup display with Ollama information
def display_startup_info():
    """Display comprehensive startup information with Ollama status"""
    print("=" * 100)
    print("🚀 ENHANCED MENTAL HEALTH CHATBOT SYSTEM v2.1 - WITH OLLAMA LLAMA 3")
    print("=" * 100)
    print()
    print("🦙 OLLAMA LLAMA 3 INTEGRATION:")
    print(f" {'✅' if ollama_llama3.is_available else '❌'} Ollama Status: {'Available' if ollama_llama3.is_available else 'Not Available'}")
    print(f" 🔗 Model: {ollama_llama3.client.model}")
    print(f" 🌐 Base URL: {ollama_llama3.client.base_url}")
    
    if ollama_llama3.is_available:
        print(" 🎯 AI-Enhanced Features: Natural conversation, contextual responses, intelligent intent analysis")
    else:
        print(" ⚠️ Fallback Mode: Using rule-based responses with full safety protocols")
    
    print()
    print("🌟 ENHANCED FEATURES:")
    print(" ✅ Progressive conversation stages (human-like, no walls of text)")
    print(" ✅ Optimized crisis detection (reduced false positives)")  
    print(" ✅ Advanced method effectiveness tracking across sessions")
    print(" ✅ Professional counselor integration with intelligent matching")
    print(" ✅ Comprehensive user analytics and progress tracking")
    print(" ✅ Enhanced database with complete relationship mapping")
    print(" ✅ Real-time system health monitoring and metrics")
    print(" ✅ Advanced conversation memory with context awareness")
    print(" ✅ Intelligent personalization and user preference learning")
    print(" ✅ Comprehensive security and session management")
    
    if ollama_llama3.is_available:
        print(" 🆕 AI-Enhanced Natural Language Understanding")
        print(" 🆕 Context-Aware Response Generation")
        print(" 🆕 Intelligent Crisis Analysis Enhancement")
    
    print()
    print("🧠 AI COMPONENTS STATUS:")
    for component, status in system_status.items():
        status_icon = "✅" if status else "❌"
        component_name = component.replace('_', ' ').title()
        if component == 'ollama_llama3':
            component_name = "Ollama Llama 3"
        print(f" {status_icon} {component_name}")
    
    print()
    print("🌐 COMPLETE API ENDPOINTS:")
    print(" • POST /v1/register - Enhanced user registration with validation")
    print(" • POST /v1/login - Secure user authentication with session tracking")
    print(" • POST /v1/logout - Clean session termination")
    print(" • POST /v1/predict - Main conversation endpoint with full AI pipeline + Ollama")
    print(" • POST /v1/method-feedback - Advanced method effectiveness tracking")
    print(" • POST /v1/book-counselor - Intelligent counselor booking system")
    print(" • POST /v1/history - Comprehensive conversation history with analytics")
    print(" • POST /v1/user-stats - Detailed user statistics and progress tracking")
    print(" • GET /v1/health - Comprehensive system health diagnostics")
    print(" • GET /v1/ollama/status - Ollama Llama 3 status and statistics")
    print(" • POST /v1/ollama/test - Test Ollama Llama 3 functionality")
    print(" • GET /v1/admin/metrics - System metrics (admin)")
    print(" • POST /v1/save-models - Manual model saving (admin)")
    print(" • GET /v1/system/status - System status information")
    
    print()
    print("📞 EMERGENCY RESOURCES:")
    print(" • National Suicide Prevention Lifeline: 988")
    print(" • Crisis Text Line: Text HOME to 741741")
    print(" • Emergency Services: 911")
    print(" • SAMHSA National Helpline: 1-800-662-4357")
    
    print()
    print("💡 THIS SYSTEM NOW PROVIDES AI-ENHANCED MENTAL HEALTH SUPPORT!")
    print("   WITH ADVANCED NATURAL LANGUAGE UNDERSTANDING AND CONTEXTUAL RESPONSES")
    
    print()
    print("=" * 100)
    print("🎉 SYSTEM READY - ALL ENHANCED FEATURES + OLLAMA LLAMA 3 ACTIVE")
    print("=" * 100)

if __name__ == "__main__":
    # Display startup information
    display_startup_info()
    
    # Initialize periodic tasks
    schedule_periodic_tasks()
    
    # Track initial system startup
    with app.app_context():
        try:
            track_system_metrics()
        except Exception as e:
            logger.error(f"Failed to track startup metrics: {e}")
    
    # Start the Flask application
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)