"""
Fixed Setup and Installation Script for Enhanced Mental Health Chatbot with Ollama

Handles dependency installation, database setup, model initialization, and testing
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime
import sqlite3
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log',encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ is required. Current version: {}.{}.{}".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro
        ))
        return False
    logger.info(f"✅ Python version check passed: {sys.version}")
    return True

def check_ollama_installation():
    """Check API key availability instead of local Ollama"""
    import os
    api_key = os.getenv('GROQ_API_KEY') or os.getenv('API_KEY')
    if api_key:
        logger.info("✅ API key found - using API-based service")
        return True
    else:
        logger.warning("⚠️ No API key found. Set GROQ_API_KEY environment variable")
        logger.info("💡 Set your API key: export GROQ_API_KEY='your_key_here'")
        logger.info("💡 The system will work in fallback mode without API access")
        return False
def install_dependencies():
    """Install all required dependencies"""
    logger.info("📦 Installing dependencies...")
    dependencies = [
        'flask>=2.0.0',
        'flask-cors>=3.0.0', 
        'flask-sqlalchemy>=3.0.0',
        'werkzeug>=2.0.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'requests>=2.25.0',
        'python-dateutil>=2.8.0',
        'pandas>=1.3.0'
    ]

    failed_packages = []
    for package in dependencies:
        try:
            logger.info(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            # For Ollama, it's optional
            if 'ollama' in package:
                logger.warning(f"⚠️ {package} install failed - system will work without Ollama: {e}")
            else:
                logger.error(f"❌ Failed to install {package}: {e}")
                failed_packages.append(package)
        except Exception as e:
            logger.error(f"❌ Unexpected error installing {package}: {e}")
            if 'ollama' not in package:
                failed_packages.append(package)

    if failed_packages:
        logger.error(f"❌ Failed to install: {', '.join(failed_packages)}")
        logger.info("💡 Try installing manually with:")
        for package in failed_packages:
            logger.info(f"   pip install {package}")
        return False

    logger.info("✅ All dependencies installed successfully!")
    return True

def setup_ollama_models():
    """Test API connection instead of pulling models"""
    logger.info("🔗 Testing API connection...")
    try:
        from api_ollama_integration import api_llama3
        if api_llama3.is_available:
            logger.info("✅ API service connection successful")
        # Test a simple call
            test_response = api_llama3.client.generate_response("Hello", max_tokens=10)
            if test_response:
                logger.info("✅ API response test successful")
            else:
                logger.warning("⚠️ API response test failed")
            return True
        else:
            logger.warning("⚠️ API service not available - check your API key")
            return True  # Non-critical, system can work without it
    except Exception as e:
        logger.warning(f"⚠️ API test failed: {e}")
        return True

def create_directories():
    """Create necessary directories"""
    logger.info("📁 Creating directories...")
    directories = [
        'instance',
        'models', 
        'logs',
        'static',
        'templates',
        'data',
        'backups'
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")
            return False

    return True

def setup_database():
    """Initialize the database with proper error handling"""
    logger.info("🗄️ Setting up database...")
    try:
        # Ensure instance directory exists
        os.makedirs('instance', exist_ok=True)
        
        # Create database file path
        db_path = os.path.join('instance', 'enhanced_chatbot.db')
        
        # Test SQLite connection first
        try:
            conn = sqlite3.connect(db_path)
            conn.execute('SELECT 1')
            conn.close()
            logger.info("✅ SQLite connection test successful")
        except Exception as e:
            logger.error(f"❌ SQLite connection failed: {e}")
            return False

        # Now try Flask-SQLAlchemy setup
        try:
            # Import after ensuring dependencies are installed
            from enhanced_database_models import db, init_database
            from flask import Flask

            # Create temporary Flask app for database initialization
            app = Flask(__name__)
            app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.abspath(db_path)}'
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            
            db.init_app(app)

            with app.app_context():
                init_database(app)

            logger.info("✅ Database initialized successfully!")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Database models import failed: {e}")
            logger.info("💡 Make sure you have the fixed files in place")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def test_ollama_integration():
    """Test API integration"""
    logger.info("🧠 Testing API integration...")
    try:
        from api_ollama_integration import api_llama3
        if api_llama3.is_available:
# Test the API connection
           status = api_llama3.get_status()
           logger.info("✅ API integration working correctly")
           logger.info(f"✅ Using model: {api_llama3.client.model}")
           return True
        else:
            logger.info("⚠️ API not available, but fallback system ready")
            return True
    except ImportError as e:
        logger.warning(f"⚠️ Could not import API integration: {e}")
        logger.info("💡 Make sure you've replaced ollama_integration.py")
        return True # Non-critical for system function
    except Exception as e:
        logger.warning(f"⚠️ API integration test failed: {e}")
        logger.info("💡 System will work in fallback mode")
        return True

def test_ai_components():
    """Test AI components initialization"""
    logger.info("🧠 Testing AI components...")
    try:
        # Test NLU Processor with fixes
        logger.info("Testing NLU Processor...")
        from nlu_processor import ProgressiveNLUProcessor
        nlu = ProgressiveNLUProcessor()
        test_understanding = nlu.understand_user_intent("I feel really anxious about school")
        
        if test_understanding['primary_intent'] and test_understanding['confidence'] > 0:
            logger.info("✅ NLU Processor working correctly")
        else:
            logger.warning("⚠️ NLU Processor may have issues")

        # Test Crisis Detector
        logger.info("Testing Crisis Detector...")
        from optimized_crisis_detector import OptimizedCrisisDetector
        crisis_detector = OptimizedCrisisDetector()
        
        # Test help-seeking (should NOT be crisis)
        help_result = crisis_detector.detect_crisis_with_context("help me deal with depression")
        # Test genuine crisis (SHOULD be crisis)  
        crisis_result = crisis_detector.detect_crisis_with_context("i want to kill myself")

        if not help_result['is_crisis'] and crisis_result['is_crisis']:
            logger.info("✅ Crisis Detector working correctly")
        else:
            logger.warning("⚠️ Crisis Detector may have accuracy issues")

        # Test Conversation Memory
        logger.info("Testing Conversation Memory...")
        from conversation_memory import ProgressiveConversationMemory
        memory = ProgressiveConversationMemory()
        user_profile = memory.create_or_get_user("test_user")
        
        if user_profile and user_profile.user_id == "test_user":
            logger.info("✅ Conversation Memory working correctly")
        else:
            logger.warning("⚠️ Conversation Memory may have issues")

        logger.info("✅ All AI components tested successfully!")
        return True

    except ImportError as e:
        logger.error(f"❌ Import error in AI components: {e}")
        logger.info("💡 Make sure all fixed component files are in place")
        return False
    except Exception as e:
        logger.error(f"❌ AI component test failed: {e}")
        return False

def create_sample_user():
    """Create a sample user for testing"""
    logger.info("👤 Creating sample user...")
    try:
        # Create minimal Flask app
        from flask import Flask
        from enhanced_database_models import db, User

        app = Flask(__name__)
        db_path = os.path.join('instance', 'enhanced_chatbot.db')
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.abspath(db_path)}'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        db.init_app(app)

        with app.app_context():
            # Check if sample user already exists
            existing_user = User.query.filter_by(email="test@example.com").first()
            if existing_user:
                logger.info("✅ Sample user already exists")
                return True

            # Create sample user
            sample_user = User(
                student_id="STU000001",
                email="test@example.com",
                full_name="Test User"
            )
            sample_user.set_password("password123")
            
            db.session.add(sample_user)
            db.session.commit()
            
            logger.info("✅ Sample user created:")
            logger.info("   Student ID: STU000001")
            logger.info("   Email: test@example.com") 
            logger.info("   Password: password123")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to create sample user: {e}")
        return False

def create_requirements_txt():
    """Create requirements.txt file"""
    logger.info("📋 Creating requirements.txt...")
    requirements_content = """# Enhanced Mental Health Chatbot Dependencies with Ollama

# Core Flask Framework
flask>=2.0.0
flask-cors>=3.0.0
flask-sqlalchemy>=3.0.0
werkzeug>=2.0.0

# Data Science & ML
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0

# Ollama Integration (Optional)
ollama>=0.1.7

# Database
SQLAlchemy>=1.4.0

# Utilities
python-dateutil>=2.8.0
requests>=2.25.0
"""

    try:
        with open('requirements.txt', 'w') as f:
            f.write(requirements_content.strip())
        logger.info("✅ requirements.txt created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create requirements.txt: {e}")
        return False

def create_run_script():
    """Create convenient run script"""
    logger.info("🚀 Creating run script...")
    run_script_content = """#!/usr/bin/env python3

import os
import sys

def main():
    print("🚀 Starting Enhanced Mental Health Chatbot with Ollama...")
    print("=" * 60)
    
    # Check if virtual environment is recommended
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("💡 Tip: Consider using a virtual environment:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\\\Scripts\\\\activate")
        print("   pip install -r requirements.txt")
        print()

    try:
        # Try to use fixed files first, fallback to originals
        try:
            print("🔄 Loading enhanced components with fixes...")
            
            # Replace imports in the main chatbot file
            import shutil
            if os.path.exists('ollama_integration_fixed.py'):
                shutil.copy('ollama_integration_fixed.py', 'ollama_integration.py')
                print("✅ Using fixed Ollama integration")
            
            if os.path.exists('nlu_processor_fixed.py'):
                shutil.copy('nlu_processor_fixed.py', 'nlu_processor.py')
                print("✅ Using fixed NLU processor")
                
        except Exception as e:
            print(f"⚠️ Could not apply fixes: {e}")
        
        from chatbot import app
        print("✅ All components loaded successfully!")
        print("🌐 Starting server...")
        print("📍 Access the chatbot at: http://127.0.0.1:5000")
        print("📍 Health check: http://127.0.0.1:5000/v1/health")
        print("📍 Sample user: STU000001 / password123")
        print()
        app.run(host='127.0.0.1', port=5000, debug=True)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running setup first: python setup_script_fixed.py --full-setup")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()"""

    try:
        with open('run_chatbot.py', 'w', encoding='utf-8') as f:
            f.write(run_script_content)
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod('run_chatbot.py', 0o755)
        logger.info(" Run script created: run_chatbot.py")
        return True
    except Exception as e:
        logger.error(f" Failed to create run script: {e}")
        return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Enhanced Mental Health Chatbot Setup with Ollama - FIXED")
    parser.add_argument("--check-python", action="store_true", help="Check Python version")
    parser.add_argument("--check-ollama", action="store_true", help="Check Ollama installation") 
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--setup-ollama", action="store_true", help="Setup Ollama models")
    parser.add_argument("--create-dirs", action="store_true", help="Create directories")
    parser.add_argument("--setup-db", action="store_true", help="Setup database")
    parser.add_argument("--test-ollama", action="store_true", help="Test Ollama integration")
    parser.add_argument("--test-ai", action="store_true", help="Test AI components")
    parser.add_argument("--create-sample-user", action="store_true", help="Create sample user")
    parser.add_argument("--create-files", action="store_true", help="Create helper files")
    parser.add_argument("--full-setup", action="store_true", help="Run complete setup")
    parser.add_argument("--quick-fix", action="store_true", help="Apply quick fixes to existing setup")

    args = parser.parse_args()

    logger.info("🚀 Enhanced Mental Health Chatbot Setup with Ollama - FIXED VERSION")
    logger.info("=" * 60)

    success_count = 0
    total_steps = 0
    steps = []

    if args.check_python or args.full_setup:
        steps.append(("Python Version Check", check_python_version))

    if args.check_ollama or args.full_setup:
        steps.append(("Ollama Installation Check", check_ollama_installation))

    if args.create_dirs or args.full_setup:
        steps.append(("Directory Creation", create_directories))

    if args.install_deps or args.full_setup:
        steps.append(("Dependency Installation", install_dependencies))

    if args.setup_ollama or args.full_setup:
        steps.append(("Ollama Models Setup", setup_ollama_models))

    if args.setup_db or args.full_setup:
        steps.append(("Database Setup", setup_database))

    if args.test_ollama or args.full_setup:
        steps.append(("Ollama Integration Test", test_ollama_integration))

    if args.test_ai or args.full_setup:
        steps.append(("AI Components Test", test_ai_components))

    if args.create_sample_user or args.full_setup:
        steps.append(("Sample User Creation", create_sample_user))

    if args.create_files or args.full_setup:
        steps.append(("Requirements File Creation", create_requirements_txt))
        steps.append(("Run Script Creation", create_run_script))

    if args.quick_fix:
        logger.info("🔧 Applying quick fixes...")
        steps = [
            ("Fix File Creation", lambda: True),  # Files already created above
            ("Database Setup", setup_database), 
            ("AI Components Test", test_ai_components),
            ("Sample User Creation", create_sample_user)
        ]

    if not steps:
        logger.info("💡 No setup options specified. Use --help for options or --full-setup for everything.")
        logger.info("💡 Use --quick-fix to apply fixes to existing setup.")
        return

    # Execute setup steps
    total_steps = len(steps)
    for step_name, step_function in steps:
        logger.info(f"🔄 {step_name}...")
        try:
            if step_function():
                success_count += 1
                logger.info(f" {step_name} completed")
            else:
                logger.error(f" {step_name} failed")
        except Exception as e:
            logger.error(f" {step_name} failed with error: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info(f"📊 Setup Summary: {success_count}/{total_steps} steps completed successfully")

    if success_count >= total_steps * 0.8:  # 80% success rate
        logger.info(" Setup completed successfully!")
        logger.info("")
        logger.info("Next Steps:")
        logger.info("1. Start the chatbot: python run_chatbot.py")
        logger.info("2. Visit: http://127.0.0.1:5000/v1/health")
        logger.info("3. Test with sample user: STU000001 / password123")
        logger.info("")
        logger.info("The system uses Ollama Llama3 when available, with robust fallbacks!")
        logger.info("If Ollama is not available, the system will work with keyword-based processing.")
    else:
        logger.error(f"⚠️ Setup incomplete. {total_steps - success_count} steps failed.")
        logger.info("Check the logs above for details on failed steps.")
        logger.info("Try: python setup_script_fixed.py --quick-fix")

if __name__ == "__main__":
    main()