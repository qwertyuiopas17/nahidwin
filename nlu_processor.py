"""

Progressive NLU Processor - ENHANCED FIXED VERSION

Advanced Natural Language Understanding for Mental Health with Ollama integration
and comprehensive functionality restored from the original model

"""

import os
import pickle
import logging
import re
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import threading

# Try to import advanced NLP libraries with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from api_ollama_integration import ollama_llama3
    HAS_OLLAMA = ollama_llama3.is_available if ollama_llama3 else False
except ImportError:
    HAS_OLLAMA = False
    ollama_llama3 = None
class ProgressiveNLUProcessor:
    """
    Advanced NLU processor with Ollama Llama 3 integration for enhanced mental health understanding.
    Includes all methods expected by the main chatbot application.
    """

    def __init__(self, model_path: str = None, ollama_model: str = "phi"):
        self.logger = logging.getLogger(__name__)
        self.ollama_model = ollama_model
        self.use_ollama = False
        self._lock = threading.RLock()
        
        # Try to connect to Ollama
        if HAS_OLLAMA and ollama_llama3:
             try:
# Check if the API service is available
                 if ollama_llama3.is_available:
                      self.use_ollama = True
                      self.logger.info(f"✅ API connection successful. Using API model for NLU processing.")
                 else:
                      self.use_ollama = False
                      self.logger.warning(f"⚠️ API service not available. Falling back to keyword-based NLU.")
             except Exception as e:
                  self.logger.warning(f"⚠️ Could not connect to API service. Falling back to keyword-based NLU. Error: {e}")
                  self.use_ollama = False

        # Initialize semantic model for enhanced understanding (optional)
        self.sentence_model = None
        self.use_semantic = False
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.use_semantic = True
                self.logger.info("✅ Semantic model loaded for enhanced NLU")
            except Exception as e:
                self.logger.warning(f"Could not load semantic model: {e}")

        # Enhanced mental health intent categories
        self.intent_categories = {
            'depression_symptoms': {
                'keywords': ['depressed', 'sad', 'empty', 'hopeless', 'worthless', 'meaningless', 'no motivation', 'tired all the time', 'hate myself', 'burden', 'no energy', 'lost interest', 'numb inside'],
                'severity_indicators': {'mild': ['sad', 'down'], 'moderate': ['depressed', 'hopeless'], 'severe': ['hate myself', 'burden', 'better off without me']}
            },
            'anxiety_panic': {
                'keywords': ['anxious', 'worried', 'panic', 'nervous', 'overwhelmed', 'racing heart', 'cant breathe', 'panic attack', 'racing thoughts', 'on edge'],
                'severity_indicators': {'mild': ['nervous', 'worried'], 'moderate': ['anxious', 'overwhelmed'], 'severe': ['panic attack', 'cant breathe']}
            },
            'bullying_harassment': {
                'keywords': ['bullied', 'bullying', 'picked on', 'harassed', 'excluded', 'mean to me', 'threatening me', 'cyber bullying'],
                'severity_indicators': {'mild': ['teasing', 'exclusion'], 'moderate': ['bullying', 'harassment'], 'severe': ['physical violence', 'serious threats']}
            },
            'academic_pressure': {
                'keywords': ['failing', 'grades', 'homework', 'study', 'exam', 'test', 'school stress', 'academic pressure', 'too much work', 'falling behind'],
                'severity_indicators': {'mild': ['stressed about grades', 'homework pressure'], 'moderate': ['academic pressure', 'failing classes'], 'severe': ['academic burnout', 'suicidal over grades']}
            },
            'family_conflicts': {
                'keywords': ['parents fighting', 'family problems', 'arguing with parents', 'family stress', 'toxic family', 'abusive parent'],
                'severity_indicators': {'mild': ['family disagreements'], 'moderate': ['constant fighting', 'toxic family'], 'severe': ['abusive family', 'unsafe at home']}
            },
            'social_anxiety': {
                'keywords': ['social anxiety', 'shy', 'awkward', 'cant make friends', 'nervous around people', 'judged by others', 'avoid social events'],
                'severity_indicators': {'mild': ['shy', 'nervous'], 'moderate': ['social anxiety', 'avoid events'], 'severe': ['complete social isolation']}
            },
            'loneliness_isolation': {
                'keywords': ['lonely', 'alone', 'no friends', 'isolated', 'no one to talk to', 'disconnected', 'breakup', 'heartbroken'],
                'severity_indicators': {'mild': ['sometimes lonely'], 'moderate': ['very lonely', 'no close friends'], 'severe': ['complete isolation']}
            },
            'self_harm': {
                'keywords': ['cut myself', 'hurt myself', 'harm myself', 'cutting', 'self injury', 'want to cut', 'deserve pain', 'punish myself'],
                'severity_indicators': {'mild': ['feeling urges'], 'moderate': ['regular self harm'], 'severe': ['daily self harm', 'life threatening']}
            },
            'crisis_situation': {
                'keywords': ['kill myself', 'suicide', 'end my life', 'want to die', 'suicidal thoughts', 'better off dead', 'ending it all', 'have a plan'],
                'severity_indicators': {'mild': ['passive suicidal thoughts'], 'moderate': ['active suicidal thoughts'], 'severe': ['imminent suicide risk', 'have a plan']}
            },
            'help_seeking': {
                'keywords': ['help me', 'what should i do', 'give me advice', 'need help', 'can you help', 'show me how', 'coping strategies', 'need guidance'],
                'severity_indicators': {'mild': ['looking for tips'], 'moderate': ['need help', 'guidance needed'], 'severe': ['urgent help needed']}
            },
            'sleep_problems': {
                'keywords': ['cant sleep', 'insomnia', 'nightmares', 'tired but cant sleep', 'exhausted', 'sleep schedule messed up'],
                'severity_indicators': {'mild': ['occasional sleeplessness'], 'moderate': ['chronic insomnia'], 'severe': ['no sleep for days']}
            },
            'inappropriate_content': {
                'keywords': ['horny', 'sexy', 'hot', 'sexual', 'sex', 'hookup', 'dating', 'romantic', 'drug dealer', 'buy drugs', 'violence', 'hurt someone', 'kill someone'],
                'severity_indicators': {'mild': ['dating', 'romantic'], 'moderate': ['horny', 'sexual thoughts'], 'severe': ['explicit sexual content', 'violence']}
            },
            'out_of_scope': {
                'keywords': ['what is 10+10', 'do you like', 'your name', 'who are you', 'weather', 'news', 'sports', 'movies', 'music', 'what can you do'],
                'severity_indicators': {} # No severity for this
},
            'general_support': {
                'keywords': ['need someone to talk', 'feeling overwhelmed', 'struggling', 'need support', 'feeling lost', 'confused', 'emotional support'],
                'severity_indicators': {'mild': ['need to talk'], 'moderate': ['overwhelmed', 'struggling'], 'severe': ['crisis level overwhelm']}
            }
        }

        self.conversation_stages = [
            'initial_contact', 'understanding', 'trust_building', 'gentle_help_offering',
            'method_suggestion', 'method_follow_up', 'ongoing_support', 'crisis_intervention',
            'professional_referral'
        ]

        # Build semantic embeddings if available
        if self.use_semantic:
            self._build_semantic_embeddings()

        # Load saved model if available
        if model_path and os.path.exists(model_path):
            self.load_nlu_model(model_path)

    def _build_semantic_embeddings(self):
        """Build semantic embeddings for each intent category"""
        try:
            self.category_embeddings = {}
            for category, data in self.intent_categories.items():
                # Use keywords to create pseudo-sentences for embedding
                keywords = data['keywords'][:5]  # Use top 5 keywords
                pseudo_sentences = [f"I feel {keyword}" for keyword in keywords]
                
                # Create embeddings
                embeddings = self.sentence_model.encode(pseudo_sentences)
                
                # Use mean embedding as category representation
                self.category_embeddings[category] = np.mean(embeddings, axis=0)
            
            self.logger.info("✅ Semantic embeddings built for all intent categories")
        except Exception as e:
            self.logger.error(f"Failed to build semantic embeddings: {e}")
            self.use_semantic = False

    # THIS IS THE CORRECT LINE
    def understand_user_intent(self, user_message: str, conversation_history: List[Dict[str, Any]] = None, excluded_intents: List[str] = None) -> Dict[str, Any]:
        """
        Processes a user's message to understand intent, severity, and other NLU metrics,
        using Ollama Llama 3 if available, otherwise falling back to a robust keyword-based system.
        """
        cleaned_message = self._clean_and_preprocess(user_message)
        
        # Immediate override for inappropriate content for safety and focus.
        inappropriate_keywords = ['horny', 'sexy', 'sexual', 'sex', 'hookup']
        if any(keyword in cleaned_message for keyword in inappropriate_keywords):
            return self._generate_inappropriate_content_response()

        # Attempt to use Ollama for primary analysis.
        if self.use_ollama:
            ollama_result = self._get_ollama_analysis(cleaned_message, conversation_history)
            if ollama_result:
                # If Ollama provides a valid result, use it.
                return self._compile_final_analysis(ollama_result, cleaned_message)

        # Fallback to the internal keyword-based system
        self.logger.info(f"Falling back to keyword-based NLU for message: '{cleaned_message[:50]}...'")
        fallback_result = self._get_fallback_analysis(cleaned_message, excluded_intents)
        return self._compile_final_analysis(fallback_result, cleaned_message)

    def _get_ollama_analysis(self, user_message: str, conversation_history: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Uses Ollama with conversation history for a contextual NLU analysis."""
        try:
            history_str = "No previous conversation history."
            if conversation_history:
                history_context = []
                for turn in conversation_history:
                    role = turn.get('role', 'user').title()
                    content = turn.get('content', '')
                    history_context.append(f"{role}: {content}")
                history_str = "\n".join(history_context)
            # Construct a detailed prompt for the LLM.
            prompt = f"""
You are an expert mental health NLU system. Analyze the user's LATEST message based on the provided conversation history. Return ONLY a valid JSON response...
{{
    "primary_intent": "one of: {', '.join(self.intent_categories.keys())}",
    "confidence": 0.85,
    "severity_score": 0.65,
    "emotional_state": {{
        "primary_emotion": "sad/anxious/angry/hopeless/neutral",
        "intensity": 0.7
    }},
    "urgency_level": "low/medium/high/crisis",
    "requires_immediate_help": false,
    "context_entities": {{ "people": [], "places": [], "triggers": [] }},
    "user_needs": ["emotional_support", "coping_strategies"],
    "risk_factors": ["isolation", "suicidal_ideation"],
    "in_scope": true
}}

Guidelines:
- Use the HISTORY to understand the context of short messages (e.g., "no", "yes", "help me").
- The primary_intent should reflect the topic of the ONGOING conversation unless the user clearly changes the subject.
- crisis_situation: Only for explicit suicidal ideation or immediate danger.
- severity_score: 0.0-1.0 (crisis=0.9+, severe=0.7+, moderate=0.5+, mild=0.3+).
- Return ONLY valid JSON, no other text.
HISTORY:
{history_str}

Analyze this message: '{user_message}'
"""

            response_text = ollama_llama3.client.generate_response(prompt, max_tokens=400, temperature=0.3)
            if not response_text:
                return None
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                 json_str = response_text[json_start:json_end]
                 analysis = json.loads(json_str)
            else:
                return None

            # The response content should already be a JSON string.
           
            self.logger.info(f"✅ Ollama NLU analysis successful for: {user_message[:50]}...")
            return analysis

        except Exception as e:
            self.logger.error(f"❌ Ollama NLU analysis failed: {e}")
            return None

    def _get_fallback_analysis(self, message: str, conversation_history: List[Dict[str, Any]] = None, excluded_intents: List[str] = None) -> Dict[str, Any]:
        """Generates NLU analysis using keywords, with a context-aware fallback for short messages."""
        
        # --- FIX START: Handle short, context-dependent messages ---
        # If the message is short (e.g., "no", "ok"), infer intent from the last turn.
        if len(message.split()) <= 2 and conversation_history:
            last_intent = 'general_support'
            # Find the intent from the most recent turn in history
            for turn in reversed(conversation_history):
                if turn.get('intent'):
                    last_intent = turn['intent']
                    break
            
            self.logger.info(f"Short message detected. Inheriting intent '{last_intent}' from conversation history.")
            # Create a minimal analysis, inheriting the previous intent
            analysis = self._comprehensive_intent_detection(message)
            analysis['primary_intent'] = last_intent
            analysis['confidence'] = 0.90 # High confidence it's a contextual follow-up
        else:
            # Otherwise, perform a full keyword-based analysis
            analysis = self._comprehensive_intent_detection(message, excluded_intents)
        emotional_analysis = self._advanced_emotional_analysis(message)
        context_entities = self._extract_comprehensive_context(message)
        urgency_analysis = self._assess_urgency_and_severity(message, analysis)
        scope_validation = self._validate_mental_health_scope(message)
        user_needs = self._identify_comprehensive_user_needs(analysis['primary_intent'], emotional_analysis, urgency_analysis)

        return {
            'primary_intent': analysis['primary_intent'],
            'confidence': analysis['confidence'],
            'severity_score': urgency_analysis['severity_score'],
            'emotional_state': emotional_analysis,
            'urgency_level': urgency_analysis['urgency_level'],
            'requires_immediate_help': urgency_analysis['requires_immediate_help'],
            'context_entities': context_entities,
            'user_needs': user_needs,
            'risk_factors': urgency_analysis['risk_factors'],
            'in_scope': scope_validation['in_scope']
        }
    
    # In nlu_processor.py, inside the ProgressiveNLUProcessor class

    def _safe_float(self, value, default=0.0):
        """Safely convert a value to a float, handling lists or other types."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list) and len(value) > 0:
            # Try to convert the first element of the list
            try:
                return float(value[0])
            except (ValueError, TypeError):
                return default
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default

    def _compile_final_analysis(self, analysis_data: Dict[str, Any], cleaned_message: str) -> Dict[str, Any]:
        """Compiles the final NLU response object from the analysis data."""
         # --- FIX START: Add a safety check for the primary_intent format ---
        primary_intent_value = analysis_data.get('primary_intent', 'general_support')
        if isinstance(primary_intent_value, list) and len(primary_intent_value) > 0:
        # If it's a list, take the first element
           primary_intent_value = primary_intent_value[0]
        elif not isinstance(primary_intent_value, str):
        # If it's something else, default to a string
           primary_intent_value = 'general_support'
    # --- FIX END ---
        analysis = {
            'primary_intent': analysis_data.get('primary_intent', 'general_support'),
            'confidence': analysis_data.get('confidence', 0.5),
            'all_scores': {analysis_data.get('primary_intent', 'general_support'): analysis_data.get('confidence', 0.5)},
            'secondary_intents': analysis_data.get('secondary_intents', [])
        }
        # --- FIX START: Add a safety check for the emotional_state format ---
        emotional_state_data = analysis_data.get('emotional_state', {})
        if not isinstance(emotional_state_data, dict):
        # If it's a string, create a default dictionary structure
           emotional_state_data = {'primary_emotion': str(emotional_state_data), 'intensity': 0.5}
        # --- FIX END ---


        conversation_stage = self._determine_conversation_stage(cleaned_message, analysis)
        response_recommendations = self._generate_response_recommendations(analysis, analysis_data.get('emotional_state', {}), analysis_data, conversation_stage)

        # Sanitize and structure the final output
        return {
            'primary_intent': analysis['primary_intent'],
            'confidence': float(analysis['confidence']),
            'intent_distribution': analysis['all_scores'],
            'secondary_intents': analysis['secondary_intents'],
            'emotional_state': analysis_data.get('emotional_state', {'primary_emotion': 'neutral', 'intensity': 0.0}),
            'emotional_intensity': self._safe_float(analysis_data.get('emotional_state', {}).get('intensity', 0.0)),
            'context_entities': analysis_data.get('context_entities', {}),
            'conversation_stage': conversation_stage,
            'urgency_level': analysis_data.get('urgency_level', 'low'),
            'severity_score': self._safe_float(analysis_data.get('severity_score', 0.0)),
            'risk_factors': analysis_data.get('risk_factors', []),
            'requires_immediate_help': bool(analysis_data.get('requires_immediate_help', False)),
            'in_scope': bool(analysis_data.get('in_scope', True)),
            'user_needs': analysis_data.get('user_needs', ['emotional_support']),
            'response_recommendations': response_recommendations,
            'processing_timestamp': datetime.now().isoformat(),
            'ollama_analysis_used': self.use_ollama
        }

    def _generate_inappropriate_content_response(self) -> Dict[str, Any]:
        """Returns a structured response for inappropriate content."""
        return {
            'primary_intent': 'inappropriate_content',
            'confidence': 0.99,
            'intent_distribution': {'inappropriate_content': 0.99},
            'emotional_state': {'primary_emotion': 'neutral', 'intensity': 0.1},
            'context_entities': {},
            'conversation_stage': 'initial_contact',
            'urgency_level': 'low',
            'severity_score': 0.1,
            'requires_immediate_help': False,
            'risk_factors': [],
            'in_scope': True,
            'user_needs': ['boundary_setting'],
            'response_recommendations': {'response_type': 'redirect'},
            'processing_timestamp': datetime.now().isoformat(),
            'ollama_analysis_used': False
        }

    # --- Fallback Methods (Keyword and Pattern-Based) ---

    def _clean_and_preprocess(self, message: str) -> str:
        """Cleans and standardizes the user's message for analysis."""
        cleaned = message.lower().strip()
        contractions = {
            "can't": "cannot", "won't": "will not", "don't": "do not", "didn't": "did not",
            "i'm": "i am", "you're": "you are", "it's": "it is", "i've": "i have"
        }

        for contraction, expansion in contractions.items():
            cleaned = cleaned.replace(contraction, expansion)
        cleaned = re.sub(r'[^\w\s]', '', cleaned)  # Remove punctuation
        return cleaned

    def _comprehensive_intent_detection(self, message: str, excluded_intents: List[str] = None) -> Dict[str, Any]:
        """Combines keyword and pattern matching for fallback intent detection."""
        keyword_scores = self._enhanced_keyword_intent_detection(message)
        
        if excluded_intents:
            for intent in excluded_intents:
                if intent in keyword_scores:
                    keyword_scores[intent] *= 0.1  # Penalize excluded intents

        if not keyword_scores:
            primary_intent = 'general_support'
            confidence = 0.3
        else:
            primary_intent = max(keyword_scores, key=keyword_scores.get)
            confidence = keyword_scores[primary_intent]

        return {
            'primary_intent': primary_intent,
            'confidence': min(confidence, 1.0),
            'all_scores': keyword_scores
        }

    def _enhanced_keyword_intent_detection(self, message: str) -> Dict[str, float]:
        """Detects intent based on keywords with severity weighting."""
        scores = {}
        for category, data in self.intent_categories.items():
            score = 0.0
            for keyword in data['keywords']:
                if re.search(r'\b' + re.escape(keyword) + r'\b', message):
                    score += 0.1 * len(keyword.split())  # Weight longer phrases more

            for level, indicators in data.get('severity_indicators', {}).items():
                multiplier = {'mild': 1.2, 'moderate': 1.5, 'severe': 2.0}.get(level, 1.0)
                for indicator in indicators:
                    if re.search(r'\b' + re.escape(indicator) + r'\b', message):
                        score *= multiplier

            if score > 0:
                scores[category] = min(score, 1.0)
        return scores

    def _advanced_emotional_analysis(self, message: str) -> Dict[str, Any]:
        """Analyzes the emotional tone of the message."""
        emotions = {
            'sad': ['sad', 'depressed', 'down', 'hopeless', 'empty'],
            'anxious': ['anxious', 'worried', 'nervous', 'scared', 'panic'],
            'angry': ['angry', 'mad', 'frustrated', 'irritated'],
            'neutral': ['okay', 'fine', 'alright']
        }

        detected_emotions = {emotion: sum(1 for keyword in keywords if keyword in message) for emotion, keywords in emotions.items()}
        if any(detected_emotions.values()):
            primary_emotion = max(detected_emotions, key=detected_emotions.get)
            intensity = min(detected_emotions[primary_emotion] * 0.3, 1.0)
        else:
            primary_emotion = 'neutral'
            intensity = 0.0

        return {'primary_emotion': primary_emotion, 'intensity': intensity}

    def _extract_comprehensive_context(self, message: str) -> Dict[str, List[str]]:
        """Extracts basic entities like people and places."""
        context = {'people': [], 'places': []}
        people_keywords = ['mom', 'dad', 'parents', 'friend', 'teacher', 'boss']
        place_keywords = ['school', 'work', 'home']

        context['people'] = [p for p in people_keywords if p in message]
        context['places'] = [p for p in place_keywords if p in message]
        return context

    def _assess_urgency_and_severity(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Assesses urgency and severity based on keywords and intent."""
        intent = analysis['primary_intent']
        base_severity = {
            'crisis_situation': 1.0, 'self_harm': 0.9, 'depression_symptoms': 0.7,
            'anxiety_panic': 0.6, 'bullying_harassment': 0.6, 'family_conflicts': 0.5,
            'academic_pressure': 0.4, 'social_anxiety': 0.4, 'loneliness_isolation': 0.4,
            'sleep_problems': 0.3, 'help_seeking': 0.2, 'general_support': 0.2,
            'inappropriate_content': 0.1
        }.get(intent, 0.2)

        urgency_keywords = ['right now', 'tonight', 'today', 'immediately', 'urgent']
        is_urgent = any(keyword in message for keyword in urgency_keywords)
        urgency_level = 'high' if is_urgent else ('medium' if base_severity > 0.5 else 'low')

        risk_factors = []
        if 'suicide' in message or 'kill myself' in message:
            risk_factors.append('suicidal_ideation')
        if 'cut myself' in message or 'self harm' in message:
            risk_factors.append('self_harm')

        final_severity = base_severity + (0.2 if is_urgent else 0) + (0.1 * len(risk_factors))
        requires_immediate_help = intent == 'crisis_situation' or final_severity >= 0.9

        return {
            'urgency_level': urgency_level,
            'severity_score': min(final_severity, 1.0),
            'risk_factors': risk_factors,
            'requires_immediate_help': requires_immediate_help
        }

    def _determine_conversation_stage(self, message: str, analysis: Dict) -> str:
        """Determines the current stage of the conversation."""
        intent = analysis['primary_intent']
        
        if intent == 'crisis_situation':
            return 'crisis_intervention'
        if intent == 'help_seeking':
            return 'method_suggestion'
        if any(kw in message for kw in ['tried', 'worked', 'helped']):
            return 'method_follow_up'
        return 'understanding'

    def _validate_mental_health_scope(self, message: str) -> Dict[str, Any]:
        """Validates if the message is within the chatbot's scope."""
        is_in_scope = any(intent for intent, data in self.intent_categories.items() 
                         if any(kw in message for kw in data['keywords']))
        return {'in_scope': is_in_scope, 'relevance_score': 1.0 if is_in_scope else 0.1}

    def _identify_comprehensive_user_needs(self, primary_intent: str, emotional_analysis: Dict, urgency_analysis: Dict) -> List[str]:
        """Identifies user needs based on the analysis."""
        needs = {'emotional_support'}
        
        if primary_intent == 'help_seeking':
            needs.add('coping_strategies')
        if urgency_analysis['requires_immediate_help']:
            needs.add('immediate_safety')
        return list(needs)

    def _generate_response_recommendations(self, analysis: Dict, emotional_analysis: Dict, urgency_analysis: Dict, conversation_stage: str) -> Dict[str, Any]:
        """Generates recommendations for the response generator."""
        if urgency_analysis.get('requires_immediate_help'):
            return {'response_type': 'crisis_intervention', 'priority': 'immediate_safety'}
        if conversation_stage == 'method_suggestion':
            return {'response_type': 'skill_teaching', 'priority': 'concrete_help'}
        return {'response_type': 'supportive', 'priority': 'emotional_support'}

    # ============================================================================
    # CRITICAL: Methods expected by the main chatbot application
    # ============================================================================

    def save_nlu_model(self, filepath: str) -> bool:
        """Save NLU model configuration and learned parameters."""
        try:
            with self._lock:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                config = {
                    'intent_categories': self.intent_categories,
                    'conversation_stages': self.conversation_stages,
                    'use_ollama': self.use_ollama,
                    'ollama_model': self.ollama_model,
                    'use_semantic': self.use_semantic,
                    'model_version': '2.1.0',
                    'save_timestamp': datetime.now().isoformat()
                }
                
                # Save semantic embeddings if available
                if hasattr(self, 'category_embeddings') and self.category_embeddings:
                    config['category_embeddings'] = {
                        category: embedding.tolist() 
                        for category, embedding in self.category_embeddings.items()
                    }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(config, f)
                
                self.logger.info(f"✅ NLU model configuration saved to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save NLU model: {e}")
            return False

    def load_nlu_model(self, filepath: str) -> bool:
        """Load NLU model configuration and parameters."""
        try:
            with self._lock:
                with open(filepath, 'rb') as f:
                    config = pickle.load(f)
                
                # Load configurations
                self.intent_categories = config.get('intent_categories', self.intent_categories)
                self.conversation_stages = config.get('conversation_stages', self.conversation_stages)
                self.ollama_model = config.get('ollama_model', self.ollama_model)
                
                # Load semantic embeddings if available
                if 'category_embeddings' in config and self.use_semantic:
                    self.category_embeddings = {
                        category: np.array(embedding) 
                        for category, embedding in config['category_embeddings'].items()
                    }
                
                self.logger.info(f"✅ NLU model configuration loaded from {filepath}")
                return True
                
        except FileNotFoundError:
            self.logger.warning(f"⚠️ NLU model file not found: {filepath}. Using defaults.")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error loading NLU model: {e}. Using defaults.")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            'model_type': 'Progressive NLU Processor with Ollama Integration',
            'version': '2.1.0',
            'ollama_enabled': self.use_ollama,
            'ollama_model': self.ollama_model if self.use_ollama else None,
            'semantic_enabled': self.use_semantic,
            'intent_categories_count': len(self.intent_categories),
            'conversation_stages_count': len(self.conversation_stages),
            'initialized_at': datetime.now().isoformat()
        }

    def validate_configuration(self) -> bool:
        """Validate the current model configuration."""
        try:
            # Check essential components
            if not self.intent_categories:
                self.logger.error("❌ No intent categories defined")
                return False
            
            if not self.conversation_stages:
                self.logger.error("❌ No conversation stages defined")
                return False
            
            # Test basic functionality
            test_result = self.understand_user_intent("I feel sad")
            if not test_result or 'primary_intent' not in test_result:
                self.logger.error("❌ Basic intent detection failed")
                return False
            
            self.logger.info("✅ NLU model configuration is valid")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False

    # Backward compatibility methods
    def analyze_user_message(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Backward compatibility method - alias for understand_user_intent."""
        return self.understand_user_intent(message, context.get('excluded_intents') if context else None)

    def get_intent_confidence(self, message: str, intent: str) -> float:
        """Get confidence score for a specific intent."""
        result = self.understand_user_intent(message)
        return result['intent_distribution'].get(intent, 0.0)

    def is_crisis_detected(self, message: str) -> bool:
        """Quick check if message indicates crisis situation."""
        result = self.understand_user_intent(message)
        return result['requires_immediate_help'] or result['primary_intent'] == 'crisis_situation'