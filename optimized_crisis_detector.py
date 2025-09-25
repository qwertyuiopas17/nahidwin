"""
Optimized Crisis Detector - Accurate Crisis Detection with Minimal False Positives
Advanced implementation with contextual analysis and improved accuracy
"""

import re
import logging
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

# Try to import advanced NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class OptimizedCrisisDetector:
    """
    Advanced crisis detection with high accuracy and minimal false positives.
    Uses multi-layered approach: keywords, context, semantic analysis, and conversation history.
    """

    def __init__(self, model_path: str = None, load_semantic_model: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Initialize semantic model for advanced detection
        self.sentence_model = None
        self.use_semantic = False
        
        if load_semantic_model and HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.use_semantic = True
                self.logger.info("✅ Semantic model loaded for enhanced crisis detection")
            except Exception as e:
                self.logger.warning(f"Could not load semantic model: {e}")
        
        # CRITICAL keywords that indicate immediate crisis (highest confidence)
        self.critical_crisis_keywords = [
            'kill myself', 'want to kill myself', 'going to kill myself', 'plan to kill myself',
            'end my life', 'want to end my life', 'going to end my life', 'ready to end my life',
            'commit suicide', 'plan to die', 'taking my own life', 'going to die today',
            'better off dead', 'world better without me', 'not worth living',
            'final goodbye', 'this is goodbye', 'ending it tonight', 'ending it today',
            'have the pills', 'have the rope', 'have the gun', 'ready to jump',
            'writing my will', 'saying goodbye', 'can\'t go on anymore'
        ]
        
        # Strong crisis indicators (high confidence)
        self.strong_crisis_keywords = [
            'suicidal thoughts', 'thinking about suicide', 'suicide plan', 'want to disappear',
            'ending it all', 'cant go on living', 'no reason to live', 'life has no meaning',
            'tired of living', 'ready to give up', 'nothing left for me', 'pointless existence',
            'everyone would be better off', 'burden to everyone', 'cant take the pain',
            'permanent solution', 'escape this pain', 'make it stop forever'
        ]
        
        # Self-harm indicators
        self.self_harm_keywords = [
            'cut myself', 'hurt myself', 'harm myself', 'cutting myself',
            'self injury', 'want to cut', 'need to cut', 'deserve pain',
            'punish myself', 'make myself bleed', 'razor blade', 'cutting scars',
            'burning myself', 'hitting myself', 'starving myself'
        ]
        
        # Help-seeking phrases that should NOT trigger crisis (crucial for false positive reduction)
        self.help_seeking_phrases = [
            'help me', 'help me with', 'help me deal with', 'help me understand',
            'what should i do', 'what can i do', 'tell me what to do', 'give me advice',
            'need guidance', 'need help', 'can you help', 'show me how', 'teach me',
            'give me steps', 'give me tips', 'advise me', 'guide me', 'support me',
            'want to get help', 'need treatment', 'see a therapist', 'get counseling',
            'find a counselor', 'medication help', 'therapy options', 'professional help',
            'treatment options', 'recovery help', 'healing process', 'getting better'
        ]
        
        # Context words that indicate help-seeking rather than crisis
        self.help_context_words = [
            'cope', 'deal with', 'handle', 'manage', 'overcome', 'get through',
            'steps', 'advice', 'guidance', 'suggestions', 'tips', 'strategies',
            'methods', 'techniques', 'ways to', 'how to', 'learn', 'understand',
            'improve', 'better', 'healing', 'recovery', 'treatment', 'therapy'
        ]
        
        # Academic/informational seeking indicators
        self.academic_indicators = [
            'research', 'study', 'paper', 'assignment', 'project', 'report',
            'statistics', 'information', 'facts', 'definition', 'explain',
            'homework', 'school work', 'presentation', 'essay'
        ]
        
        # Emotional severity indicators (support crisis detection)
        self.severe_emotional_indicators = [
            'hopeless', 'helpless', 'worthless', 'unbearable', 'cant take it',
            'breaking point', 'give up', 'there is no hope', 'lost all hope', 'empty inside',
            'numb', 'dead inside', 'soul crushing', 'overwhelmed', 'drowning',
            'suffocating', 'trapped', 'desperate', 'agony', 'torment'
        ]
        
        # Temporal urgency indicators
        self.urgency_indicators = [
            'tonight', 'today', 'right now', 'this moment', 'soon', 'tomorrow',
            'this weekend', 'cant wait', 'immediately', 'urgent', 'emergency'
        ]
        
        # Risk level thresholds (carefully calibrated)
        self.risk_thresholds = {
            'crisis': 0.85,     # Immediate intervention needed (Higher threshold for safety)
            'high': 0.60,       # Close monitoring needed  
            'medium': 0.40,     # Enhanced support
            'low': 0.20         # Standard support
        }
        
        # Conversation pattern tracking
        self.escalation_patterns = [
            'getting worse', 'cant handle', 'too much', 'breaking down',
            'losing control', 'falling apart', 'spiraling', 'deteriorating'
        ]
        
        if self.use_semantic:
            self._build_semantic_embeddings()
        
        if model_path and os.path.exists(model_path):
            self.load_detector(model_path)
    
    def _build_semantic_embeddings(self):
        # (This method remains unchanged)
        try:
            crisis_examples = [
                "I want to kill myself tonight", "I'm going to end my life",
                "I have a plan to commit suicide", "I can't go on living anymore",
                "The world would be better without me", "I'm ready to die",
                "This pain is unbearable and I want it to stop forever"
            ]
            help_seeking_examples = [
                "Help me deal with suicidal thoughts", "What should I do about feeling depressed",
                "I need advice on handling anxiety", "Can you help me cope with these feelings",
                "Give me strategies for managing stress", "How can I get help for my mental health"
            ]
            crisis_embeddings = self.sentence_model.encode(crisis_examples)
            help_embeddings = self.sentence_model.encode(help_seeking_examples)
            self.crisis_embedding = np.mean(crisis_embeddings, axis=0)
            self.help_seeking_embedding = np.mean(help_embeddings, axis=0)
            self.logger.info("✅ Semantic embeddings built successfully")
        except Exception as e:
            self.logger.error(f"Failed to build semantic embeddings: {e}")
            self.use_semantic = False
    
    def detect_crisis_with_context(self,
                                 text: str,
                                 conversation_history: List[str] = None,
                                 user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main crisis detection method with advanced contextual analysis.
        """
        analysis_components = {}
        text_lower = text.lower().strip()
        # --- FIX START: Add a pre-filter for inappropriate/off-topic messages ---
        inappropriate_keywords = [
    # sexual/nsfw
    'horny', 'h0rny', 'h0rn¥', 'sexy', 's3xy', 'sex', 's3x', 's*x', 'seggs',
    'nsfw', '18+', 'adult content', 'explicit', 'explicit content',
    'porn', 'pr0n', 'p0rn', 'xxx',
    'nude', 'nudes', 'nudity', 'naked', 'n*d3', 'n00ds', 'send nudes',
    'sext', 'sexting', 'onlyfans', '0nlyfans', 'fetish', 'erotic', 'lewd',
    'thirst trap', 'hookup', 'fwb', 'sugar daddy', 'camgirl', 'cam site',
    'strip', 'strip club',


    # harassment/profanity (censored)
    'fuck', 'fuck off', 'motherfucker', 'shit', 'bitch', 'asshole',
    'dick', 'punk', 'bastard', 'd**n', 'ffs', 'stfu', 'shut up',
    'loser', 'idiot', 'stupid', 'dumb', 'clown', 'trash', 'worthless', 'go away',

    # hate/extremism (contextual — avoid blocking neutral identity words)
    'racist', 'racism', 'hate speech', 'homophobic', 'transphobic',
    'bigot', 'white supremacy', 'white supremacist', 'neo-n*zi', 'supremacist',

    # illegal/drugs/weapons
    'illegal', 'contraband', 'drug dealing', 'buy drugs', 'sell drugs', 'narcotics',
    'weed', 'marijuana', 'cannabis', 'coke', 'cocaine', 'meth', 'heroin',
    'opioids', 'fentanyl', 'gun', 'firearm', 'weapon', 'ammo',

    # spam/scam/exploitation
    'scam', 'scamming', 'phishing', 'fraud', 'fraudulent',
    'malware', 'spyware', 'keylogger', 'ransomware',
    'hacking', 'hack tool', 'ddos', 'dox', 'doxx', 'blackmail',

    # sensitive/graphic
    'gore', 'graphic content', 'violent content',
    'sexual assault',
]

        if any(keyword in text_lower for keyword in inappropriate_keywords):
            self.logger.info("Crisis detection bypassed due to inappropriate content.")
            return {
            'is_crisis': False, 'risk_level': 'none', 'crisis_score': 0.0,
            'analysis': 'Inappropriate or off-topic content, crisis analysis bypassed.'}
    
        

        # --- FIX: Step 1: Immediate check for critical keywords. Safety First. ---
        critical_score, critical_matches = self._check_critical_crisis_keywords(text_lower)
        analysis_components['critical_keywords'] = {'score': critical_score, 'matches': critical_matches}
        
        # If a critical keyword is found, immediately flag as a crisis and bypass nuanced scoring.
        if critical_score >= 1.0:
            final_score = 0.95  # Assign a very high score to ensure 'crisis' level
            risk_level, is_crisis = self._determine_risk_level(final_score)
            analysis_report = self._generate_analysis_report(risk_level, final_score, analysis_components, text_lower)
            
            return {
                'is_crisis': is_crisis,
                'risk_level': risk_level,
                'crisis_score': round(final_score, 3),
                'confidence': 0.99,
                'components': analysis_components,
                'analysis': analysis_report,
                'detected_indicators': self._get_all_detected_indicators(critical_matches, [], []),
                'recommendations': self._generate_recommendations(risk_level, is_crisis),
                'timestamp': datetime.now().isoformat()
            }

        # --- Proceed with nuanced analysis ONLY if no critical keywords were found ---
        
        # Step 2: Advanced semantic analysis
        semantic_score = 0.0
        if self.use_semantic:
            semantic_score = self._semantic_crisis_analysis(text)
            analysis_components['semantic_analysis'] = {'score': semantic_score}
        
        # Step 3: Check for strong crisis indicators
        strong_score, strong_matches = self._check_strong_crisis_keywords(text_lower)
        
        # --- FIX: Apply help-seeking reduction ONLY to strong (non-critical) keywords ---
        help_seeking_score = self._assess_help_seeking_context(text_lower)
        if help_seeking_score > 0.6 and strong_score > 0:
            strong_score *= 0.4 # Significantly reduce score for phrases like "help with suicidal thoughts"
            analysis_components['help_seeking_adjustment'] = {
                'applied': True,
                'reduction_factor': 0.6,
                'reason': 'Strong help-seeking context detected, reducing non-critical risk score.'
            }
        analysis_components['strong_indicators'] = {'score': strong_score, 'matches': strong_matches}
        
        # Step 4: Self-harm assessment
        self_harm_score, harm_matches = self._check_self_harm_keywords(text_lower)
        analysis_components['self_harm'] = {'score': self_harm_score, 'matches': harm_matches}
        
        # Step 5: Emotional severity analysis
        emotional_score = self._analyze_severe_emotions(text_lower)
        analysis_components['emotional_severity'] = {'score': emotional_score}
        
        # Step 6: Temporal urgency assessment
        urgency_score = self._assess_temporal_urgency(text_lower)
        analysis_components['temporal_urgency'] = {'score': urgency_score}
        
        # Step 7: Conversation history analysis
        history_score = 0.0
        if conversation_history:
            history_score = self._analyze_conversation_history(conversation_history, text_lower)
            analysis_components['history_escalation'] = {'score': history_score}
        
        # Step 8: User context analysis
        context_multiplier = 1.0
        if user_context:
            context_multiplier = self._analyze_user_context(user_context)
            analysis_components['user_context'] = {'multiplier': context_multiplier}
        
        # Step 9: Calculate weighted final score
        weights = {
            'semantic': 0.20,
            'strong': 0.40,
            'self_harm': 0.45,
            'emotional': 0.10,
            'urgency': 0.15,
            'history': 0.10
        }
        
        final_score = (
            semantic_score * weights['semantic'] +
            strong_score * weights['strong'] +
            self_harm_score * weights['self_harm'] +
            emotional_score * weights['emotional'] +
            urgency_score * weights['urgency'] +
            history_score * weights['history']
        ) * context_multiplier
        
        # Step 10: Determine risk level and crisis status
        risk_level, is_crisis = self._determine_risk_level(final_score)
        
        # Step 11: Generate comprehensive analysis
        analysis_report = self._generate_analysis_report(
            risk_level, final_score, analysis_components, text_lower
        )
        
        return {
            'is_crisis': is_crisis,
            'risk_level': risk_level,
            'crisis_score': round(final_score, 3),
            'confidence': min(final_score + 0.2, 1.0),
            'components': analysis_components,
            'analysis': analysis_report,
            'detected_indicators': self._get_all_detected_indicators(
                critical_matches, strong_matches, harm_matches
            ),
            'recommendations': self._generate_recommendations(risk_level, is_crisis),
            'timestamp': datetime.now().isoformat()
        }

    def _check_critical_crisis_keywords(self, text: str) -> Tuple[float, List[str]]:
        """Check for critical crisis keywords with exact matching"""
        matches = []
        for keyword in self.critical_crisis_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                matches.append(keyword)
        
        score = 1.0 if matches else 0.0
        return score, matches

    # All other private methods (_semantic_crisis_analysis, _check_strong_crisis_keywords, etc.)
    # remain unchanged from your original file. I am omitting them here for brevity,
    # but you should keep them in your file. I will include them below just in case.

    def _semantic_crisis_analysis(self, text: str) -> float:
        """Advanced semantic analysis for crisis detection"""
        if not self.use_semantic or not hasattr(self, 'crisis_embedding'):
            return 0.0
        
        try:
            text_embedding = self.sentence_model.encode([text])[0]
            
            # Calculate similarity to crisis patterns
            crisis_similarity = np.dot(text_embedding, self.crisis_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(self.crisis_embedding)
            )
            
            # Calculate similarity to help-seeking patterns
            help_similarity = np.dot(text_embedding, self.help_seeking_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(self.help_seeking_embedding)
            )
            
            # Return normalized score favoring crisis over help-seeking
            semantic_score = max(0, crisis_similarity - help_similarity * 0.5)
            return min(semantic_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Semantic analysis error: {e}")
            return 0.0

    def _check_strong_crisis_keywords(self, text: str) -> Tuple[float, List[str]]:
        """Check for strong crisis indicators"""
        score = 0.0
        matches = []
        
        for keyword in self.strong_crisis_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                score += 0.6
                matches.append(keyword)
        
        # Bonus for multiple indicators
        if len(matches) > 1:
            score += 0.2
        
        return min(score, 1.0), matches
    
    def _check_self_harm_keywords(self, text: str) -> Tuple[float, List[str]]:
        """Check for self-harm indicators"""
        score = 0.0
        matches = []
        
        for keyword in self.self_harm_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                score = 0.7  # Self-harm is serious
                matches.append(keyword)
                break
        
        return score, matches
    
    def _assess_help_seeking_context(self, text: str) -> float:
        """
        Assess if message is in help-seeking context.
        Critical for preventing false positives.
        """
        
        help_score = 0.0
        
        # Check for explicit help-seeking phrases
        for phrase in self.help_seeking_phrases:
            if phrase in text:
                help_score += 0.3
        
        # Check for help-seeking context words
        help_context_count = sum(1 for word in self.help_context_words if word in text)
        help_score += help_context_count * 0.1
        
        # Check for academic/informational indicators
        academic_count = sum(1 for word in self.academic_indicators if word in text)
        help_score += academic_count * 0.15
        
        # Check for question patterns
        help_patterns = [
            r'how (do|can) i',
            r'what (should|can) i do',
            r'can you (help|tell|show|teach)',
            r'i need (help|advice|guidance)',
            r'please (help|tell|show) me',
            r'give me (advice|tips|steps)',
            r'what are (ways|methods|strategies)'
        ]
        
        for pattern in help_patterns:
            if re.search(pattern, text):
                help_score += 0.25
        
        return min(help_score, 1.0)
    
    def _analyze_severe_emotions(self, text: str) -> float:
        """Analyze severity of emotional distress"""
        score = 0.0
        matches = 0
        
        for indicator in self.severe_emotional_indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', text):
                score += 0.15
                matches += 1
        
        # Bonus for multiple severe emotions
        if matches >= 3:
            score += 0.2
        elif matches >= 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_temporal_urgency(self, text: str) -> float:
        """Assess temporal urgency indicators"""
        score = 0.0
        
        for indicator in self.urgency_indicators:
            if indicator in text:
                score += 0.2
                break  # Any urgency indicator is significant
        
        return min(score, 1.0)
    
    def _analyze_conversation_history(self, history: List[str], current_text: str) -> float:
        """Analyze conversation history for escalation patterns"""
        if not history or len(history) < 2:
            return 0.0
        
        escalation_score = 0.0
        
        # Look at last 3 messages for escalation
        recent_messages = history[-3:]
        
        for message in recent_messages:
            message_lower = message.lower()
            
            # Check for crisis escalation
            for keyword in self.critical_crisis_keywords:
                if keyword in message_lower:
                    escalation_score += 0.15
            
            for keyword in self.escalation_patterns:
                if keyword in message_lower:
                    escalation_score += 0.1
        
        # Check if current message shows escalation from history
        if escalation_score > 0 and any(keyword in current_text for keyword in self.critical_crisis_keywords):
            escalation_score += 0.2
        
        return min(escalation_score, 0.4)  # History contributes at most 40%
    
    def _analyze_user_context(self, user_context: Dict[str, Any]) -> float:
        """Analyze user context for risk multipliers"""
        multiplier = 1.0
        
        # Previous crisis history
        if user_context.get('previous_crises', 0) > 0:
            multiplier += 0.2
        
        # Current risk level
        risk_level = user_context.get('current_risk_level', 'low')
        risk_multipliers = {
            'high': 1.3,
            'medium': 1.1,
            'low': 1.0,
            'crisis': 1.5
        }
        multiplier *= risk_multipliers.get(risk_level, 1.0)
        
        # Recent crisis events
        recent_crisis_date = user_context.get('last_crisis_date')
        if recent_crisis_date:
            try:
                if isinstance(recent_crisis_date, str):
                    recent_crisis_date = datetime.fromisoformat(recent_crisis_date)
                
                days_since_crisis = (datetime.now() - recent_crisis_date).days
                if days_since_crisis < 7:  # Within a week
                    multiplier += 0.3
                elif days_since_crisis < 30:  # Within a month
                    multiplier += 0.1
            except:
                pass
        
        return min(multiplier, 2.0)  # Cap at 2x multiplier
    
    def _determine_risk_level(self, score: float) -> Tuple[str, bool]:
        """Determine risk level and crisis status"""
        if score >= self.risk_thresholds['crisis']:
            return 'crisis', True
        elif score >= self.risk_thresholds['high']:
            return 'high', True
        elif score >= self.risk_thresholds['medium']:
            return 'medium', False # Medium risk is not an active crisis for intervention
        else:
            return 'low', False
    
    def _generate_analysis_report(self, risk_level: str, score: float, 
                                components: Dict, text: str) -> str:
        """Generate detailed analysis report"""
        
        if risk_level == 'crisis':
            return f"CRITICAL RISK DETECTED (score: {score:.3f}). Immediate professional intervention required. Multiple crisis indicators present."
        
        elif risk_level == 'high':
            return f"HIGH RISK situation identified (score: {score:.3f}). Enhanced monitoring and support needed. Consider professional referral."
        
        elif risk_level == 'medium':
            return f"MODERATE distress detected (score: {score:.3f}). Provide supportive intervention and monitor closely."
        
        else:
            return f"LOW RISK assessment (score: {score:.3f}). Standard supportive response appropriate."
    
    def _get_all_detected_indicators(self, critical_matches: List[str], 
                                   strong_matches: List[str], 
                                   harm_matches: List[str]) -> List[str]:
        """Get comprehensive list of detected indicators"""
        indicators = []
        
        for match in critical_matches:
            indicators.append(f"CRITICAL: {match}")
        
        for match in strong_matches[:3]:  # Limit to top 3
            indicators.append(f"Strong: {match}")
        
        for match in harm_matches:
            indicators.append(f"Self-harm: {match}")
        
        return indicators
    
    def _generate_recommendations(self, risk_level: str, is_crisis: bool) -> List[str]:
        """Generate specific recommendations based on risk level"""
        
        if is_crisis: # Covers 'crisis' and 'high' risk levels
            return [
                "IMMEDIATE ACTION REQUIRED",
                "Provide crisis intervention response",
                "Offer emergency hotline numbers",
                "Document crisis event thoroughly"
            ]
        
        elif risk_level == 'medium':
            return [
                "Provide supportive response with concrete help",
                "Offer coping strategies and resources",
                "Check in within 2-3 days",
                "Monitor for escalation patterns"
            ]
        
        else:
            return [
                "Provide standard supportive response",
                "Offer relevant coping techniques",
                "Continue regular monitoring"
            ]

    def save_detector(self, filepath: str):
        # (This method remains unchanged)
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            config = {
                'critical_crisis_keywords': self.critical_crisis_keywords,
                'strong_crisis_keywords': self.strong_crisis_keywords,
                'self_harm_keywords': self.self_harm_keywords,
                'help_seeking_phrases': self.help_seeking_phrases,
                'help_context_words': self.help_context_words,
                'severe_emotional_indicators': self.severe_emotional_indicators,
                'urgency_indicators': self.urgency_indicators,
                'risk_thresholds': self.risk_thresholds,
                'escalation_patterns': self.escalation_patterns,
                'academic_indicators': self.academic_indicators,
                'use_semantic': self.use_semantic
            }
            if hasattr(self, 'crisis_embedding'):
                config['crisis_embedding'] = self.crisis_embedding.tolist()
                config['help_seeking_embedding'] = self.help_seeking_embedding.tolist()
            with open(filepath, 'wb') as f:
                pickle.dump(config, f)
            self.logger.info(f"✅ Crisis detector configuration saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save crisis detector: {e}")
    
    def load_detector(self, filepath: str):
        # (This method remains unchanged)
        try:
            with open(filepath, 'rb') as f:
                config = pickle.load(f)
            self.critical_crisis_keywords = config.get('critical_crisis_keywords', self.critical_crisis_keywords)
            self.strong_crisis_keywords = config.get('strong_crisis_keywords', self.strong_crisis_keywords)
            self.self_harm_keywords = config.get('self_harm_keywords', self.self_harm_keywords)
            self.help_seeking_phrases = config.get('help_seeking_phrases', self.help_seeking_phrases)
            self.help_context_words = config.get('help_context_words', self.help_context_words)
            self.severe_emotional_indicators = config.get('severe_emotional_indicators', self.severe_emotional_indicators)
            self.urgency_indicators = config.get('urgency_indicators', self.urgency_indicators)
            self.risk_thresholds = config.get('risk_thresholds', self.risk_thresholds)
            self.escalation_patterns = config.get('escalation_patterns', self.escalation_patterns)
            self.academic_indicators = config.get('academic_indicators', self.academic_indicators)
            if config.get('crisis_embedding') and config.get('help_seeking_embedding'):
                self.crisis_embedding = np.array(config['crisis_embedding'])
                self.help_seeking_embedding = np.array(config['help_seeking_embedding'])
            self.logger.info(f"✅ Crisis detector configuration loaded from {filepath}")
        except FileNotFoundError:
            self.logger.warning(f"Crisis detector config file not found: {filepath}. Using defaults.")
        except Exception as e:
            self.logger.error(f"Error loading crisis detector config: {e}. Using defaults.")

# Alias for backward compatibility
ImprovedCrisisDetector = OptimizedCrisisDetector