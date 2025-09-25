"""
Progressive Conversation Memory - Advanced User State & Method Tracking System
Enhanced with comprehensive method effectiveness tracking, user progress analysis, and counselor integration
"""

import json
import uuid
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import logging
from collections import defaultdict, deque
import threading
import numpy as np

@dataclass
class ConversationTurn:
    """Individual conversation turn with comprehensive analysis"""
    id: str
    user_message: str
    bot_response: str
    timestamp: datetime
    intent: str
    confidence: float
    severity_score: float
    conversation_stage: str
    emotional_state: Dict[str, Any] = field(default_factory=dict)
    method_suggested: Optional[str] = None
    user_feedback: Optional[str] = None
    response_effectiveness: Optional[float] = None
    user_satisfaction: Optional[int] = None  # 1-5 rating
    context_entities: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class MethodExperience:
    """Detailed tracking of user's experience with a specific method"""
    method_id: str
    method_name: str
    first_suggested: datetime
    times_suggested: int = 0
    times_used: int = 0
    effectiveness_ratings: List[float] = field(default_factory=list)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    context_when_used: List[Dict[str, Any]] = field(default_factory=list)
    last_used: Optional[datetime] = None
    user_notes: List[str] = field(default_factory=list)
    
    @property
    def average_effectiveness(self) -> float:
        return np.mean(self.effectiveness_ratings) if self.effectiveness_ratings else 0.5
    
    @property
    def consistency_score(self) -> float:
        """How consistent the effectiveness ratings are (lower std = more consistent)"""
        if len(self.effectiveness_ratings) < 2:
            return 1.0
        return max(0.0, 1.0 - np.std(self.effectiveness_ratings))
    
    @property
    def usage_frequency(self) -> float:
        """How often the user actually uses the method when suggested"""
        return self.times_used / max(self.times_suggested, 1)

@dataclass 
class UserProfile:
    """Comprehensive user profile with advanced tracking capabilities"""
    user_id: str
    first_interaction: datetime
    last_interaction: datetime
    total_interactions: int = 0
    current_stage: str = 'initial_contact'
    
    # Mental health tracking
    primary_concerns: List[str] = field(default_factory=list)
    concern_severity_history: Dict[str, List[Tuple[datetime, float]]] = field(default_factory=dict)
    severity_trend: List[Tuple[datetime, float]] = field(default_factory=list)
    emotional_patterns: Dict[str, List[Tuple[datetime, float]]] = field(default_factory=dict)
    
    # Method and intervention tracking
    methods_experienced: Dict[str, MethodExperience] = field(default_factory=dict)
    current_method: Optional[str] = None
    method_check_due: Optional[datetime] = None
    preferred_method_types: List[str] = field(default_factory=list)
    avoided_method_types: List[str] = field(default_factory=list)
    
    # Professional help tracking
    counselor_referral_status: str = 'none'  # none, suggested, interested, contacted, booked, attending
    counselor_preferences: Dict[str, Any] = field(default_factory=dict)
    referral_history: List[Dict[str, Any]] = field(default_factory=list)
    referral_date: Optional[datetime] = None
    
    # Crisis and risk management
    risk_level: str = 'low'  # low, medium, high, crisis
    crisis_history: List[Dict[str, Any]] = field(default_factory=list)
    safety_plan: Dict[str, Any] = field(default_factory=dict)
    support_network: List[Dict[str, str]] = field(default_factory=list)
    
    # Progress and outcomes tracking
    improvement_indicators: List[Dict[str, Any]] = field(default_factory=list)
    regression_indicators: List[Dict[str, Any]] = field(default_factory=list)
    goal_progress: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    milestone_achievements: List[Dict[str, Any]] = field(default_factory=list)
    
    # User preferences and personalization
    communication_preferences: Dict[str, str] = field(default_factory=dict)
    response_preferences: Dict[str, Any] = field(default_factory=dict)
    privacy_settings: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.first_interaction, str):
            self.first_interaction = datetime.fromisoformat(self.first_interaction)
        if isinstance(self.last_interaction, str):
            self.last_interaction = datetime.fromisoformat(self.last_interaction)
        if isinstance(self.method_check_due, str):
            self.method_check_due = datetime.fromisoformat(self.method_check_due) if self.method_check_due else None
        if isinstance(self.referral_date, str):
            self.referral_date = datetime.fromisoformat(self.referral_date) if self.referral_date else None
    
    def calculate_overall_progress(self) -> Dict[str, float]:
        """Calculate comprehensive progress metrics"""
        if len(self.severity_trend) < 2:
            return {'trend': 0.0, 'stability': 0.5, 'improvement': 0.0}
        
        # Calculate trend over time
        recent_scores = [score for _, score in self.severity_trend[-5:]]
        early_scores = [score for _, score in self.severity_trend[:5]]
        
        trend = np.mean(early_scores) - np.mean(recent_scores)  # Positive = improvement
        stability = max(0.0, 1.0 - np.std(recent_scores))
        
        # Calculate improvement rate
        if len(self.severity_trend) > 1:
            time_span = (self.severity_trend[-1][0] - self.severity_trend[0][0]).days
            improvement = trend / max(time_span / 30, 1)  # Per month
        else:
            improvement = 0.0
        
        return {
            'trend': trend,
            'stability': stability, 
            'improvement': min(max(improvement, -1.0), 1.0),
            'method_success_rate': self.calculate_method_success_rate()
        }
    
    def calculate_method_success_rate(self) -> float:
        """Calculate overall method success rate"""
        if not self.methods_experienced:
            return 0.5
        
        total_effectiveness = sum(exp.average_effectiveness for exp in self.methods_experienced.values())
        return total_effectiveness / len(self.methods_experienced)
    
    def get_most_effective_methods(self, limit: int = 3) -> List[Tuple[str, float]]:
        """Get most effective methods for this user"""
        method_effectiveness = [
            (method_id, exp.average_effectiveness)
            for method_id, exp in self.methods_experienced.items()
            if exp.effectiveness_ratings
        ]
        method_effectiveness.sort(key=lambda x: x[1], reverse=True)
        return method_effectiveness[:limit]
    
    def get_current_concerns_with_severity(self) -> Dict[str, float]:
        """Get current concerns with their latest severity scores"""
        current_concerns = {}
        for concern in self.primary_concerns:
            if concern in self.concern_severity_history:
                history = self.concern_severity_history[concern]
                if history:
                    current_concerns[concern] = history[-1][1]  # Latest severity
        return current_concerns

class ProgressiveConversationMemory:
    """
    Advanced conversation memory system with comprehensive user tracking,
    method effectiveness analysis, and intelligent personalization
    """

    def __init__(self, save_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.save_path = save_path
        self._lock = threading.RLock()
        
        # Memory stores
        self.user_profiles: Dict[str, UserProfile] = {}
        self.conversation_history: Dict[str, List[ConversationTurn]] = defaultdict(list)
        self.short_term_memory: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Enhanced method library with comprehensive tracking
        self.method_library = {
            'breathing_exercises': {
                'name': 'Deep Breathing Exercises',
                'category': 'anxiety_management',
                'difficulty': 'easy',
                'time_required_minutes': 5,
                'evidence_strength': 'high',
                'effectiveness_rate': 0.8,
                'best_for': ['anxiety_panic', 'academic_pressure', 'general_stress'],
                'contraindications': ['severe_hyperventilation_disorder'],
                'check_in_days': 3,
                'variations': {
                    'basic': '4-4-6 breathing pattern',
                    'advanced': 'Box breathing with visualization',
                    'crisis': 'Emergency calm breathing'
                },
                'instructions': {
                    'preparation': [
                        "Find a quiet, comfortable place where you won't be interrupted",
                        "Sit or lie down in a relaxed position",
                        "Place one hand on your chest, one on your belly"
                    ],
                    'execution': [
                        "Breathe in slowly through your nose for 4 counts",
                        "Hold your breath gently for 4 counts", 
                        "Breathe out slowly through your mouth for 6 counts",
                        "Focus on making the belly hand move more than the chest hand",
                        "Repeat this pattern 5-10 times"
                    ],
                    'completion': [
                        "Notice how your body feels now compared to when you started",
                        "Take a moment to appreciate this time you gave yourself"
                    ]
                },
                'troubleshooting': {
                    'dizziness': 'Breathe more gently and naturally',
                    'cant_focus': 'Count on your fingers or use an app',
                    'feels_weird': 'This is normal - your body is learning to relax'
                },
                'personalization_factors': ['stress_level', 'time_available', 'privacy_level']
            },
            
            'grounding_5432l': {
                'name': '5-4-3-2-1 Grounding Technique',
                'category': 'anxiety_management',
                'difficulty': 'easy',
                'time_required_minutes': 3,
                'evidence_strength': 'moderate',
                'effectiveness_rate': 0.75,
                'best_for': ['anxiety_panic', 'dissociation', 'overwhelming_thoughts'],
                'contraindications': ['severe_sensory_processing_issues'],
                'check_in_days': 2,
                'variations': {
                    'basic': 'Standard 5-4-3-2-1 sequence',
                    'nature': 'Using outdoor/nature elements',
                    'indoor': 'Adapted for indoor environments'
                },
                'instructions': {
                    'preparation': [
                        "Sit or stand comfortably",
                        "Take a deep breath and look around you"
                    ],
                    'execution': [
                        "Name 5 things you can see (look for details like colors, textures)",
                        "Name 4 things you can touch (feel different textures)",
                        "Name 3 things you can hear (near and far sounds)",
                        "Name 2 things you can smell",
                        "Name 1 thing you can taste"
                    ],
                    'completion': [
                        "Take another deep breath",
                        "Notice how you feel more present and grounded"
                    ]
                },
                'troubleshooting': {
                    'cant_smell_anything': 'Think of a recent scent or skip to taste',
                    'getting_anxious': 'Go slower and breathe between each step',
                    'feels_silly': 'This is a proven technique used by professionals'
                },
                'personalization_factors': ['location', 'sensory_preferences', 'anxiety_type']
            },
            
            'behavioral_activation': {
                'name': 'One Small Win Technique',
                'category': 'depression_management',
                'difficulty': 'easy',
                'time_required_minutes': 10,
                'evidence_strength': 'high',
                'effectiveness_rate': 0.7,
                'best_for': ['depression_symptoms', 'low_motivation', 'hopelessness'],
                'contraindications': ['severe_fatigue_disorders'],
                'check_in_days': 5,
                'variations': {
                    'micro': 'Extremely small tasks (2-3 minutes)',
                    'moderate': 'Small but meaningful tasks (10-15 minutes)',
                    'social': 'Tasks involving connection with others'
                },
                'instructions': {
                    'preparation': [
                        "Choose ONE very small, specific task",
                        "Make sure it's something you can complete in 10-15 minutes",
                        "Examples: make bed, send one text, take a shower, eat something"
                    ],
                    'execution': [
                        "Set a timer if it helps you feel less overwhelmed",
                        "Focus only on starting, not on doing it perfectly",
                        "Do just that one thing - nothing more is required",
                        "Pay attention to how you feel while doing it"
                    ],
                    'completion': [
                        "Take a moment to acknowledge what you accomplished",
                        "Recognize this as a genuine victory, no matter how small",
                        "Rest if you need to - that's enough for now"
                    ]
                },
                'troubleshooting': {
                    'too_tired': 'Choose an even smaller task or just planning counts',
                    'feels_pointless': 'Small actions build momentum for bigger changes',
                    'didnt_finish': 'Starting is an achievement - be kind to yourself'
                },
                'personalization_factors': ['energy_level', 'available_time', 'depression_severity']
            },
            
            'safety_planning': {
                'name': 'Personal Safety Planning',
                'category': 'safety_building',
                'difficulty': 'medium',
                'time_required_minutes': 15,
                'evidence_strength': 'high',
                'effectiveness_rate': 0.85,
                'best_for': ['bullying_harassment', 'crisis_situation', 'self_harm'],
                'contraindications': [],
                'check_in_days': 7,
                'variations': {
                    'basic': 'Essential safety contacts and strategies',
                    'comprehensive': 'Detailed plan with multiple scenarios',
                    'digital': 'Online safety and cyberbullying focus'
                },
                'instructions': {
                    'preparation': [
                        "Find a private space where you can think clearly",
                        "Have paper/phone ready to write things down",
                        "Remember: this is about taking care of yourself"
                    ],
                    'execution': [
                        "Identify 2-3 trusted people you can reach out to immediately",
                        "Write down their phone numbers and when they're available",
                        "Think of safe places where you can go if needed",
                        "List activities that help you feel calmer or more grounded",
                        "Keep crisis hotline numbers easily accessible",
                        "Identify warning signs that you need to use this plan"
                    ],
                    'completion': [
                        "Keep this plan somewhere easily accessible",
                        "Share appropriate parts with trusted people",
                        "Review and update it regularly"
                    ]
                },
                'troubleshooting': {
                    'no_trusted_people': 'Start with professional hotlines and counselors',
                    'unsafe_home': 'Focus on school counselors, teachers, or public places',
                    'feels_overwhelming': 'Start with just one contact and one safe place'
                },
                'personalization_factors': ['risk_level', 'support_network', 'specific_threats']
            },
            
            'study_stress_management': {
                'name': 'Academic Stress Management System',
                'category': 'academic_support',
                'difficulty': 'medium',
                'time_required_minutes': 20,
                'evidence_strength': 'moderate',
                'effectiveness_rate': 0.73,
                'best_for': ['academic_pressure', 'overwhelm', 'procrastination'],
                'contraindications': ['severe_attention_disorders_without_treatment'],
                'check_in_days': 10,
                'variations': {
                    'time_management': 'Focus on scheduling and prioritization',
                    'study_techniques': 'Effective learning strategies',
                    'stress_reduction': 'Managing academic anxiety'
                },
                'instructions': {
                    'preparation': [
                        "Gather all your assignments, tests, and deadlines",
                        "Find a quiet space with minimal distractions",
                        "Have a calendar or planner available"
                    ],
                    'execution': [
                        "Write down everything you need to do, no matter how small",
                        "Break large assignments into smaller, specific tasks",
                        "Prioritize: urgent/important, important/not urgent, etc.",
                        "Schedule specific times for each task, including breaks",
                        "Use 25-minute focused study periods (Pomodoro technique)",
                        "Plan buffer time for unexpected delays"
                    ],
                    'completion': [
                        "Review your plan daily and adjust as needed",
                        "Celebrate completed tasks, even small ones",
                        "Be flexible - plans can change and that's okay"
                    ]
                },
                'troubleshooting': {
                    'too_overwhelming': 'Focus on just today\'s tasks first',
                    'perfectionism': 'Aim for good enough, not perfect',
                    'procrastination': 'Start with the easiest task to build momentum'
                },
                'personalization_factors': ['study_style', 'time_available', 'stress_level']
            },
            
            'social_confidence_building': {
                'name': 'Gradual Social Confidence Building',
                'category': 'social_skills',
                'difficulty': 'medium',
                'time_required_minutes': 30,
                'evidence_strength': 'moderate',
                'effectiveness_rate': 0.65,
                'best_for': ['social_anxiety', 'loneliness_isolation', 'low_self_esteem'],
                'contraindications': ['severe_social_phobia_without_treatment'],
                'check_in_days': 14,
                'variations': {
                    'online': 'Building connections through digital platforms',
                    'in_person': 'Face-to-face interaction practice',
                    'group_activities': 'Joining clubs or group activities'
                },
                'instructions': {
                    'preparation': [
                        "Start with very low-pressure social situations",
                        "Practice self-compassion - progress takes time",
                        "Identify one small social goal for the week"
                    ],
                    'execution': [
                        "Week 1: Make brief eye contact and smile at one person",
                        "Week 2: Give someone a genuine compliment",
                        "Week 3: Ask someone a simple question (time, directions)",
                        "Week 4: Have a brief conversation with someone new",
                        "Continue building gradually at your own pace"
                    ],
                    'completion': [
                        "Reflect on what went well, not what went wrong",
                        "Reward yourself for taking social risks",
                        "Plan your next small step forward"
                    ]
                },
                'troubleshooting': {
                    'too_scary': 'Take smaller steps - even thinking about it is progress',
                    'people_rejected_me': 'Their response is about them, not your worth',
                    'no_opportunities': 'Look for structured activities like clubs or classes'
                },
                'personalization_factors': ['anxiety_level', 'social_goals', 'available_contexts']
            }
        }
        
        # Enhanced counselor database with detailed matching capabilities
        self.counselor_database = [
            {
                'id': 'dr_sarah_smith',
                'name': 'Dr. Sarah Smith',
                'title': 'Clinical Psychologist',
                'specialties': ['anxiety', 'depression', 'academic_stress', 'young_adults', 'CBT'],
                'approaches': ['cognitive_behavioral', 'mindfulness', 'solution_focused'],
                'demographics': {'age_groups': ['teens', 'young_adults'], 'languages': ['english']},
                'availability': {
                    'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'saturday'],
                    'times': ['evening', 'saturday_morning'],
                    'wait_time_weeks': 2
                },
                'logistics': {
                    'insurance': ['most_major_plans', 'sliding_scale_available'],
                    'session_types': ['in_person', 'video', 'phone'],
                    'location': 'Downtown Medical Center'
                },
                'contact': {
                    'phone': '(555) 123-4567',
                    'website': 'dr-sarah-smith.com',
                    'booking_method': 'online_or_phone'
                },
                'match_strength': {
                    'anxiety_panic': 0.95,
                    'depression_symptoms': 0.90,
                    'academic_pressure': 0.85,
                    'social_anxiety': 0.80
                },
                'patient_reviews': {
                    'average_rating': 4.7,
                    'common_praise': ['very understanding', 'practical techniques', 'makes you feel heard']
                },
                'description': 'Dr. Smith specializes in helping teens and young adults develop practical coping skills for anxiety and depression using evidence-based techniques.'
            },
            {
                'id': 'dr_michael_jones',
                'name': 'Dr. Michael Jones',
                'title': 'Licensed Clinical Social Worker',
                'specialties': ['trauma', 'bullying', 'family_therapy', 'crisis_intervention', 'EMDR'],
                'approaches': ['trauma_informed', 'family_systems', 'strengths_based'],
                'demographics': {'age_groups': ['teens', 'adults', 'families'], 'languages': ['english', 'spanish']},
                'availability': {
                    'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
                    'times': ['afternoon', 'early_evening'],
                    'wait_time_weeks': 1
                },
                'logistics': {
                    'insurance': ['most_plans', 'medicaid', 'sliding_scale'],
                    'session_types': ['in_person', 'video'],
                    'location': 'Community Health Center'
                },
                'contact': {
                    'phone': '(555) 234-5678',
                    'website': 'communitycounseling.org/michael-jones',
                    'booking_method': 'phone_intake'
                },
                'match_strength': {
                    'bullying_harassment': 0.95,
                    'family_conflicts': 0.90,
                    'crisis_situation': 0.95,
                    'depression_symptoms': 0.75
                },
                'patient_reviews': {
                    'average_rating': 4.8,
                    'common_praise': ['creates safe space', 'experienced with trauma', 'family-oriented']
                },
                'description': 'Dr. Jones provides trauma-informed care with expertise in bullying, family conflicts, and crisis intervention.'
            },
            {
                'id': 'dr_maria_garcia',
                'name': 'Dr. Maria Garcia',
                'title': 'Licensed Marriage and Family Therapist',
                'specialties': ['social_anxiety', 'relationship_issues', 'self_esteem', 'identity_development'],
                'approaches': ['humanistic', 'acceptance_commitment', 'interpersonal'],
                'demographics': {'age_groups': ['teens', 'young_adults'], 'languages': ['english', 'spanish']},
                'availability': {
                    'days': ['monday', 'wednesday', 'friday', 'sunday'],
                    'times': ['morning', 'afternoon'],
                    'wait_time_weeks': 1
                },
                'logistics': {
                    'insurance': ['private_pay', 'some_insurance'],
                    'session_types': ['in_person', 'video'],
                    'location': 'Private Practice Downtown'
                },
                'contact': {
                    'phone': '(555) 345-6789',
                    'website': 'therapyspace.com/maria-garcia',
                    'booking_method': 'secure_portal'
                },
                'match_strength': {
                    'social_anxiety': 0.90,
                    'loneliness_isolation': 0.85,
                    'family_conflicts': 0.80,
                    'self_esteem': 0.90
                },
                'patient_reviews': {
                    'average_rating': 4.6,
                    'common_praise': ['warm and accepting', 'helps with self-acceptance', 'great for relationships']
                },
                'description': 'Dr. Garcia focuses on helping people build authentic relationships and develop strong self-identity through humanistic approaches.'
            }
        ]
        
        # Load existing data if available
        if save_path and os.path.exists(save_path):
            self.load_memory()

    def create_or_get_user(self, user_id: str) -> UserProfile:
        """Create new user profile or get existing one with thread safety"""
        
        with self._lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    first_interaction=datetime.now(),
                    last_interaction=datetime.now()
                )
                self.logger.info(f"Created new user profile: {user_id}")
            
            return self.user_profiles[user_id]

    def add_conversation_turn(self,
                            user_id: str,
                            user_message: str,
                            bot_response: str,
                            nlu_data: Dict[str, Any],
                            response_data: Dict[str, Any]) -> str:
        """Add new conversation turn and comprehensively update user profile"""
        
        with self._lock:
            # Create conversation turn
            turn_id = str(uuid.uuid4())
            turn = ConversationTurn(
                id=turn_id,
                user_message=user_message,
                bot_response=bot_response,
                timestamp=datetime.now(),
                intent=nlu_data['primary_intent'],
                confidence=nlu_data['confidence'],
                severity_score=nlu_data['severity_score'],
                conversation_stage=response_data.get('conversation_stage', 'unknown'),
                emotional_state=nlu_data.get('emotional_state', {}),
                method_suggested=response_data.get('method_suggested'),
                context_entities=nlu_data.get('context_entities', {})
            )
            
            # Add to memory stores
            self.conversation_history[user_id].append(turn)
            self.short_term_memory[user_id].append(turn)
            
            # Update user profile comprehensively
            profile = self.create_or_get_user(user_id)
            self._update_user_profile_from_turn(profile, turn, nlu_data, response_data)
            
            # Update method tracking if method was suggested
            if response_data.get('method_suggested'):
                self._track_method_suggestion(profile, response_data['method_suggested'], nlu_data)
            
            self.logger.info(f"Added conversation turn for {user_id}: {turn.intent} (confidence: {turn.confidence:.2f})")
            return turn_id

    def _update_user_profile_from_turn(self, profile: UserProfile, turn: ConversationTurn,
                                     nlu_data: Dict[str, Any], response_data: Dict[str, Any]):
        """Comprehensively update user profile from conversation turn"""
        
        # Basic updates
        profile.last_interaction = turn.timestamp
        profile.total_interactions += 1
        profile.current_stage = response_data.get('conversation_stage', profile.current_stage)
        
        # Update primary concerns
        intent = turn.intent
        if intent not in profile.primary_concerns and intent not in ['general_support', 'help_seeking']:
            profile.primary_concerns.append(intent)
            profile.primary_concerns = profile.primary_concerns[-5:]  # Keep top 5
        
        # Update severity tracking
        severity = turn.severity_score
        profile.severity_trend.append((turn.timestamp, severity))
        if len(profile.severity_trend) > 50:  # Keep last 50 interactions
            profile.severity_trend = profile.severity_trend[-50:]
        
        # Update concern-specific severity
        if intent in profile.concern_severity_history:
            profile.concern_severity_history[intent].append((turn.timestamp, severity))
        else:
            profile.concern_severity_history[intent] = [(turn.timestamp, severity)]
        
        # Update emotional patterns
        emotional_state = turn.emotional_state
        if emotional_state and 'primary_emotion' in emotional_state:
            emotion = emotional_state['primary_emotion']
            intensity = emotional_state.get('intensity', 0.0)
            
            if emotion in profile.emotional_patterns:
                profile.emotional_patterns[emotion].append((turn.timestamp, intensity))
            else:
                profile.emotional_patterns[emotion] = [(turn.timestamp, intensity)]
        
        # Update risk level
        profile.risk_level = self._assess_current_risk_level(profile, nlu_data)
        
        # Track crisis situations
        if intent == 'crisis_situation' or nlu_data.get('requires_immediate_help'):
            crisis_event = {
                'timestamp': turn.timestamp,
                'severity': severity,
                'message_preview': turn.user_message[:100],
                'intervention_provided': response_data.get('is_crisis_response', False),
                'risk_factors': nlu_data.get('risk_factors', [])
            }
            profile.crisis_history.append(crisis_event)
        
        # Update progress indicators
        self._update_progress_indicators(profile, turn, nlu_data)

    def _track_method_suggestion(self, profile: UserProfile, method_id: str, nlu_data: Dict[str, Any]):
        """Track method suggestion with detailed context"""
        
        method_data = self.method_library.get(method_id, {})
        
        if method_id in profile.methods_experienced:
            # Update existing method experience
            experience = profile.methods_experienced[method_id]
            experience.times_suggested += 1
        else:
            # Create new method experience
            experience = MethodExperience(
                method_id=method_id,
                method_name=method_data.get('name', method_id),
                first_suggested=datetime.now(),
                times_suggested=1
            )
            profile.methods_experienced[method_id] = experience
        
        # Add context when suggested
        context = {
            'severity_when_suggested': nlu_data.get('severity_score', 0.5),
            'intent_when_suggested': nlu_data.get('primary_intent'),
            'emotional_state': nlu_data.get('emotional_state', {}),
            'conversation_stage': nlu_data.get('conversation_stage', 'unknown')
        }
        experience.context_when_used.append(context)
        
        # Set follow-up timing
        check_in_days = method_data.get('check_in_days', 7)
        profile.current_method = method_id
        profile.method_check_due = datetime.now() + timedelta(days=check_in_days)

    def suggest_method(self, user_id: str, intent: str, severity: float, 
                      context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Intelligently suggest methods based on user profile and current situation"""
        
        with self._lock:
            profile = self.create_or_get_user(user_id)
            
            # Find methods suitable for this intent
            suitable_methods = []
            
            for method_id, method_data in self.method_library.items():
                # Check if method is appropriate for intent
                if intent in method_data.get('best_for', []):
                    # Check if user has had negative experiences with this method
                    if method_id in profile.methods_experienced:
                        experience = profile.methods_experienced[method_id]
                        if experience.average_effectiveness < 0.3:
                            continue  # Skip methods that haven't worked well
                    
                    suitable_methods.append((method_id, method_data))
            
            if not suitable_methods:
                self.logger.warning(f"No suitable methods found for {user_id} with intent {intent}")
                return None
            
            # Select best method based on multiple factors
            best_method = self._select_optimal_method(suitable_methods, profile, severity, context)
            method_id, method_data = best_method
            
            # Track the suggestion
            self._track_method_suggestion(profile, method_id, {'severity_score': severity, 'primary_intent': intent})
            
            # Personalize method instructions if possible
            personalized_data = self._personalize_method(method_data, profile, context)
            
            return {
                'method_id': method_id,
                'method_data': personalized_data,
                'check_in_date': profile.method_check_due,
                'personalization_applied': personalized_data != method_data,
                'selection_factors': self._get_selection_rationale(method_id, profile, severity)
            }

    def _select_optimal_method(self, suitable_methods: List[Tuple[str, Dict[str, Any]]], 
                             profile: UserProfile, severity: float, 
                             context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Select optimal method using multiple criteria"""
        
        method_scores = {}
        
        for method_id, method_data in suitable_methods:
            score = 0.0
            
            # Base effectiveness rate
            score += method_data.get('effectiveness_rate', 0.5) * 30
            
            # User's personal experience with this method
            if method_id in profile.methods_experienced:
                experience = profile.methods_experienced[method_id]
                score += experience.average_effectiveness * 40
                
                # Boost score if method was consistent
                score += experience.consistency_score * 10
                
                # Reduce score if user doesn't actually use it when suggested
                score *= experience.usage_frequency
            else:
                # New methods get a moderate boost for novelty
                score += 15
            
            # Difficulty appropriateness
            difficulty = method_data.get('difficulty', 'medium')
            if severity > 0.8 and difficulty == 'easy':
                score += 10  # High severity needs easy methods
            elif severity < 0.4 and difficulty == 'medium':
                score += 5   # Low severity can handle more complex methods
            
            # Time availability (from context)
            time_available = context.get('time_available', 'medium') if context else 'medium'
            method_time = method_data.get('time_required_minutes', 15)
            
            if time_available == 'low' and method_time <= 5:
                score += 8
            elif time_available == 'high' and method_time >= 15:
                score += 5
            
            # Evidence strength
            evidence = method_data.get('evidence_strength', 'moderate')
            evidence_bonus = {'high': 10, 'moderate': 5, 'low': 0}.get(evidence, 0)
            score += evidence_bonus
            
            # Avoid methods in user's avoided list
            if method_data.get('category') in profile.avoided_method_types:
                score *= 0.5
            
            # Boost methods in user's preferred list
            if method_data.get('category') in profile.preferred_method_types:
                score *= 1.3
            
            method_scores[method_id] = score
        
        # Select method with highest score
        best_method_id = max(method_scores, key=method_scores.get)
        best_method_data = next(data for mid, data in suitable_methods if mid == best_method_id)
        
        return best_method_id, best_method_data

    def _personalize_method(self, method_data: Dict[str, Any], profile: UserProfile, 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Personalize method instructions based on user profile"""
        
        personalized = method_data.copy()
        
        # Personalize based on user preferences
        comm_prefs = profile.communication_preferences
        
        if comm_prefs.get('instruction_style') == 'brief':
            # Shorten instructions
            if 'instructions' in personalized:
                for phase in personalized['instructions']:
                    if isinstance(personalized['instructions'][phase], list):
                        personalized['instructions'][phase] = personalized['instructions'][phase][:3]
        
        # Adapt based on previous experiences
        if method_data['name'] in profile.methods_experienced:
            experience = profile.methods_experienced[method_data['name']]
            
            # If user struggled with basic version, suggest variations
            if experience.average_effectiveness < 0.5 and 'variations' in personalized:
                # Suggest alternative variation
                variations = list(personalized['variations'].keys())
                if 'basic' in variations and len(variations) > 1:
                    alt_variation = variations[1] if variations[0] == 'basic' else variations[0]
                    personalized['suggested_variation'] = alt_variation
        
        # Add personalized encouragement
        if profile.total_interactions > 10:
            personalized['personal_note'] = "You've shown real commitment to working on your mental health. That persistence is going to serve you well with this technique."
        
        return personalized

    def _get_selection_rationale(self, method_id: str, profile: UserProfile, severity: float) -> Dict[str, str]:
        """Generate explanation for why this method was selected"""
        
        rationale = {}
        
        # Check user history
        if method_id in profile.methods_experienced:
            experience = profile.methods_experienced[method_id]
            if experience.average_effectiveness > 0.7:
                rationale['past_success'] = f"You've found this method helpful before (effectiveness: {experience.average_effectiveness:.1f})"
        else:
            rationale['novelty'] = "This is a new technique that might work well for your situation"
        
        # Severity appropriateness
        method_data = self.method_library.get(method_id, {})
        if severity > 0.7 and method_data.get('difficulty') == 'easy':
            rationale['severity_match'] = "Selected an easy-to-use technique since you're dealing with high distress"
        
        # Evidence basis
        evidence = method_data.get('evidence_strength', 'moderate')
        if evidence == 'high':
            rationale['evidence'] = "This technique has strong research support for effectiveness"
        
        return rationale

    def process_method_feedback(self, user_id: str, feedback_message: str, 
                              method_id: str = None) -> Dict[str, Any]:
        """Process comprehensive user feedback on suggested method"""
        
        with self._lock:
            profile = self.create_or_get_user(user_id)
            
            # Determine which method the feedback is about
            if not method_id:
                method_id = profile.current_method
            
            if not method_id:
                return {'processed': False, 'reason': 'No current method to evaluate'}
            
            if method_id not in profile.methods_experienced:
                return {'processed': False, 'reason': 'Method not found in user experience'}
            
            experience = profile.methods_experienced[method_id]
            
            # Analyze feedback with enhanced sentiment analysis
            feedback_analysis = self._analyze_feedback_sentiment(feedback_message)
            effectiveness_score = feedback_analysis['effectiveness_score']
            feedback_details = feedback_analysis['details']
            
            # Update method experience
            experience.times_used += 1
            experience.effectiveness_ratings.append(effectiveness_score)
            experience.last_used = datetime.now()
            
            # Record detailed feedback
            feedback_record = {
                'date': datetime.now(),
                'feedback_text': feedback_message,
                'effectiveness_score': effectiveness_score,
                'analysis': feedback_details,
                'context': self._get_current_context(profile)
            }
            experience.feedback_history.append(feedback_record)
            
            # Update user preferences based on feedback
            self._update_user_preferences_from_feedback(profile, method_id, feedback_analysis)
            
            # Clear current method
            profile.current_method = None
            profile.method_check_due = None
            
            # Generate next steps recommendation
            next_steps = self._generate_method_next_steps(experience, feedback_analysis, profile)
            
            # Update progress indicators
            self._update_progress_from_method_feedback(profile, method_id, effectiveness_score)
            
            return {
                'processed': True,
                'method_id': method_id,
                'effectiveness_score': effectiveness_score,
                'feedback_category': feedback_analysis['category'],
                'detailed_analysis': feedback_details,
                'next_steps': next_steps,
                'method_summary': {
                    'times_used': experience.times_used,
                    'average_effectiveness': experience.average_effectiveness,
                    'consistency': experience.consistency_score
                }
            }

    def _analyze_feedback_sentiment(self, feedback_message: str) -> Dict[str, Any]:
        """Enhanced feedback sentiment analysis"""
        
        message_lower = feedback_message.lower()
        
        # Define sentiment indicators with weights
        positive_indicators = {
            'helped': 0.8, 'better': 0.7, 'good': 0.6, 'worked': 0.8, 
            'useful': 0.7, 'effective': 0.9, 'easier': 0.6, 'calming': 0.8,
            'peaceful': 0.7, 'relieved': 0.8, 'improved': 0.9, 'amazing': 0.9,
            'fantastic': 0.9, 'love it': 0.9, 'really helped': 0.9
        }
        
        negative_indicators = {
            'didnt help': 0.2, 'not working': 0.2, 'worse': 0.1, 'harder': 0.3,
            'useless': 0.1, 'bad': 0.3, 'terrible': 0.1, 'hate it': 0.1,
            'pointless': 0.1, 'waste of time': 0.1, 'stupid': 0.2, 'annoying': 0.3
        }
        
        neutral_indicators = {
            'okay': 0.5, 'fine': 0.5, 'meh': 0.4, 'mixed': 0.5, 'sometimes': 0.5,
            'depends': 0.5, 'not sure': 0.5, 'maybe': 0.4
        }
        
        # Calculate weighted sentiment scores
        positive_score = 0.0
        negative_score = 0.0
        neutral_score = 0.0
        
        found_indicators = []
        
        for indicator, weight in positive_indicators.items():
            if indicator in message_lower:
                positive_score += weight
                found_indicators.append(f"positive: {indicator}")
        
        for indicator, weight in negative_indicators.items():
            if indicator in message_lower:
                negative_score += (1.0 - weight)  # Convert to negative scale
                found_indicators.append(f"negative: {indicator}")
        
        for indicator, weight in neutral_indicators.items():
            if indicator in message_lower:
                neutral_score += weight
                found_indicators.append(f"neutral: {indicator}")
        
        # Determine overall effectiveness score
        if positive_score > negative_score and positive_score > neutral_score:
            effectiveness_score = min(0.5 + positive_score * 0.3, 1.0)
            category = 'positive'
        elif negative_score > positive_score and negative_score > neutral_score:
            effectiveness_score = max(0.5 - negative_score * 0.3, 0.0)
            category = 'negative'
        elif neutral_score > 0:
            effectiveness_score = 0.5
            category = 'neutral'
        else:
            effectiveness_score = 0.5
            category = 'unclear'
        
        # Look for specific aspects mentioned
        aspects = {
            'difficulty': any(word in message_lower for word in ['hard', 'easy', 'difficult', 'simple']),
            'time': any(word in message_lower for word in ['time', 'long', 'short', 'quick', 'slow']),
            'effectiveness': any(word in message_lower for word in ['worked', 'helped', 'effective', 'useless']),
            'comfort': any(word in message_lower for word in ['comfortable', 'uncomfortable', 'weird', 'natural'])
        }
        
        return {
            'effectiveness_score': effectiveness_score,
            'category': category,
            'confidence': abs(effectiveness_score - 0.5) * 2,  # How confident we are in the assessment
            'details': {
                'indicators_found': found_indicators,
                'aspects_mentioned': aspects,
                'positive_strength': positive_score,
                'negative_strength': negative_score,
                'neutral_strength': neutral_score
            }
        }

    def _update_user_preferences_from_feedback(self, profile: UserProfile, method_id: str, 
                                             feedback_analysis: Dict[str, Any]):
        """Update user preferences based on method feedback"""
        
        method_data = self.method_library.get(method_id, {})
        category = method_data.get('category', 'unknown')
        difficulty = method_data.get('difficulty', 'medium')
        
        effectiveness = feedback_analysis['effectiveness_score']
        
        # Update category preferences
        if effectiveness > 0.7:
            if category not in profile.preferred_method_types:
                profile.preferred_method_types.append(category)
        elif effectiveness < 0.3:
            if category not in profile.avoided_method_types:
                profile.avoided_method_types.append(category)
        
        # Update difficulty preferences
        aspects = feedback_analysis['details']['aspects_mentioned']
        if aspects.get('difficulty'):
            if effectiveness > 0.6:
                profile.communication_preferences['preferred_difficulty'] = difficulty
            elif effectiveness < 0.4 and difficulty == 'hard':
                profile.communication_preferences['preferred_difficulty'] = 'easy'
        
        # Update time preferences
        if aspects.get('time'):
            time_required = method_data.get('time_required_minutes', 15)
            if effectiveness > 0.6:
                profile.communication_preferences['preferred_time_commitment'] = time_required

    def _generate_method_next_steps(self, experience: MethodExperience, 
                                  feedback_analysis: Dict[str, Any], 
                                  profile: UserProfile) -> Dict[str, str]:
        """Generate intelligent next steps based on method feedback"""
        
        effectiveness = feedback_analysis['effectiveness_score']
        category = feedback_analysis['category']
        
        next_steps = {}
        
        if category == 'positive':
            if effectiveness > 0.8:
                next_steps['recommendation'] = "This method is working really well for you! Let's continue using it and maybe explore related techniques."
                next_steps['action'] = "continue_and_expand"
            else:
                next_steps['recommendation'] = "This method seems to be helping. Keep practicing it to strengthen the benefits."
                next_steps['action'] = "continue_practice"
        
        elif category == 'negative':
            if effectiveness < 0.2:
                next_steps['recommendation'] = "This method doesn't seem to be a good fit for you right now. Let's try a completely different approach."
                next_steps['action'] = "try_different_category"
            else:
                next_steps['recommendation'] = "This method had mixed results. We could try a variation or explore why it wasn't more helpful."
                next_steps['action'] = "try_variation_or_explore"
        
        else:  # neutral or unclear
            next_steps['recommendation'] = "It sounds like you had a mixed experience. Can you tell me more about what aspects worked and what didn't?"
            next_steps['action'] = "gather_more_information"
        
        # Add specific suggestions based on method library
        method_data = self.method_library.get(experience.method_id, {})
        if 'troubleshooting' in method_data:
            next_steps['troubleshooting_available'] = "I have some troubleshooting tips that might help if you want to try this method again."
        
        if 'variations' in method_data and len(method_data['variations']) > 1:
            next_steps['variations_available'] = "There are different ways to do this technique that might work better for you."
        
        return next_steps

    def should_follow_up_on_method(self, user_id: str) -> Dict[str, Any]:
        """Check if we should follow up on a previously suggested method"""
        
        with self._lock:
            profile = self.user_profiles.get(user_id)
            if not profile or not profile.method_check_due:
                return {'should_follow_up': False}
            
            if datetime.now() >= profile.method_check_due:
                method_data = self.method_library.get(profile.current_method, {})
                
                return {
                    'should_follow_up': True,
                    'method_id': profile.current_method,
                    'method_name': method_data.get('name', profile.current_method),
                    'days_since_suggestion': (datetime.now() - profile.method_check_due).days + method_data.get('check_in_days', 7),
                    'suggested_check_in': f"How did the {method_data.get('name', 'technique')} work for you?"
                }
            
            return {'should_follow_up': False}

    def get_counselor_recommendation(self, user_id: str, urgency: str = 'standard') -> Dict[str, Any]:
        """Get intelligent counselor recommendation based on comprehensive user profile analysis"""
        
        with self._lock:
            profile = self.create_or_get_user(user_id)
            
            # Analyze user needs for counselor matching
            user_analysis = self._analyze_user_for_counselor_matching(profile)
            
            # Find best matching counselors
            counselor_matches = []
            
            for counselor in self.counselor_database:
                match_score = self._calculate_counselor_match_score(counselor, user_analysis)
                counselor_matches.append((counselor, match_score))
            
            # Sort by match score
            counselor_matches.sort(key=lambda x: x[1], reverse=True)
            best_match, best_score = counselor_matches[0]
            
            # Generate comprehensive referral information
            referral_info = self._generate_comprehensive_referral(best_match, user_analysis, best_score, profile)
            
            return {
                'counselor': best_match,
                'match_score': best_score,
                'referral_info': referral_info,
                'alternative_options': [match[0] for match in counselor_matches[1:3]],  # Top 2 alternatives
                'referral_urgency': self._assess_referral_urgency(profile),
                'preparation_suggestions': self._generate_therapy_preparation_suggestions(profile)
            }

    def _analyze_user_for_counselor_matching(self, profile: UserProfile) -> Dict[str, Any]:
        """Comprehensive analysis of user needs for counselor matching"""
        
        analysis = {
            'primary_concerns': profile.primary_concerns,
            'severity_level': self._calculate_average_recent_severity(profile),
            'crisis_history': len(profile.crisis_history),
            'method_response_patterns': self._analyze_method_response_patterns(profile),
            'communication_style': profile.communication_preferences,
            'urgency_level': self._assess_referral_urgency(profile),
            'support_needs': self._identify_support_needs(profile)
        }
        
        return analysis

    def _calculate_counselor_match_score(self, counselor: Dict[str, Any], 
                                       user_analysis: Dict[str, Any]) -> float:
        """Calculate how well a counselor matches user needs"""
        
        score = 0.0
        
        # Specialty matching (40% of score)
        specialty_score = 0.0
        for concern in user_analysis['primary_concerns']:
            if concern in counselor.get('match_strength', {}):
                specialty_score += counselor['match_strength'][concern]
        
        if user_analysis['primary_concerns']:
            specialty_score /= len(user_analysis['primary_concerns'])
        
        score += specialty_score * 0.4
        
        # Approach matching (20% of score)
        approach_score = 0.0
        user_method_patterns = user_analysis.get('method_response_patterns', {})
        
        counselor_approaches = counselor.get('approaches', [])
        if 'structured_methods_preferred' in user_method_patterns and user_method_patterns['structured_methods_preferred']:
            if 'cognitive_behavioral' in counselor_approaches:
                approach_score += 0.8
        
        if 'relationship_focused_needs' in user_analysis.get('support_needs', []):
            if any(approach in counselor_approaches for approach in ['humanistic', 'interpersonal']):
                approach_score += 0.7
        
        score += approach_score * 0.2
        
        # Availability matching (15% of score)
        availability_score = 0.6  # Base score
        wait_time = counselor.get('availability', {}).get('wait_time_weeks', 4)
        
        if user_analysis['urgency_level'] == 'high' and wait_time <= 1:
            availability_score = 1.0
        elif user_analysis['urgency_level'] == 'medium' and wait_time <= 2:
            availability_score = 0.8
        elif wait_time > 4:
            availability_score = 0.4
        
        score += availability_score * 0.15
        
        # Crisis experience (15% of score)
        crisis_score = 0.6  # Base score
        if user_analysis['crisis_history'] > 0:
            if 'crisis_intervention' in counselor.get('specialties', []):
                crisis_score = 1.0
            elif 'trauma' in counselor.get('specialties', []):
                crisis_score = 0.8
        
        score += crisis_score * 0.15
        
        # Patient reviews (10% of score)
        review_score = counselor.get('patient_reviews', {}).get('average_rating', 4.0) / 5.0
        score += review_score * 0.1
        
        return min(score, 1.0)

    def _generate_comprehensive_referral(self, counselor: Dict[str, Any], 
                                       user_analysis: Dict[str, Any], 
                                       match_score: float, profile: 'UserProfile') -> Dict[str, Any]:
        """Generate comprehensive referral information"""
        
        referral_info = {
            'why_recommended': self._generate_recommendation_rationale(counselor, user_analysis, match_score),
            'what_to_expect': self._generate_therapy_expectations(counselor, user_analysis),
            'how_to_prepare': self._generate_therapy_preparation_suggestions(profile),
            'logistics': self._extract_logistics_info(counselor),
            'next_steps': self._generate_referral_next_steps(counselor)
        }
        
        return referral_info

    def _generate_recommendation_rationale(self, counselor: Dict[str, Any], 
                                         user_analysis: Dict[str, Any], 
                                         match_score: float) -> str:
        """Generate clear rationale for counselor recommendation"""
        
        rationale_parts = []
        
        # Specialty match
        matched_specialties = []
        for concern in user_analysis['primary_concerns']:
            if concern in counselor.get('specialties', []):
                matched_specialties.append(concern.replace('_', ' '))
        
        if matched_specialties:
            rationale_parts.append(f"Dr. {counselor['name'].split()[-1]} specializes in {', '.join(matched_specialties)}")
        
        # Approach match
        approaches = counselor.get('approaches', [])
        if 'cognitive_behavioral' in approaches:
            rationale_parts.append("Uses evidence-based cognitive-behavioral techniques")
        if 'trauma_informed' in approaches:
            rationale_parts.append("Experienced in trauma-informed care")
        
        # Reviews and ratings
        reviews = counselor.get('patient_reviews', {})
        if reviews.get('average_rating', 0) >= 4.5:
            common_praise = reviews.get('common_praise', [])
            if common_praise:
                rationale_parts.append(f"Patients often mention: {', '.join(common_praise[:2])}")
        
        return ". ".join(rationale_parts) + "."

    def update_counselor_referral_status(self, user_id: str, status: str, details: Dict[str, Any] = None):
        """Update user's counselor referral status with detailed tracking"""
        
        with self._lock:
            profile = self.create_or_get_user(user_id)
            
            old_status = profile.counselor_referral_status
            profile.counselor_referral_status = status
            
            # Track referral progression
            referral_update = {
                'timestamp': datetime.now(),
                'from_status': old_status,
                'to_status': status,
                'details': details or {}
            }
            
            profile.referral_history.append(referral_update)
            
            # Update referral date if this is the first suggestion
            if status == 'suggested' and not profile.referral_date:
                profile.referral_date = datetime.now()
            
            self.logger.info(f"Updated counselor referral status for {user_id}: {old_status} -> {status}")

    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user summary for analytics and insights"""
        
        with self._lock:
            profile = self.user_profiles.get(user_id)
            if not profile:
                return {'user_id': user_id, 'exists': False}
            
            # Calculate comprehensive progress metrics
            progress_metrics = profile.calculate_overall_progress()
            
            # Get conversation statistics
            conversation_stats = self._calculate_conversation_statistics(user_id)
            
            # Get method effectiveness summary
            method_summary = self._generate_method_effectiveness_summary(profile)
            
            # Get risk assessment
            risk_assessment = self._generate_comprehensive_risk_assessment(profile)
            
            return {
                'user_id': user_id,
                'exists': True,
                'profile_summary': {
                    'first_interaction': profile.first_interaction,
                    'last_interaction': profile.last_interaction,
                    'total_interactions': profile.total_interactions,
                    'current_stage': profile.current_stage,
                    'primary_concerns': profile.primary_concerns,
                    'risk_level': profile.risk_level
                },
                'progress_metrics': progress_metrics,
                'conversation_stats': conversation_stats,
                'method_effectiveness': method_summary,
                'risk_assessment': risk_assessment,
                'professional_help': {
                    'referral_status': profile.counselor_referral_status,
                    'referral_date': profile.referral_date,
                    'referral_history': profile.referral_history[-3:]  # Last 3 updates
                },
                'personalization': {
                    'communication_preferences': profile.communication_preferences,
                    'preferred_methods': profile.preferred_method_types,
                    'avoided_methods': profile.avoided_method_types
                }
            }

    def _calculate_conversation_statistics(self, user_id: str) -> Dict[str, Any]:
        """Calculate detailed conversation statistics"""
        
        turns = self.conversation_history.get(user_id, [])
        
        if not turns:
            return {}
        
        # Intent distribution
        intent_counts = {}
        severity_scores = []
        confidence_scores = []
        
        for turn in turns:
            intent_counts[turn.intent] = intent_counts.get(turn.intent, 0) + 1
            severity_scores.append(turn.severity_score)
            confidence_scores.append(turn.confidence)
        
        # Calculate trends
        recent_turns = turns[-10:] if len(turns) >= 10 else turns
        recent_severity = [turn.severity_score for turn in recent_turns]
        
        return {
            'total_turns': len(turns),
            'intent_distribution': intent_counts,
            'average_severity': np.mean(severity_scores) if severity_scores else 0.0,
            'recent_average_severity': np.mean(recent_severity) if recent_severity else 0.0,
            'severity_trend': 'improving' if len(recent_severity) > 1 and recent_severity[-1] < recent_severity[0] else 'stable',
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'conversation_span_days': (turns[-1].timestamp - turns[0].timestamp).days if len(turns) > 1 else 0
        }

    def _generate_method_effectiveness_summary(self, profile: UserProfile) -> Dict[str, Any]:
        """Generate comprehensive method effectiveness summary"""
        
        if not profile.methods_experienced:
            return {'total_methods': 0, 'overall_success_rate': 0.0}
        
        method_summaries = {}
        total_effectiveness = 0.0
        
        for method_id, experience in profile.methods_experienced.items():
            method_summaries[method_id] = {
                'times_suggested': experience.times_suggested,
                'times_used': experience.times_used,
                'usage_rate': experience.usage_frequency,
                'average_effectiveness': experience.average_effectiveness,
                'consistency': experience.consistency_score,
                'last_used': experience.last_used
            }
            total_effectiveness += experience.average_effectiveness
        
        return {
            'total_methods': len(profile.methods_experienced),
            'overall_success_rate': total_effectiveness / len(profile.methods_experienced),
            'method_details': method_summaries,
            'most_effective': profile.get_most_effective_methods(3),
            'preferred_categories': profile.preferred_method_types,
            'avoided_categories': profile.avoided_method_types
        }

    def _generate_comprehensive_risk_assessment(self, profile: UserProfile) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        
        risk_factors = []
        protective_factors = []
        
        # Analyze severity trends
        if len(profile.severity_trend) > 5:
            recent_trend = [score for _, score in profile.severity_trend[-5:]]
            if np.mean(recent_trend) > 0.7:
                risk_factors.append('high_recent_severity')
            
            if len(recent_trend) > 1 and recent_trend[-1] > recent_trend[0]:
                risk_factors.append('increasing_severity')
        
        # Crisis history
        if len(profile.crisis_history) > 0:
            risk_factors.append('crisis_history')
            
            recent_crisis = any(
                (datetime.now() - event['timestamp']).days < 30 
                for event in profile.crisis_history
            )
            if recent_crisis:
                risk_factors.append('recent_crisis')
        
        # Method effectiveness
        if profile.calculate_method_success_rate() > 0.6:
            protective_factors.append('responsive_to_interventions')
        elif profile.calculate_method_success_rate() < 0.3:
            risk_factors.append('poor_method_response')
        
        # Engagement level
        if profile.total_interactions > 10:
            protective_factors.append('high_engagement')
        
        # Professional help
        if profile.counselor_referral_status in ['attending', 'booked']:
            protective_factors.append('professional_support')
        elif profile.counselor_referral_status == 'suggested' and (datetime.now() - profile.referral_date).days > 30:
            risk_factors.append('declined_professional_help')
        
        return {
            'current_risk_level': profile.risk_level,
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'risk_score': len(risk_factors) / max(len(risk_factors) + len(protective_factors), 1),
            'recommendations': self._generate_risk_recommendations(risk_factors, protective_factors)
        }

    def _generate_risk_recommendations(self, risk_factors: List[str], protective_factors: List[str]) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        if 'crisis_history' in risk_factors:
            recommendations.append('Maintain regular check-ins and crisis planning')
        
        if 'poor_method_response' in risk_factors:
            recommendations.append('Consider professional referral for specialized interventions')
        
        if 'increasing_severity' in risk_factors:
            recommendations.append('Increase frequency of support and monitoring')
        
        if 'high_engagement' in protective_factors:
            recommendations.append('Continue building on user engagement and motivation')
        
        if not protective_factors:
            recommendations.append('Focus on building therapeutic relationship and engagement')
        
        return recommendations

    def _assess_current_risk_level(self, profile: UserProfile, nlu_data: Dict[str, Any]) -> str:
        """Assess current risk level based on multiple factors"""
        
        # Check for immediate crisis indicators
        if (nlu_data.get('primary_intent') == 'crisis_situation' or
            nlu_data.get('requires_immediate_help') or
            nlu_data.get('severity_score', 0) > 0.9):
            return 'crisis'
        
        # Check recent crisis history
        recent_crisis = any(
            (datetime.now() - event['timestamp']).days < 7
            for event in profile.crisis_history
        )
        if recent_crisis:
            return 'high'
        
        # Check severity trends
        if len(profile.severity_trend) >= 3:
            recent_scores = [score for _, score in profile.severity_trend[-3:]]
            avg_recent = np.mean(recent_scores)
            
            if avg_recent > 0.7:
                return 'high'
            elif avg_recent > 0.5:
                return 'medium'
        
        return 'low'

    def _update_progress_indicators(self, profile: UserProfile, turn: ConversationTurn, nlu_data: Dict[str, Any]):
        """Update progress indicators based on conversation turn"""
        
        # Look for improvement indicators
        improvement_keywords = ['better', 'improved', 'helping', 'working', 'easier', 'calmer']
        if any(keyword in turn.user_message.lower() for keyword in improvement_keywords):
            indicator = {
                'timestamp': turn.timestamp,
                'type': 'subjective_improvement',
                'description': 'User reported feeling better',
                'severity_at_time': turn.severity_score
            }
            profile.improvement_indicators.append(indicator)
        
        # Look for regression indicators
        regression_keywords = ['worse', 'harder', 'overwhelming', 'cant cope', 'giving up']
        if any(keyword in turn.user_message.lower() for keyword in regression_keywords):
            indicator = {
                'timestamp': turn.timestamp,
                'type': 'subjective_decline',
                'description': 'User reported increased difficulty',
                'severity_at_time': turn.severity_score
            }
            profile.regression_indicators.append(indicator)

    def _update_progress_from_method_feedback(self, profile: UserProfile, method_id: str, effectiveness_score: float):
        """Update progress indicators based on method feedback"""
        
        if effectiveness_score > 0.7:
            indicator = {
                'timestamp': datetime.now(),
                'type': 'method_success',
                'description': f'Method {method_id} was effective',
                'method_id': method_id,
                'effectiveness_score': effectiveness_score
            }
            profile.improvement_indicators.append(indicator)
        elif effectiveness_score < 0.3:
            indicator = {
                'timestamp': datetime.now(),
                'type': 'method_failure',
                'description': f'Method {method_id} was not effective',
                'method_id': method_id,
                'effectiveness_score': effectiveness_score
            }
            profile.regression_indicators.append(indicator)

    def _calculate_average_recent_severity(self, profile: UserProfile) -> float:
        """Calculate average severity over recent interactions"""
        
        if len(profile.severity_trend) < 3:
            return 0.5  # Default moderate severity
        
        recent_scores = [score for _, score in profile.severity_trend[-5:]]
        return np.mean(recent_scores)

    def _analyze_method_response_patterns(self, profile: UserProfile) -> Dict[str, Any]:
        """Analyze patterns in how user responds to different methods"""
        
        patterns = {}
        
        if not profile.methods_experienced:
            return patterns
        
        # Analyze by method category
        category_effectiveness = {}
        for experience in profile.methods_experienced.values():
            method_data = self.method_library.get(experience.method_id, {})
            category = method_data.get('category', 'unknown')
            
            if category in category_effectiveness:
                category_effectiveness[category].append(experience.average_effectiveness)
            else:
                category_effectiveness[category] = [experience.average_effectiveness]
        
        # Find preferences
        best_category = None
        best_avg = 0.0
        
        for category, scores in category_effectiveness.items():
            avg_score = np.mean(scores)
            if avg_score > best_avg:
                best_avg = avg_score
                best_category = category
        
        patterns['best_responding_category'] = best_category
        patterns['category_effectiveness'] = {
            cat: np.mean(scores) for cat, scores in category_effectiveness.items()
        }
        
        # Analyze difficulty preferences
        difficulty_scores = {}
        for experience in profile.methods_experienced.values():
            method_data = self.method_library.get(experience.method_id, {})
            difficulty = method_data.get('difficulty', 'medium')
            
            if difficulty in difficulty_scores:
                difficulty_scores[difficulty].append(experience.average_effectiveness)
            else:
                difficulty_scores[difficulty] = [experience.average_effectiveness]
        
        patterns['difficulty_preferences'] = {
            diff: np.mean(scores) for diff, scores in difficulty_scores.items()
        }
        
        # Check for structured vs flexible preference
        structured_effectiveness = []
        flexible_effectiveness = []
        
        for experience in profile.methods_experienced.values():
            method_data = self.method_library.get(experience.method_id, {})
            if method_data.get('category') in ['anxiety_management', 'academic_support']:
                structured_effectiveness.append(experience.average_effectiveness)
            else:
                flexible_effectiveness.append(experience.average_effectiveness)
        
        if structured_effectiveness and flexible_effectiveness:
            patterns['structured_methods_preferred'] = np.mean(structured_effectiveness) > np.mean(flexible_effectiveness)
        
        return patterns

    def _identify_support_needs(self, profile: UserProfile) -> List[str]:
        """Identify specific support needs based on user profile"""
        
        needs = []
        
        # Based on primary concerns
        for concern in profile.primary_concerns:
            if concern in ['loneliness_isolation', 'social_anxiety']:
                needs.append('relationship_focused_needs')
            elif concern in ['crisis_situation', 'self_harm']:
                needs.append('crisis_intervention_needs')
            elif concern in ['family_conflicts']:
                needs.append('family_therapy_needs')
            elif concern in ['academic_pressure']:
                needs.append('specialized_academic_support')
        
        # Based on method response patterns
        method_patterns = self._analyze_method_response_patterns(profile)
        if method_patterns.get('structured_methods_preferred'):
            needs.append('structured_intervention_needs')
        
        # Based on engagement level
        if profile.total_interactions > 15:
            needs.append('long_term_therapeutic_relationship')
        
        # Based on crisis history
        if len(profile.crisis_history) > 1:
            needs.append('crisis_management_expertise')
        
        return list(set(needs))  # Remove duplicates

    def _assess_referral_urgency(self, profile: UserProfile) -> str:
        """Assess urgency level for professional referral"""
        
        # High urgency indicators
        if (profile.risk_level == 'crisis' or
            len(profile.crisis_history) > 2 or
            any((datetime.now() - event['timestamp']).days < 7 for event in profile.crisis_history)):
            return 'high'
        
        # Medium urgency indicators
        if (profile.risk_level == 'high' or
            self._calculate_average_recent_severity(profile) > 0.7 or
            profile.calculate_method_success_rate() < 0.3):
            return 'medium'
        
        return 'low'

    def _generate_therapy_preparation_suggestions(self, profile: UserProfile) -> List[str]:
        """Generate suggestions to help user prepare for therapy"""
        
        suggestions = [
            "Think about what you most want to work on in therapy",
            "Consider what times and days work best for your schedule",
            "Prepare a brief summary of your main concerns",
            "Think about what has and hasn't been helpful so far"
        ]
        
        # Customize based on user profile
        if profile.primary_concerns:
            concern_list = ', '.join(profile.primary_concerns[:3])
            suggestions.append(f"Be ready to discuss your experiences with {concern_list}")
        
        if profile.methods_experienced:
            suggestions.append("Share information about techniques you've tried and how they worked")
        
        if len(profile.crisis_history) > 0:
            suggestions.append("Be honest about any crisis situations you've experienced")
        
        return suggestions

    def _get_current_context(self, profile: UserProfile) -> Dict[str, Any]:
        """Get current user context for feedback recording"""
        
        return {
            'current_severity': profile.severity_trend[-1][1] if profile.severity_trend else 0.5,
            'current_stage': profile.current_stage,
            'recent_concerns': profile.primary_concerns,
            'total_interactions': profile.total_interactions,
            'risk_level': profile.risk_level
        }

    def _extract_logistics_info(self, counselor: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format logistics information for counselor"""
        
        logistics = counselor.get('logistics', {})
        availability = counselor.get('availability', {})
        contact = counselor.get('contact', {})
        
        return {
            'wait_time': f"{availability.get('wait_time_weeks', 'Unknown')} weeks",
            'session_types': ', '.join(logistics.get('session_types', ['in_person'])),
            'insurance': ', '.join(logistics.get('insurance', ['Contact for details'])),
            'location': logistics.get('location', 'Contact for details'),
            'phone': contact.get('phone', 'Contact information not available'),
            'website': contact.get('website', 'Not available'),
            'booking_method': contact.get('booking_method', 'Call for appointment')
        }

    def _generate_therapy_expectations(self, counselor: Dict[str, Any], user_analysis: Dict[str, Any]) -> str:
        """Generate what user can expect from therapy with this counselor"""
        
        expectations = []
        
        approaches = counselor.get('approaches', [])
        if 'cognitive_behavioral' in approaches:
            expectations.append("learn practical coping strategies and thought management techniques")
        if 'trauma_informed' in approaches:
            expectations.append("work through difficult experiences in a safe, understanding environment")
        if 'family_systems' in approaches:
            expectations.append("explore family dynamics and improve communication")
        
        if not expectations:
            expectations.append("develop personalized strategies for managing your mental health")
        
        return f"In therapy with {counselor['name']}, you can expect to {', and '.join(expectations)}."

    def _generate_referral_next_steps(self, counselor: Dict[str, Any]) -> List[str]:
        """Generate specific next steps for referral"""
        
        contact = counselor.get('contact', {})
        booking_method = contact.get('booking_method', 'phone')
        
        steps = []
        
        if booking_method == 'online_or_phone':
            steps.append(f"Visit {contact.get('website', 'their website')} or call {contact.get('phone', 'their office')}")
        elif booking_method == 'phone_intake':
            steps.append(f"Call {contact.get('phone', 'their office')} to schedule an intake appointment")
        elif booking_method == 'secure_portal':
            steps.append(f"Use the secure booking portal at {contact.get('website', 'their website')}")
        else:
            steps.append("Contact their office to schedule an initial appointment")
        
        steps.extend([
            "Prepare information about your insurance coverage",
            "Think about your availability for appointments",
            "Prepare to discuss your main concerns briefly"
        ])
        
        return steps

    def save_memory(self, filepath: str = None):
        """Save comprehensive memory data to file with error handling"""
        
        save_path = filepath or self.save_path
        if not save_path:
            self.logger.warning("No save path specified for memory")
            return
        
        try:
            with self._lock:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Prepare data for saving with proper serialization
                memory_data = {
                    'user_profiles': {},
                    'conversation_history': {},
                    'method_library': self.method_library,
                    'counselor_database': self.counselor_database,
                    'save_timestamp': datetime.now().isoformat(),
                    'version': '2.0'
                }
                
                # Serialize user profiles
                for uid, profile in self.user_profiles.items():
                    profile_dict = asdict(profile)
                    # Convert datetime objects to ISO strings
                    profile_dict = self._serialize_datetime_fields(profile_dict)
                    memory_data['user_profiles'][uid] = profile_dict
                
                # Serialize conversation history
                for uid, turns in self.conversation_history.items():
                    memory_data['conversation_history'][uid] = []
                    for turn in turns:
                        turn_dict = asdict(turn)
                        turn_dict = self._serialize_datetime_fields(turn_dict)
                        memory_data['conversation_history'][uid].append(turn_dict)
                
                # Save to file
                with open(save_path, 'wb') as f:
                    pickle.dump(memory_data, f)
                
                self.logger.info(f"✅ Memory saved successfully to {save_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save memory: {e}")

    def _serialize_datetime_fields(self, obj):
        """Recursively serialize datetime fields to ISO strings"""
        
        if isinstance(obj, dict):
            return {k: self._serialize_datetime_fields(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime_fields(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def load_memory(self, filepath: str = None):
        """Load comprehensive memory data from file with error handling"""
        
        load_path = filepath or self.save_path
        if not load_path or not os.path.exists(load_path):
            self.logger.warning(f"Memory file not found: {load_path}")
            return
        
        try:
            with self._lock:
                with open(load_path, 'rb') as f:
                    memory_data = pickle.load(f)
                
                # Load user profiles
                self.user_profiles = {}
                for uid, profile_data in memory_data.get('user_profiles', {}).items():
                    # Convert ISO strings back to datetime objects
                    profile_data = self._deserialize_datetime_fields(profile_data)
                    # --- NEW: Reconstruct MethodExperience objects ---
                    if 'methods_experienced' in profile_data:
                        reconstructed_methods = {}
                        for method_id, exp_data in profile_data['methods_experienced'].items():
            # Make sure datetime fields inside the experience are also converted
                           exp_data = self._deserialize_datetime_fields(exp_data)
                           reconstructed_methods[method_id] = MethodExperience(**exp_data)
                        profile_data['methods_experienced'] = reconstructed_methods
                    
                    # Handle dataclass field defaults
                    profile = UserProfile(**profile_data)
                    self.user_profiles[uid] = profile
                
                # Load conversation history
                self.conversation_history = defaultdict(list)
                for uid, turns_data in memory_data.get('conversation_history', {}).items():
                    turns = []
                    for turn_data in turns_data:
                        turn_data = self._deserialize_datetime_fields(turn_data)
                        turn = ConversationTurn(**turn_data)
                        turns.append(turn)
                    self.conversation_history[uid] = turns
                
                # Update libraries if available
                if 'method_library' in memory_data:
                    self.method_library.update(memory_data['method_library'])
                
                if 'counselor_database' in memory_data:
                    self.counselor_database = memory_data['counselor_database']
                
                self.logger.info(f"✅ Memory loaded successfully from {load_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load memory: {e}")

    def _deserialize_datetime_fields(self, obj):
        """Recursively deserialize datetime fields from ISO strings"""
        
        if isinstance(obj, dict):
            return {k: self._deserialize_datetime_fields(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_datetime_fields(item) for item in obj]
        elif isinstance(obj, str) and self._is_iso_datetime(obj):
            try:
                return datetime.fromisoformat(obj)
            except ValueError:
                return obj
        else:
            return obj

    def _is_iso_datetime(self, s: str) -> bool:
        """Check if string looks like ISO datetime format"""
        
        if len(s) < 19:  # Minimum length for ISO datetime
            return False
        
        # Simple heuristic: check if it contains date-like patterns
        return ('T' in s or ' ' in s) and '-' in s and ':' in s

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old conversation data while preserving important user profile information"""
        
        with self._lock:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cleaned_users = 0
            
            for user_id in list(self.conversation_history.keys()):
                # Remove old conversation turns
                original_count = len(self.conversation_history[user_id])
                self.conversation_history[user_id] = [
                    turn for turn in self.conversation_history[user_id]
                    if turn.timestamp >= cutoff_date
                ]
                
                # If no recent conversations, mark user as inactive but keep profile
                if not self.conversation_history[user_id]:
                    del self.conversation_history[user_id]
                    
                    # Keep user profile but mark as inactive
                    if user_id in self.user_profiles:
                        profile = self.user_profiles[user_id]
                        # Clean old data from profile but keep essential info
                        profile.severity_trend = [
                            (timestamp, score) for timestamp, score in profile.severity_trend
                            if timestamp >= cutoff_date
                        ]
                        
                        # Keep only recent method experiences
                        for method_id in list(profile.methods_experienced.keys()):
                            experience = profile.methods_experienced[method_id]
                            if experience.last_used and experience.last_used < cutoff_date:
                                # Keep the experience but clean old feedback
                                experience.feedback_history = [
                                    feedback for feedback in experience.feedback_history
                                    if feedback['date'] >= cutoff_date
                                ]
                    
                    cleaned_users += 1
            
            self.logger.info(f"🧹 Cleaned up data for {cleaned_users} inactive users (keeping {days_to_keep} days)")

    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        
        with self._lock:
            total_users = len(self.user_profiles)
            active_users = len(self.conversation_history)
            
            # Calculate engagement metrics
            total_interactions = sum(profile.total_interactions for profile in self.user_profiles.values())
            avg_interactions = total_interactions / max(total_users, 1)
            
            # Method effectiveness analytics
            all_method_experiences = []
            for profile in self.user_profiles.values():
                all_method_experiences.extend(profile.methods_experienced.values())
            
            if all_method_experiences:
                overall_method_success = np.mean([exp.average_effectiveness for exp in all_method_experiences])
                method_usage_rate = np.mean([exp.usage_frequency for exp in all_method_experiences])
            else:
                overall_method_success = 0.0
                method_usage_rate = 0.0
            
            # Risk level distribution
            risk_distribution = {}
            for profile in self.user_profiles.values():
                risk_level = profile.risk_level
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            
            # Referral analytics
            referral_stats = {}
            for profile in self.user_profiles.values():
                status = profile.counselor_referral_status
                referral_stats[status] = referral_stats.get(status, 0) + 1
            
            # Progress analytics
            improving_users = 0
            for profile in self.user_profiles.values():
                progress = profile.calculate_overall_progress()
                if progress['improvement'] > 0.1:
                    improving_users += 1
            
            return {
                'user_metrics': {
                    'total_users': total_users,
                    'active_users': active_users,
                    'total_interactions': total_interactions,
                    'average_interactions_per_user': avg_interactions,
                    'users_showing_improvement': improving_users,
                    'improvement_rate': improving_users / max(total_users, 1)
                },
                'method_effectiveness': {
                    'overall_success_rate': overall_method_success,
                    'usage_rate': method_usage_rate,
                    'total_method_experiences': len(all_method_experiences),
                    'methods_in_library': len(self.method_library)
                },
                'risk_assessment': {
                    'risk_distribution': risk_distribution,
                    'high_risk_users': risk_distribution.get('high', 0) + risk_distribution.get('crisis', 0),
                    'crisis_users': risk_distribution.get('crisis', 0)
                },
                'professional_referrals': {
                    'referral_distribution': referral_stats,
                    'referral_rate': sum(
                        count for status, count in referral_stats.items() 
                        if status != 'none'
                    ) / max(total_users, 1),
                    'successful_referrals': referral_stats.get('attending', 0)
                },
                'system_health': {
                    'memory_efficiency': len(self.conversation_history) / max(total_users, 1),
                    'average_profile_completeness': self._calculate_average_profile_completeness(),
                    'data_quality_score': self._calculate_data_quality_score()
                }
            }

    def _calculate_average_profile_completeness(self) -> float:
        """Calculate how complete user profiles are on average"""
        
        if not self.user_profiles:
            return 0.0
        
        total_completeness = 0.0
        
        for profile in self.user_profiles.values():
            completeness = 0.0
            total_fields = 0
            
            # Check various profile completeness indicators
            fields_to_check = [
                ('primary_concerns', len(profile.primary_concerns) > 0),
                ('severity_trend', len(profile.severity_trend) > 0),
                ('methods_experienced', len(profile.methods_experienced) > 0),
                ('communication_preferences', len(profile.communication_preferences) > 0),
                ('interaction_count', profile.total_interactions > 0)
            ]
            
            for field_name, has_data in fields_to_check:
                total_fields += 1
                if has_data:
                    completeness += 1
            
            total_completeness += completeness / total_fields
        
        return total_completeness / len(self.user_profiles)

    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score"""
        
        if not self.user_profiles:
            return 0.0
        
        quality_score = 0.0
        total_profiles = len(self.user_profiles)
        
        for profile in self.user_profiles.values():
            profile_quality = 0.0
            
            # Check data consistency and quality
            if len(profile.severity_trend) > 1:
                profile_quality += 0.3  # Has trend data
            
            if len(profile.methods_experienced) > 0:
                # Check if methods have feedback
                methods_with_feedback = sum(
                    1 for exp in profile.methods_experienced.values()
                    if exp.effectiveness_ratings
                )
                feedback_ratio = methods_with_feedback / len(profile.methods_experienced)
                profile_quality += 0.4 * feedback_ratio
            
            if profile.total_interactions > 0:
                conversations = self.conversation_history.get(profile.user_id, [])
                if len(conversations) == profile.total_interactions:
                    profile_quality += 0.3  # Data consistency
            
            quality_score += profile_quality
        
        return quality_score / total_profiles