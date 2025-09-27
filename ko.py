"""
Progressive Response Generator - Enhanced with Ollama Llama 3 Integration
Advanced Human-like Response Generation with AI enhancement and comprehensive mental health support
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import threading
from collections import defaultdict
from api_ollama_integration import ollama_llama3

# Import Ollama integration
# Note: You need to have an 'api_ollama_integration.py' file with the necessary functions.
# from api_ollama_integration import ollama_llama3

# Mocking the api_ollama_integration for standalone functionality


class ProgressiveResponseGenerator:
    """
    Advanced response generator with Ollama Llama 3 integration for more natural,
    contextual responses while maintaining mental health safety protocols
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        # Track conversation stages and user preferences for each user
        self.user_conversation_states = {}
        self.user_response_preferences = {}

        # Enhanced response templates organized by conversation stage and intent
        self.stage_responses = {
            'initial_contact': {
                'depression_symptoms': {
                    'responses': [
                        "I can hear how heavy and overwhelming things feel for you right now. Thank you for reaching out.",
                        "Depression can feel so isolating and exhausting. I'm glad you're here to talk about it.",
                        "What you're experiencing sounds really difficult. You're brave for sharing this with me.",
                        "I hear the pain in your words, and I want you to know that what you're feeling is real and valid.",
                        "It takes courage to talk about depression. I'm here to listen and support you through this."
                    ],
                    'follow_ups': [
                        "Can you tell me what's been the hardest part for you lately?",
                        "How long have you been feeling this way?",
                        "What does a typical day look like for you right now?"
                    ]
                },
                'inappropriate_content': {
                    'responses': [
                        "I'm here to help with mental health and emotional support. I can't assist with sexual or inappropriate topics.",
                        "I'm designed to provide mental health support. For personal relationships or other topics, you might want to speak with friends, family, or appropriate professionals.",
                        "I focus on emotional wellbeing and mental health support. Is there something related to your mental health I can help you with instead?",
                        "I'm not able to help with that type of content. I'm here to support your emotional and mental wellbeing though."
                    ],
                    'redirects': [
                        "Is there anything about your emotional wellbeing I can support you with?",
                        "How are you feeling emotionally today? I'm here to help with any mental health concerns.",
                        "I'm here if you need to talk about stress, anxiety, depression, or other mental health topics."
                    ]
                },
                'anxiety_panic': {
                    'responses': [
                        "Anxiety can feel so overwhelming and scary. I understand how exhausting constant worry can be.",
                        "Panic and racing thoughts can make everything feel out of control. You're not alone in this.",
                        "I hear you, and I want you to know that what you're experiencing with anxiety is very real.",
                        "The physical symptoms of anxiety can be terrifying. Thank you for trusting me with this.",
                        "It sounds like anxiety is really impacting your daily life. That must be incredibly draining."
                    ],
                    'follow_ups': [
                        "What tends to trigger your anxiety the most?",
                        "How do these feelings usually show up in your body?",
                        "Have you noticed any patterns in when you feel most anxious?"
                    ]
                },
                'bullying_harassment': {
                    'responses': [
                        "I'm really sorry you're going through this. Bullying is serious and it's absolutely not your fault.",
                        "What you're experiencing isn't okay, and you deserve to feel safe. I'm here to support you.",
                        "Being bullied is traumatic and exhausting. Thank you for having the courage to talk about it.",
                        "Nobody deserves to be treated this way. I want to help you feel safer and more supported.",
                        "Harassment can make you feel so alone and powerless. I'm glad you reached out."
                    ],
                    'follow_ups': [
                        "Can you tell me more about what's been happening?",
                        "How is this affecting your daily life?",
                        "Do you have any trusted adults you've been able to talk to about this?"
                    ]
                },
                'academic_pressure': {
                    'responses': [
                        "Academic stress can feel crushing when everything piles up at once. That sounds overwhelming.",
                        "The pressure to perform academically can be intense. I hear how much this is weighing on you.",
                        "School stress is real and can affect everything in your life. Thank you for sharing this with me.",
                        "When academic expectations feel impossible to meet, it can be incredibly stressful. I understand.",
                        "The fear of academic failure can consume so much mental energy. That sounds exhausting."
                    ],
                    'follow_ups': [
                        "What aspect of school is causing you the most stress right now?",
                        "How are you managing your workload currently?",
                        "What kind of pressure are you feeling from family or yourself?"
                    ]
                },
                'family_conflicts': {
                    'responses': [
                        "Family problems can be really draining, especially when home doesn't feel like a safe space.",
                        "Conflict at home is exhausting because it's supposed to be your refuge. That's really hard.",
                        "Family stress can affect everything else in your life. I can hear how difficult this is for you.",
                        "When the people closest to you feel distant or hostile, it's incredibly painful.",
                        "Home should feel safe and supportive. I'm sorry you're dealing with tension and conflict there."
                    ],
                    'follow_ups': [
                        "What's the family situation like at home right now?",
                        "How long has this conflict been going on?",
                        "How is this affecting your mental health and daily life?"
                    ]
                },
                'social_anxiety': {
                    'responses': [
                        "Social anxiety can make everyday interactions feel terrifying. That takes so much energy to navigate.",
                        "The fear of judgment from others can be paralyzing. I understand how isolating that feels.",
                        "Social situations can feel impossible when you're dealing with anxiety. You're not alone in this.",
                        "It's exhausting to constantly worry about what others think. That sounds really overwhelming.",
                        "Social anxiety can make you feel like an outsider in your own life. That's incredibly lonely."
                    ],
                    'follow_ups': [
                        "What social situations feel most challenging for you?",
                        "How do you usually cope when you have to be in social settings?",
                        "Has this gotten worse recently, or been going on for a while?"
                    ]
                },
                'loneliness_isolation': {
                    'responses': [
                        "Feeling alone can be one of the most painful experiences. I'm glad you're reaching out right now.",
                        "Loneliness hurts in such a real, physical way. Thank you for sharing this with me.",
                        "Isolation can make everything feel hopeless. I want you to know you're not alone in this moment.",
                        "The ache of loneliness is something many people understand. Your feelings are completely valid.",
                        "Feeling disconnected from others can be devastating. I'm here with you right now."
                    ],
                    'follow_ups': [
                        "What does loneliness feel like for you?",
                        "Have you been feeling isolated for a long time?",
                        "What relationships in your life feel most important to you?"
                    ]
                },
                'general_support': {
                    'responses': [
                        "It sounds like you're going through a really difficult time right now. I'm here to listen.",
                        "I hear that things are tough for you. Thank you for sharing this with me.",
                        "Sometimes life can feel overwhelming. I want you to know that reaching out was brave.",
                        "I can sense you're struggling right now. You don't have to go through this alone.",
                        "Whatever you're facing, your feelings are valid and important. I'm here to support you."
                    ],
                    'follow_ups': [
                        "What's been weighing on you the most lately?",
                        "How have you been taking care of yourself during this time?",
                        "What would feel most helpful to talk about right now?"
                    ]
                }
            },
            'understanding': {
                'responses': [
                    "Thank you for sharing more details. I'm starting to understand what you're going through.",
                    "That gives me a clearer picture of your situation. This sounds really challenging.",
                    "I appreciate you opening up about this. It helps me understand how to best support you.",
                    "What you're describing sounds incredibly difficult to navigate alone."
                ],
                'follow_up_questions': [
                    "Can you tell me a bit more about what's been happening?",
                    "How long have you been feeling this way?",
                    "What's been the hardest part for you?",
                    "When did you first notice these feelings?",
                    "Is there anything specific that seems to trigger this?",
                    "How is this affecting other areas of your life?",
                    "What does support look like for you right now?",
                    "Have you experienced anything like this before?"
                ]
            },
            'trust_building': {
                'responses': [
                    "I want you to know that this is a safe space to share whatever you need to.",
                    "You're doing something really brave by talking about these difficult experiences.",
                    "I'm honored that you trust me with something so personal and important.",
                    "Your willingness to be vulnerable shows incredible strength.",
                    "I can see how much courage it takes for you to share these feelings."
                ],
                'validation_statements': [
                    "Your feelings make complete sense given what you're going through.",
                    "What you're experiencing is a normal response to an abnormal situation.",
                    "You're not overreacting - this is genuinely difficult.",
                    "Many people in your situation would feel exactly the same way.",
                    "Your emotions are valid and understandable."
                ]
            },
            'gentle_help_offering': {
                'responses': [
                    "Would it help if I shared a gentle technique that many people find useful for situations like this?",
                    "I know a simple approach that might bring you some relief. Would you like me to share it?",
                    "There's a technique I can teach you that might help with what you're experiencing. Interested?",
                    "I have something that might help you feel a bit more in control. Should I share it with you?",
                    "Would you be open to trying one small thing that might make this a bit easier to manage?",
                    "There's a gentle method that many people find helpful. Would you like to learn about it?"
                ],
                'encouragement': [
                    "You don't have to figure this out all at once. We can take it one step at a time.",
                    "Small steps can lead to meaningful changes. You're already taking the first step by being here.",
                    "You have more strength than you realize. Let's build on that together.",
                    "Healing isn't linear, but every small effort counts. You're already doing something important.",
                    "You deserve support and relief from this pain. Let's work on finding that together."
                ]
            },
            'method_suggestion': {
                'breathing_exercises': {
                    'intro': [
                        "Here's a gentle breathing technique that can help when you're feeling overwhelmed:",
                        "This is a simple but powerful breathing method that many people find calming:",
                        "Let me share a breathing technique that can help your nervous system relax:"
                    ],
                    'instructions': {
                        'basic': [
                            "Find a comfortable place to sit or lie down",
                            "Place one hand on your chest, one on your belly",
                            "Breathe in slowly through your nose for 4 counts",
                            "Hold your breath gently for 4 counts",
                            "Breathe out slowly through your mouth for 6 counts",
                            "Repeat this pattern 5-10 times"
                        ],
                        'advanced': [
                            "Get comfortable and close your eyes if that feels safe",
                            "Notice your natural breathing rhythm first",
                            "Slowly extend your exhales to be longer than your inhales",
                            "Try the 4-4-6 pattern: in for 4, hold for 4, out for 6",
                            "Focus on making the breath smooth and gentle",
                            "Continue for 2-3 minutes or until you feel calmer"
                        ]
                    },
                    'encouragement': [
                        "This helps activate your body's natural relaxation response. Even doing it once can make a difference.",
                        "The longer exhale tells your nervous system it's safe to relax. Be patient with yourself as you practice.",
                        "If you feel dizzy, just breathe normally. Your body will find its own comfortable rhythm.",
                        "This technique works best when practiced regularly, but it can help in the moment too."
                    ]
                },
                'grounding_technique': {
                    'intro': [
                        "When anxiety feels overwhelming, this grounding technique can help bring you back to the present:",
                        "Here's a simple grounding method that uses your senses to help you feel more stable:",
                        "This is called the 5-4-3-2-1 technique, and it can help when you feel disconnected or panicked:"
                    ],
                    'instructions': {
                        'basic': [
                            "Look around and name 5 things you can see",
                            "Touch and name 4 things you can feel (texture, temperature)",
                            "Listen for and name 3 things you can hear",
                            "Identify 2 things you can smell",
                            "Notice 1 thing you can taste"
                        ],
                        'adapted': [
                            "Name 5 things you can see in detail (colors, shapes, textures)",
                            "Name 4 things you can physically touch right now",
                            "Name 3 sounds you can hear (near and far)",
                            "Name 2 scents you notice",
                            "Name 1 thing you can taste or the last thing you drank"
                        ]
                    },
                    'encouragement': [
                        "This helps your mind focus on the present moment instead of worrying thoughts.",
                        "It's okay if some senses are harder to identify than others. Just do what you can.",
                        "The goal is to ground yourself in the here and now, where you are safe.",
                        "Take your time with each step. There's no rush."
                    ]
                },
                'one_small_win': {
                    'intro': [
                        "When everything feels impossible, this approach can help you build momentum:",
                        "Depression can make everything feel overwhelming. This technique breaks things down:",
                        "Here's a gentle way to create positive momentum when motivation is low:"
                    ],
                    'instructions': {
                        'basic': [
                            "Choose ONE very small task (make bed, brush teeth, text one person)",
                            "Do just that one thing - nothing more is required",
                            "Take a moment to acknowledge what you accomplished",
                            "Recognize this as a genuine victory",
                            "Rest if you need to - that's enough for now"
                        ],
                        'advanced': [
                            "Pick the smallest possible version of something helpful",
                            "Set a timer for just 5 minutes if it helps",
                            "Focus only on starting, not finishing perfectly",
                            "Celebrate the attempt, regardless of the outcome",
                            "Notice how it feels to take one small action"
                        ]
                    },
                    'encouragement': [
                        "Small steps are still steps forward. Your effort matters, no matter how small it seems.",
                        "This isn't about being productive - it's about proving to yourself that you can do things.",
                        "Some days, getting out of bed IS the victory. Be gentle with yourself.",
                        "Progress isn't always visible, but every small action builds your strength."
                    ]
                },
                'safety_planning': {
                    'intro': [
                        "Let's work together to create a plan that helps you feel safer in difficult situations:",
                        "Having a safety plan can provide comfort and concrete steps when things feel chaotic:",
                        "Safety planning is about identifying your resources and creating a roadmap for tough times:"
                    ],
                    'instructions': {
                        'basic': [
                            "Identify 2-3 trusted people you can reach out to immediately",
                            "Write down their phone numbers and when they're available",
                            "Think of safe spaces where you can go if needed",
                            "List activities that help you feel calmer or more grounded",
                            "Keep crisis hotline numbers easily accessible"
                        ],
                        'comprehensive': [
                            "Identify your personal warning signs that things are getting difficult",
                            "List internal coping strategies you can use alone",
                            "Identify people and social settings that provide support",
                            "List professionals or agencies you can contact",
                            "Identify ways to make your environment safer",
                            "Plan what to do in a crisis situation"
                        ]
                    },
                    'encouragement': [
                        "Having a plan doesn't mean you're expecting the worst - it means you're taking care of yourself.",
                        "Safety planning is a form of self-compassion and preparation, not pessimism.",
                        "You deserve to feel safe and supported. This plan is about ensuring that.",
                        "Even making this plan is an act of hope and self-care."
                    ]
                },
                'study_stress_management': {
                    'intro': [
                        "Academic stress can feel overwhelming. Here's a systematic approach to make it more manageable:",
                        "Let's break down your academic workload into something more manageable:",
                        "Here's a method that can help reduce the overwhelm of academic pressure:"
                    ],
                    'instructions': {
                        'basic': [
                            "Write down everything you need to do, no matter how small",
                            "Break large assignments into smaller, specific tasks",
                            "Prioritize tasks: urgent/important, important/not urgent, etc.",
                            "Schedule specific times for each task, including breaks",
                            "Start with the smallest or easiest task to build momentum"
                        ],
                        'advanced': [
                            "Use the 25-minute focused study periods (Pomodoro technique)",
                            "Take 5-minute breaks between study sessions",
                            "Set realistic daily goals based on your energy and schedule",
                            "Create a dedicated study space free from distractions",
                            "Plan buffer time for unexpected delays or additional work",
                            "Include time for meals, sleep, and relaxation"
                        ]
                    },
                    'encouragement': [
                        "The goal isn't perfection - it's progress and reducing overwhelm.",
                        "Academic success doesn't define your worth as a person.",
                        "It's okay to adjust your expectations based on your current capacity.",
                        "Taking breaks isn't lazy - it's necessary for sustainable performance."
                    ]
                }
            },
            'method_follow_up': {
                'questions': [
                    "How did that technique work for you?",
                    "Were you able to try what we talked about?",
                    "How are you feeling after giving that approach a try?",
                    "Did you notice any difference when you used that method?",
                    "What was your experience like with that technique?",
                    "How did it feel to practice that approach?",
                    "What part of the technique was most helpful, if any?",
                    "Were there any challenges in trying that method?"
                ],
                'positive_responses': [
                    "That's wonderful that you tried it! Even attempting something new takes courage and shows your commitment to feeling better.",
                    "I'm so glad it helped, even a little bit. You're doing great work on your mental health.",
                    "You should be proud of yourself for taking that step. That's real progress.",
                    "It sounds like you're really working at this, and that effort is making a difference.",
                    "Even small improvements matter. You're building skills that will serve you well."
                ],
                'adjustment_responses': [
                    "That's completely okay - different techniques work for different people. Let's try something else that might be a better fit.",
                    "No worries if that didn't feel right. Everyone's brain and body respond differently to various approaches.",
                    "Thank you for being honest about your experience. That helps me understand what might work better for you.",
                    "It's normal for some methods not to click immediately. Finding what works is a process, and we'll figure it out together."
                ],
                'neutral_responses': [
                    "Thank you for sharing your experience with that technique. How are you feeling about things overall right now?",
                    "I appreciate you trying it. Could you tell me more about what the experience was like for you?",
                    "It sounds like you gave it a genuine effort. What aspects felt most or least helpful?",
                    "Thanks for the feedback. What would feel most useful to explore next?"
                ]
            },
            'ongoing_support': {
                'check_ins': [
                    "How have things been going since we last talked?",
                    "I've been thinking about you. How are you doing with everything?",
                    "What's been on your mind lately? I'm here to listen.",
                    "How are you taking care of yourself these days?",
                    "What's feeling most manageable right now, and what's still challenging?"
                ],
                'encouragement': [
                    "You've shown real strength in working through these challenges.",
                    "I can see how much effort you're putting into your mental health, and that matters.",
                    "Progress isn't always linear, but you're moving forward even when it doesn't feel like it.",
                    "You're learning valuable skills that will help you navigate future challenges.",
                    "Remember that seeking support is a sign of wisdom and self-care, not weakness."
                ],
                'maintenance': [
                    "What strategies have been working well for you?",
                    "How can we build on the progress you've made?",
                    "What would help you maintain these positive changes?",
                    "What warning signs should we watch for?",
                    "What support do you need to continue this journey?"
                ]
            },
            'crisis_intervention': {
                'validation': [
                    "I'm really concerned about you right now, and I want you to know that your life has value and meaning.",
                    "I can hear how much pain you're in, and I'm worried about your safety. You matter, and your life matters.",
                    "This level of emotional pain must be unbearable. I want you to know you're not alone in this moment.",
                    "I'm taking what you're saying very seriously because I care about your wellbeing and safety."
                ],
                'immediate_help': [
                    "Right now, I need you to reach out for immediate help. Here are some options:",
                    "Your safety is the most important thing. Please contact one of these resources immediately:",
                    "I want to connect you with people who are specially trained to help in crisis situations:",
                    "These are people who understand what you're going through and can provide immediate support:"
                ],
                'resources': [
                    "• Emergency Services: 911 (US) or your local emergency number",
                    "• National Suicide Prevention Lifeline: 988 (US) or 1-800-273-8255",
                    "• Crisis Text Line: Text HOME to 741741",
                    "• International Crisis Lines: Available at findahelpline.com",
                    "• Go to your nearest emergency room",
                    "• Call a trusted friend or family member to stay with you"
                ],
                'follow_up': [
                    "Please don't face this crisis alone. Reach out to one of these resources right now.",
                    "You deserve help and support through this crisis. These people want to help you.",
                    "This level of pain is temporary, even though it doesn't feel that way right now.",
                    "Your life has value, and there are people trained to help you through this moment."
                ]
            },
            'professional_referral': {
                'introduction': [
                    "Based on what you've shared, I think talking to a professional counselor could be really beneficial for you.",
                    "A therapist might be able to provide you with additional tools and ongoing support that could help.",
                    "Professional counseling could give you personalized strategies and a consistent supportive relationship.",
                    "Working with a mental health professional could complement what we've been discussing and provide deeper support."
                ],
                'benefits': [
                    "Therapists are trained to help with exactly what you're experiencing.",
                    "You'd have a consistent person to work through these challenges with over time.",
                    "They can provide specialized techniques and approaches tailored to your specific needs.",
                    "Professional therapy offers a confidential space to explore these issues deeply.",
                    "Many people find that therapy helps them develop lasting coping skills and insights."
                ],
                'encouragement': [
                    "Seeking professional help is a sign of strength and self-care, not weakness.",
                    "Many people find therapy incredibly helpful for working through these kinds of challenges.",
                    "You deserve professional support and guidance through this difficult time.",
                    "Taking this step would show how committed you are to your mental health and wellbeing."
                ]
            }
        }

        # Response personalization factors
        self.personalization_factors = {
            'communication_style': {
                'direct': {'modifier': 'straightforward', 'avoid': 'overly_gentle'},
                'supportive': {'modifier': 'warm', 'avoid': 'clinical'},
                'gentle': {'modifier': 'soft', 'avoid': 'overwhelming'},
                'structured': {'modifier': 'organized', 'avoid': 'rambling'}
            },
            'response_length': {
                'short': {'word_limit': 50, 'preference': 'concise'},
                'medium': {'word_limit': 100, 'preference': 'balanced'},
                'long': {'word_limit': 200, 'preference': 'detailed'}
            },
            'emotional_state': {
                'crisis': {'tone': 'calm_urgent', 'priority': 'safety'},
                'severe_distress': {'tone': 'gentle_supportive', 'priority': 'validation'},
                'moderate_distress': {'tone': 'supportive', 'priority': 'help_offering'},
                'mild_distress': {'tone': 'encouraging', 'priority': 'skill_building'}
            }
        }

        # Professional referral database with detailed information
        self.counselor_database = {
            'dr_sarah_smith': {
                'name': 'Dr. Sarah Smith',
                'title': 'Clinical Psychologist',
                'specialties': ['anxiety', 'depression', 'academic_stress', 'young_adults'],
                'approach': 'I use cognitive-behavioral therapy and mindfulness techniques to help people develop practical coping skills.',
                'availability': 'Weekday evenings and Saturday mornings',
                'wait_time': '1-2 weeks',
                'insurance': 'Most major insurance plans accepted',
                'booking_info': 'Call (555) 123-4567 or book online at dr-sarah-smith.com',
                'why_good_fit': 'Specializes in helping college students and young adults with anxiety and academic pressure'
            },
            'dr_michael_jones': {
                'name': 'Dr. Michael Jones',
                'title': 'Licensed Clinical Social Worker',
                'specialties': ['trauma', 'bullying', 'family_therapy', 'crisis_intervention'],
                'approach': 'I provide trauma-informed care with a focus on building safety and empowerment.',
                'availability': 'Weekday afternoons and some evenings',
                'wait_time': '3-5 days',
                'insurance': 'Sliding scale available, most insurance accepted',
                'booking_info': 'Call (555) 234-5678 for intake appointment',
                'why_good_fit': 'Expert in trauma recovery and family conflict resolution'
            },
            'dr_maria_garcia': {
                'name': 'Dr. Maria Garcia',
                'title': 'Licensed Marriage and Family Therapist',
                'specialties': ['social_anxiety', 'relationship_issues', 'self_esteem', 'identity_development'],
                'approach': 'I use a humanistic approach focused on self-acceptance and authentic relationship building.',
                'availability': 'Monday, Wednesday, Friday, and Sunday mornings',
                'wait_time': '1 week',
                'insurance': 'Private pay and some insurance plans',
                'booking_info': 'Contact through secure portal at therapyspace.com/maria-garcia',
                'why_good_fit': 'Specializes in social anxiety and building confidence in relationships'
            }
        }
      # (In ko.py, replace the existing generate_response function)

# (In ko.py, replace the existing generate_response function)

    def generate_response(self,
                      nlu_understanding: Dict[str, Any],
                      user_message: str,
                      user_id: str,
                      language: str = "en",
                      conversation_memory=None,
                      response_options: Dict[str, Any] = None, ollama_client=None) -> Dict[str, Any]:
        """
        DEFINITIVE FIX #2: Resolves the UnboundLocalError by initializing referral_info.
        This version is based on the user's original file structure to ensure stability.
        """
        with self._lock:
            intent = nlu_understanding['primary_intent']
            # --- FIX START: Add a specific handler for inappropriate content ---
            if intent == 'inappropriate_content':
                return {
                'response': "I am a mental health support assistant and cannot engage with requests of that nature. My purpose is to provide a safe space to discuss emotional wellbeing.",
                'conversation_stage': 'initial_contact',
                'intent_addressed': intent,
                'severity_addressed': 0.0,
                'emotional_state_addressed': 'neutral',
                'llama_enhanced': False,
                'timestamp': datetime.now().isoformat()
            }
        # --- FIX END ---
            severity = nlu_understanding['severity_score']
            emotional_state = nlu_understanding['emotional_state']
            urgency_level = nlu_understanding['urgency_level']
            requires_immediate_help = nlu_understanding.get('requires_immediate_help', False)
            
        # --- END OF FIX ---

           

            user_state = self._get_user_state(user_id)
            current_stage = user_state.get('current_stage', 'initial_contact')

        # Determine the next stage of the conversation
            updated_stage = self._update_conversation_stage(
                current_stage, user_state, nlu_understanding, user_message, conversation_memory
                )

            llama_response = None
            response_data = {}

        
        # --- THIS IS THE FIX ---
        # Initialize referral_info to None to prevent the UnboundLocalError
            referral_info = None
        # ---------------------

        # Determine current referral status from memory
            referral_status = None
            last_counselor_id = None
            try:
                if conversation_memory:
                    profile = conversation_memory.create_or_get_user(user_id)
                    referral_status = getattr(profile, 'counselor_referral_status', 'none')
                    if profile.referral_history:
                        last_details = profile.referral_history[-1].get('details', {})
                        last_counselor_id = last_details.get('counselor_id')
            except Exception as _e:
                referral_status = 'none'

        # AI-Enhanced Response Path
            if ollama_client and ollama_client.is_available and intent not in ['inappropriate_content']:
                try:
                    context_history = self._build_conversation_context(conversation_memory, user_id)
                    llama_response = ollama_client.generate_mental_health_response(
                    user_message=user_message, user_intent=intent, conversation_stage=updated_stage,
                    severity_score=severity, context_history=context_history,
                    emotional_state=emotional_state.get('primary_emotion', 'neutral'),
                    urgency_level=urgency_level, 
                    language=language # <-- ADD THIS
                )
                    if llama_response:
                        self.logger.info(f"✅ Generated Llama 3 response for {user_id}")
                    response_data['response'] = llama_response
                except Exception as e:
                    self.logger.warning(f"⚠️ Llama 3 response generation failed: {e}")
                    llama_response = None

        # Rule-Based Fallback Path
                if not llama_response:
                    self.logger.info(f"Using fallback rule-based response for {user_id}")
                    response_data = self._generate_stage_appropriate_response(
                        updated_stage, intent, nlu_understanding, user_state, conversation_memory
                        )
        
        # Post-Generation Processing
                personalized_response = self._apply_personalization(
                    response_data, user_state, nlu_understanding, response_options
                    )

        # Suggest professional help if needed (only when no referral suggested yet)
                if referral_status in ['none'] and self._should_suggest_professional_help(nlu_understanding, user_state, conversation_memory):
                    referral_info = self._generate_professional_referral(intent, nlu_understanding, conversation_memory, user_id)
        
                if referral_info:
                    # Instead of combining the text, we send the referral in its own field
                    personalized_response['professional_referral'] = {
                        'text': referral_info['referral_text'],
                        'is_recommendation': True
                    }
                else:
                    # No new referral: post-recommendation behavior
                    # If referred but not booked yet, add a gentle booking prompt (no repetition of full card)
                    if referral_status in ['suggested', 'interested', 'very_interested']:
                        # Localized booking prompt shown only once, and not on trivial acknowledgements/refusals
                        ack_words = ['ok','okay','okey','haan','ha','yes','theek','ठीक','achha','accha','ji','ठीक है']
                        refusal_words = ['no','nah','nahi','nahin','जी नहीं','नही','नहीं','na','mat','chhodo']
                        short_follow_map = {
                            'hi': "Theek hai. Jab aap taiyaar hon, main booking mein madad kar dunga. Abhi, kya main aapko ek chhota sa technique suggest karun?",
                            'en': "Alright—when you're ready, I can help book. For now, would you like a small technique to try?"
                        }
                        msg_norm = user_message.strip().lower()
                        is_ack = msg_norm in ack_words
                        is_refusal = (msg_norm in refusal_words) or ('aur bata' in msg_norm) or ('और बता' in user_message)
                        if (is_ack or is_refusal):
                            # Send a short non-repeating confirmation once, then stop re-affirming
                            if not user_state.get('referral_affirmation_shown', False):
                                personalized_response['response'] = short_follow_map.get(language, short_follow_map['en'])
                                user_state['referral_affirmation_shown'] = True
                            user_state['booking_prompt_shown'] = True
                            user_state['awaiting_booking_confirmation'] = False
                        elif not user_state.get('booking_prompt_shown', False):
                            booking_prompt_map = {
                                'hi': "Agar aap chahein, main aapko abhi counselor ke saath appointment book karne mein madad kar sakta hoon.",
                                'en': "If you'd like, I can help you book an appointment with a counselor now."
                            }
                            booking_prompt = booking_prompt_map.get(language, booking_prompt_map['en'])
                            personalized_response['response'] += f"\n\n{booking_prompt}"
                            user_state['booking_prompt_shown'] = True
                    # If booked/attending, ask result-based follow-up and respond to positive feedback
                    elif referral_status in ['booked', 'attending', 'ready_to_book']:
                        # Get counselor display name if available
                        counselor_name = None
                        if last_counselor_id and hasattr(conversation_memory, 'counselor_database'):
                            for c in conversation_memory.counselor_database:
                                if c.get('id') == last_counselor_id:
                                    counselor_name = c.get('name')
                                    break
                        who = f" {counselor_name}" if counselor_name else ""
                        check_in_map = {
                            'hi': f"Aapka session kaisa raha? Kya aap{who} se baat karne ke baad kuch badlav mehsoos kar rahe hain?",
                            'en': f"How was your session? Are you noticing any changes since meeting with{who}?"
                        }
                        check_in = check_in_map.get(language, check_in_map['en'])
                        # Simple positive sentiment check on the latest user message
                        msg = user_message.lower()
                        positive_words = ['helped', 'better', 'good', 'useful', 'effective', 'improved', 'calmer', 'relieved', 'बेहतर', 'मदद', 'सुधार']
                        negative_words = ['worse', "didn't help", 'not working', 'bad', 'harder', 'खराब', 'मदद नहीं', 'बिगड़']
                        # Throttle check-ins: not more than once every 3 interactions unless feedback provided
                        last_turn = user_state.get('last_checkin_turn')
                        allow_checkin = (last_turn is None) or (user_state.get('interaction_count', 0) - last_turn >= 3) or any(p in msg for p in positive_words) or any(n in msg for n in negative_words)
                        if allow_checkin:
                            if any(p in msg for p in positive_words):
                                follow_map = {
                                    'hi': f"Yeh sun kar acchha laga. Main hamesha yahan hoon—jo cheez aapke liye kaam kar rahi hai, us par saath milkar aur kaam karte hain. {check_in}",
                                    'en': f"That's encouraging to hear. I'm still here to support you—let's keep building on what's working for you. {check_in}"
                                }
                                personalized_response['response'] += f"\n\n{follow_map.get(language, follow_map['en'])}"
                            elif any(n in msg for n in negative_words):
                                adjust_map = {
                                    'hi': f"Imandari se batane ke liye dhanyavaad. Hum plan ko badal sakte hain ya aapke liye aur behtar counselor dhoondh sakte hain. {check_in}",
                                    'en': f"Thank you for sharing that honestly. We can adjust the plan or find a counselor who fits you better. {check_in}"
                                }
                                personalized_response['response'] += f"\n\n{adjust_map.get(language, adjust_map['en'])}"
                            else:
                                personalized_response['response'] += f"\n\n{check_in}"
                            user_state['last_checkin_turn'] = user_state.get('interaction_count', 0)

        # Update user state and build the final response
                self._update_user_state(user_id, updated_stage, personalized_response, nlu_understanding)

                personalized_response.update({
            'intent_addressed': intent,
            'conversation_stage': updated_stage,
            'severity_addressed': severity,
            'emotional_state_addressed': emotional_state.get('primary_emotion', 'neutral'),
            'llama_enhanced': llama_response is not None,
            'timestamp': datetime.now().isoformat()
        })   
                return personalized_response

    def _build_conversation_context(self, conversation_memory, user_id: str) -> List[Dict[str, str]]:
        """Build conversation context for Llama 3"""
        context = []
        if conversation_memory and hasattr(conversation_memory, 'conversation_history'):
            user_history = conversation_memory.conversation_history.get(user_id, [])
            recent_turns = user_history[-6:] if len(user_history) >= 6 else user_history
            for turn in recent_turns:
                context.append({"role": "user", "content": turn.user_message})
                context.append({"role": "assistant", "content": turn.bot_response})
        return context

    def _response_provides_help(self, response: str) -> bool:
        """Determine if response provides concrete help"""
        help_indicators = [
            'try', 'practice', 'technique', 'exercise', 'step', 'method',
            'strategy', 'approach', 'consider', 'might help', 'could help'
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in help_indicators)

    def _should_suggest_method(self, intent: str, severity: float, stage: str) -> bool:
        """Determine if a method should be suggested"""
        return (
            stage in ['method_suggestion', 'gentle_help_offering'] and
            intent not in ['inappropriate_content', 'general_support'] and
            severity > 0.3
        )

    def _generate_method_suggestion(self, intent: str, severity: float, conversation_memory, user_id: str) -> Optional[Dict[str, Any]]:
        """Generate method suggestion text"""
        if not conversation_memory:
            return None
        try:
            method_suggestion = conversation_memory.suggest_method(user_id, intent, severity)
            if method_suggestion:
                method_data = method_suggestion['method_data']
                method_text = f"\n\n**Here's a technique that might help:**\n\n**{method_data['name']}**\n\n{method_data.get('description', '')}"
                if 'instructions' in method_data:
                    instructions = method_data['instructions'].get('basic', [])
                    if instructions:
                        method_text += "\n\n**How to do it:**\n"
                        for i, instruction in enumerate(instructions[:5], 1):
                            method_text += f"{i}. {instruction}\n"
                return {
                    'method_id': method_suggestion['method_id'],
                    'method_data': method_data,
                    'method_text': method_text
                }
        except Exception as e:
            self.logger.error(f"Error generating method suggestion: {e}")
        return None

    def _generate_immediate_method_response(self, nlu_understanding: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Generate immediate method response for direct help requests"""
        intent = nlu_understanding.get('primary_intent', 'general_support')
        method_mapping = {
            'anxiety_panic': 'breathing_exercises',
            'depression_symptoms': 'one_small_win',
            'academic_pressure': 'study_stress_management',
            'bullying_harassment': 'safety_planning'
        }
        method = method_mapping.get(intent, 'breathing_exercises')
        method_data = self.stage_responses['method_suggestion'][method]
        intro = "Here are some practical steps you can try:"
        instructions = method_data['instructions']['basic']
        response_parts = [intro, ""]
        response_parts.extend([f"**{i+1}.** {step.split('. ')[1] if '. ' in step else step}" for i, step in enumerate(instructions)])
        response_parts.extend(["", "Try this and let me know how it goes!"])
        return {
            'response': "\n".join(response_parts),
            'conversation_stage': 'method_suggestion',
            'method_suggested': method,
            'provides_concrete_help': True,
            'immediate_help': True
        }

    def _get_user_state(self, user_id: str) -> Dict[str, Any]:
        """Get or initialize user conversation state"""
        if user_id not in self.user_conversation_states:
            self.user_conversation_states[user_id] = {
                'current_stage': 'initial_contact',
                'interaction_count': 0,
                'last_interaction': None,
                'successful_methods': [],
                'unsuccessful_methods': [],
                'preferences': {
                    'response_length': 'medium',
                    'communication_style': 'supportive',
                    'detail_level': 'balanced'
                },
                # Booking UX state flags (session-scoped)
                'booking_prompt_shown': False,
                'awaiting_booking_confirmation': False,
                'last_checkin_turn': None,
                'referral_affirmation_shown': False
            }
        return self.user_conversation_states[user_id]

    def _is_method_feedback(self, message: str, conversation_memory, user_id: str) -> bool:
        """Determine if message contains feedback about a suggested method"""
        feedback_indicators = [
            'tried', 'worked', 'helped', 'didnt work', 'not working',
            'better', 'worse', 'useful', 'effective', 'good', 'bad',
            'practiced', 'attempted', 'used the', 'did the'
        ]
        message_lower = message.lower()
        if any(indicator in message_lower for indicator in feedback_indicators):
            if conversation_memory and hasattr(conversation_memory, 'should_follow_up_on_method'):
                follow_up_check = conversation_memory.should_follow_up_on_method(user_id)
                if follow_up_check.get('should_follow_up'):
                    return True
        return False

    def _generate_crisis_response(self, nlu_understanding: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Generate immediate crisis intervention response"""
        validation = random.choice(self.stage_responses['crisis_intervention']['validation'])
        help_intro = random.choice(self.stage_responses['crisis_intervention']['immediate_help'])
        resources = self.stage_responses['crisis_intervention']['resources']
        follow_up = random.choice(self.stage_responses['crisis_intervention']['follow_up'])
        response_parts = [validation, "", help_intro, ""]
        response_parts.extend(resources)
        response_parts.extend(["", follow_up])
        return {
            'response': "\n".join(response_parts),
            'is_crisis_response': True,
            'immediate_action_required': True,
            'conversation_stage': 'crisis_intervention',
            'provides_concrete_help': True,
            'urgency_level': 'crisis',
            'requires_immediate_follow_up': True,
            'resources_provided': True,
            'ai_generated': False
        }

    def _generate_method_feedback_response(self, user_message: str, user_id: str,
                                           nlu_understanding: Dict[str, Any],
                                           conversation_memory) -> Dict[str, Any]:
        """Generate response to user feedback about suggested methods"""
        message_lower = user_message.lower()
        user_state = self._get_user_state(user_id)
        positive_indicators = ['helped', 'better', 'good', 'worked', 'useful', 'effective', 'easier']
        negative_indicators = ['didnt help', 'not working', 'worse', 'harder', 'useless', 'bad']
        positive_count = sum(1 for indicator in positive_indicators if indicator in message_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in message_lower)

        if positive_count > negative_count:
            responses = self.stage_responses['method_follow_up']['positive_responses']
            user_state['successful_methods'].append('recent_method')
            next_stage = 'ongoing_support'
            follow_up = " Let's continue building on what's working for you. How are you feeling about using this technique going forward?"
        elif negative_count > positive_count:
            responses = self.stage_responses['method_follow_up']['adjustment_responses']
            user_state['unsuccessful_methods'].append('recent_method')
            next_stage = 'method_suggestion'
            follow_up = " Let's try a different approach that might be a better fit. What felt most challenging about that method?"
        else:
            responses = self.stage_responses['method_follow_up']['neutral_responses']
            next_stage = 'understanding'
            follow_up = " I'd love to understand more about your experience so I can better support you."
            
        base_response = random.choice(responses)
        return {
            'response': base_response + follow_up,
            'is_follow_up': True,
            'conversation_stage': next_stage,
            'feedback_processed': True,
            'method_effectiveness': 'positive' if positive_count > negative_count else 'negative' if negative_count > positive_count else 'mixed'
        }

    # In ko.py

    # This is the final, most robust version of the function.

    def _update_conversation_stage(self, detected_stage: str, user_state: Dict[str, Any],
                               nlu_understanding: Dict[str, Any], user_message: str, conversation_memory) -> str:
        """
        Update conversation stage using a hybrid approach that is both responsive and stable."""
    # --- 1. High-Priority Overrides (using real-time NLU data) ---
        if detected_stage == 'crisis_intervention' or nlu_understanding.get('requires_immediate_help'):
            return 'crisis_intervention'
        if self._user_wants_help(user_message, nlu_understanding):
            self.logger.info("Direct help-seeking detected - forcing progression to method suggestion.")
            return 'method_suggestion'
         
        if detected_stage == 'method_follow_up':
            return 'method_follow_up'

    # --- 2. Standard Progression Logic (using saved user history) ---
        current_stage = user_state.get('current_stage', 'initial_contact')
        interaction_count = user_state.get('interaction_count', 0)
        severity = nlu_understanding.get('severity_score', 0.0)

        if current_stage in ['understanding', 'trust_building'] and interaction_count > 2:
            self.logger.info("Proactively moving to help offering after building rapport.")
            return 'gentle_help_offering'
        if (severity > 0.7 and interaction_count > 4 and len(user_state.get('unsuccessful_methods', [])) > 0):
            return 'professional_referral'
    
    # --- 3. Low Confidence Safety Net (re-integrated) ---
        confidence = nlu_understanding.get('confidence', 0.0)
        if confidence < 0.4 and current_stage not in ['initial_contact', 'crisis_intervention']:
            self.logger.info(f"NLU confidence is low ({confidence:.2f}). Returning to 'understanding' stage to clarify.")
            return 'understanding'

    # --- 4. Default Step-by-Step Progression ---
        stage_progression = {
        'initial_contact': 'understanding',
        'understanding': 'trust_building',
        'trust_building': 'gentle_help_offering',
        'gentle_help_offering': 'method_suggestion',
        'method_suggestion': 'method_follow_up',
        'method_follow_up': 'ongoing_support',
        'ongoing_support': 'ongoing_support',
        'crisis_intervention': 'ongoing_support',
        'professional_referral': 'ongoing_support'
    }
    
        return stage_progression.get(current_stage, 'understanding')

    # THIS IS THE NEW, CORRECTED FUNCTION
    def _user_wants_help(self, user_message: str, nlu_understanding: Dict[str, Any]) -> bool:
        """
        Intelligently determine if the user is actively seeking a new method or guidance.
        Ignores short, simple replies to prevent misinterpretation.
        """
    # First, ignore very short messages that are likely answers to questions
        if len(user_message.split()) <= 2:
            return False

        intent = nlu_understanding.get('primary_intent', '')
        if intent == 'help_seeking':
            return True

    # Check for explicit phrases in the user's message
        explicit_help_phrases = [
            'can you help', 'help me with', 'what should i do', 'give me steps',
            'suggest something', 'teach me', 'show me how', 'need advice'
            ]
        message_lower = user_message.lower()
        if any(phrase in message_lower for phrase in explicit_help_phrases):
            return True

        user_needs = nlu_understanding.get('user_needs', [])
        help_needs = ['coping_strategies', 'actionable_guidance', 'techniques', 'methods']
        if any(need in help_needs for need in user_needs):
            
            return True

        return False

    def _generate_stage_appropriate_response(self, stage: str, intent: str,
                                             nlu_understanding: Dict[str, Any],
                                             user_state: Dict[str, Any],
                                             conversation_memory) -> Dict[str, Any]:
        """Generate response appropriate for the conversation stage and intent"""
        # --- FIX START: Add handler for the new out_of_scope intent ---
        if intent == 'out_of_scope':
            return {
            'response': "I am a mental health support assistant. My purpose is to help with emotional wellbeing. How are you feeling today?",
            'conversation_stage': 'initial_contact', # Reset the stage
            'provides_support': False
        }
    # --- FIX END ---
        if stage == 'initial_contact':
            return self._generate_initial_contact_response(intent, nlu_understanding)
        elif stage == 'understanding':
            return self._generate_understanding_response(intent, nlu_understanding)
        elif stage == 'trust_building':
            return self._generate_trust_building_response(nlu_understanding, user_state)
        elif stage == 'gentle_help_offering':
            return self._generate_help_offering_response(intent, nlu_understanding, user_state)
        elif stage == 'method_suggestion':
            return self._generate_method_suggestion_response(intent, nlu_understanding, user_state, conversation_memory)
        elif stage == 'method_follow_up':
            return self._generate_method_follow_up_response(user_state, conversation_memory)
        elif stage == 'ongoing_support':
            return self._generate_ongoing_support_response(nlu_understanding, user_state)
        elif stage == 'professional_referral':
            return self._generate_professional_referral_response(intent, nlu_understanding, user_state)
        else:
            return self._generate_supportive_fallback_response(nlu_understanding)

    def _generate_initial_contact_response(self, intent: str, nlu_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial contact response with empathy and validation"""
        intent_responses = self.stage_responses['initial_contact'].get(intent, self.stage_responses['initial_contact']['general_support'])
        main_response = random.choice(intent_responses['responses'])
        follow_up = random.choice(intent_responses.get('follow_ups', [""]))
        response = f"{main_response} {follow_up}" if follow_up else main_response
        return {
            'response': response,
            'conversation_stage': 'initial_contact',
            'provides_validation': True,
            'builds_rapport': True,
            'asks_follow_up_question': bool(follow_up)
        }

    def _generate_understanding_response(self, intent: str, nlu_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Generate understanding and exploration response"""
        base_response = random.choice(self.stage_responses['understanding']['responses'])
        follow_up_question = random.choice(self.stage_responses['understanding']['follow_up_questions'])
        return {
            'response': f"{base_response} {follow_up_question}",
            'conversation_stage': 'understanding',
            'explores_situation': True,
            'asks_follow_up_question': True,
            'gathers_information': True
        }

    def _generate_trust_building_response(self, nlu_understanding: Dict[str, Any], user_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trust-building and validation response"""
        main_response = random.choice(self.stage_responses['trust_building']['responses'])
        validation = random.choice(self.stage_responses['trust_building']['validation_statements'])
        return {
            'response': f"{main_response} {validation}",
            'conversation_stage': 'trust_building',
            'builds_trust': True,
            'provides_validation': True,
            'normalizes_experience': True
        }

    def _generate_help_offering_response(self, intent: str, nlu_understanding: Dict[str, Any], user_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate gentle help offering response"""
        offer = random.choice(self.stage_responses['gentle_help_offering']['responses'])
        encouragement = random.choice(self.stage_responses['gentle_help_offering']['encouragement'])
        return {
            'response': f"{offer}\n\n{encouragement}",
            'conversation_stage': 'gentle_help_offering',
            'offers_help': True,
            'provides_encouragement': True,
            'respects_autonomy': True
        }

    def _generate_method_suggestion_response(self, intent: str, nlu_understanding: Dict[str, Any],
                                             user_state: Dict[str, Any], conversation_memory) -> Dict[str, Any]:
        """Generate specific method suggestion response"""
        method_mapping = {
            'anxiety_panic': 'breathing_exercises',
            'depression_symptoms': 'one_small_win',
            'bullying_harassment': 'safety_planning',
            'academic_pressure': 'study_stress_management',
            'social_anxiety': 'grounding_technique',
            'overwhelming_thoughts': 'grounding_technique',
            'help_seeking': 'breathing_exercises',
            'general_support': 'breathing_exercises'
        }
        method = method_mapping.get(intent, 'breathing_exercises')
        unsuccessful = user_state.get('unsuccessful_methods', [])
        if method in unsuccessful:
            alternatives = {'breathing_exercises': 'grounding_technique', 'grounding_technique': 'breathing_exercises'}
            method = alternatives.get(method, 'breathing_exercises')
            
        method_data = self.stage_responses['method_suggestion'][method]
        intro = random.choice(method_data['intro'])
        instructions = method_data['instructions']['basic']
        encouragement = random.choice(method_data['encouragement'])
        response_parts = [intro, ""]
        response_parts.extend([f"**{i+1}.** {step.split('. ')[1] if '. ' in step else step}" for i, step in enumerate(instructions)])
        response_parts.extend(["", encouragement])
        
        return {
            'response': "\n".join(response_parts),
            'conversation_stage': 'method_suggestion',
            'method_suggested': method,
            'provides_concrete_help': True,
            'includes_instructions': True
        }

    def _generate_method_follow_up_response(self, user_state: Dict[str, Any], conversation_memory) -> Dict[str, Any]:
        """Generate method follow-up response"""
        return {
            'response': random.choice(self.stage_responses['method_follow_up']['questions']),
            'conversation_stage': 'method_follow_up',
            'is_follow_up': True,
            'seeks_feedback': True
        }

    def _generate_ongoing_support_response(self, nlu_understanding: Dict[str, Any], user_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ongoing support response"""
        check_in = random.choice(self.stage_responses['ongoing_support']['check_ins'])
        encouragement = random.choice(self.stage_responses['ongoing_support']['encouragement'])
        return {
            'response': f"{check_in} {encouragement}",
            'conversation_stage': 'ongoing_support',
            'provides_ongoing_support': True
        }

    def _generate_professional_referral_response(self, intent: str, nlu_understanding: Dict[str, Any], user_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate professional referral response"""
        introduction = random.choice(self.stage_responses['professional_referral']['introduction'])
        benefit = random.choice(self.stage_responses['professional_referral']['benefits'])
        encouragement = random.choice(self.stage_responses['professional_referral']['encouragement'])
        return {
            'response': f"{introduction} {benefit} {encouragement}",
            'conversation_stage': 'professional_referral',
            'suggests_professional_help': True,
            'is_recommendation': True
        }

    def _generate_supportive_fallback_response(self, nlu_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Generate supportive fallback response for unknown situations"""
        return {
            'response': random.choice([
                "I hear you, and I want you to know that I'm here to support you through whatever you're experiencing.",
                "Thank you for sharing this with me. Your feelings are valid, and you deserve support.",
                "It sounds like you're going through something difficult. I'm here to listen and help however I can."
            ]),
            'conversation_stage': 'general_support',
            'provides_support': True
        }

    def _apply_personalization(self, response_data: Dict[str, Any], user_state: Dict[str, Any],
                             nlu_understanding: Dict[str, Any], response_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply personalization to response based on user preferences and state"""
        preferences = user_state.get('preferences', {})
        response_text = response_data.get('response', '')
        length_pref = preferences.get('response_length', 'medium')
        if length_pref == 'short' and len(response_text.split()) > 50:
            sentences = response_text.split('.')
            response_text = '. '.join(sentences[:2]) + '.'
        elif length_pref == 'long' and len(response_text.split()) < 75:
            response_text += " I want you to know that you're not alone in this, and it's okay to take things one step at a time."
        
        style_pref = preferences.get('communication_style', 'supportive')
        if style_pref == 'direct' and 'I think' in response_text:
            response_text = response_text.replace('I think', 'It would be beneficial to')
        elif style_pref == 'gentle':
            response_text = response_text.replace('should', 'might want to').replace('need to', 'could consider')

        response_data['response'] = response_text
        response_data['personalization_applied'] = True
        return response_data

    def _should_suggest_professional_help(self, nlu_understanding: Dict[str, Any], 
                                          user_state: Dict[str, Any], conversation_memory) -> bool:
        """Determine if professional help should be suggested"""
        severity = nlu_understanding.get('severity_score', 0.0)
        interaction_count = user_state.get('interaction_count', 0)
        unsuccessful_methods = len(user_state.get('unsuccessful_methods', []))
        if severity > 0.7 and interaction_count > 3:
            return True
        if unsuccessful_methods > 2:
            return True
        if conversation_memory and hasattr(conversation_memory, 'get_user_summary'):
            user_summary = conversation_memory.get_user_summary(user_state.get('user_id', ''))
            if user_summary.get('exists') and user_summary.get('profile', {}).get('risk_level') in ['high', 'crisis']:
                return True
        return False

    def _generate_professional_referral(self, intent: str, nlu_understanding: Dict[str, Any],
                                        conversation_memory, user_id: str) -> Dict[str, Any]:
        """Generate detailed professional referral information"""
        user_concerns = [intent]
        if conversation_memory and hasattr(conversation_memory, 'get_user_summary'):
            user_summary = conversation_memory.get_user_summary(user_id)
            if user_summary.get('exists'):
                user_concerns = user_summary.get('profile', {}).get('primary_concerns', [intent])
        
        best_match = self._find_best_counselor_match(user_concerns)
        referral_text = f"""**Professional Support Recommendation**

Based on what you've shared, I think working with **{best_match['name']}** could be really helpful.

**Why this might be a good fit:**
• {best_match['why_good_fit']}
• **Approach:** {best_match['approach']}
• **Availability:** {best_match['availability']}

**Next steps:** {best_match['booking_info']}"""
        
        return {
            'counselor_info': best_match,
            'referral_text': referral_text
        }

    def _find_best_counselor_match(self, user_concerns: List[str]) -> Dict[str, Any]:
        """Find the best counselor match for user concerns"""
        concern_to_specialty = {
            'anxiety_panic': ['anxiety'], 'depression_symptoms': ['depression'],
            'bullying_harassment': ['trauma', 'bullying'], 'academic_pressure': ['academic_stress'],
            'family_conflicts': ['family_therapy'], 'social_anxiety': ['social_anxiety']
        }
        counselor_scores = defaultdict(int)
        for counselor_id, data in self.counselor_database.items():
            for concern in user_concerns:
                needed = concern_to_specialty.get(concern, [concern])
                if any(spec in data['specialties'] for spec in needed):
                    counselor_scores[counselor_id] += 1
        
        if counselor_scores:
            best_id = max(counselor_scores, key=counselor_scores.get)
            return self.counselor_database[best_id]
        return list(self.counselor_database.values())[0]

    def _update_user_state(self, user_id: str, stage: str, response_data: Dict[str, Any], 
                          nlu_understanding: Dict[str, Any]):
        """Update user conversation state"""
        user_state = self._get_user_state(user_id)
        user_state['current_stage'] = stage
        user_state['interaction_count'] += 1
        user_state['last_interaction'] = datetime.now()
        if response_data.get('method_suggested'):
            user_state.setdefault('methods_suggested', []).append(response_data['method_suggested'])

    def _calculate_response_confidence(self, nlu_confidence: float, stage: str, 
                                       response_data: Dict[str, Any]) -> float:
        """Calculate confidence in generated response"""
        base_confidence = nlu_confidence * 0.6
        stage_confidence = {
            'initial_contact': 0.8, 'understanding': 0.7, 'trust_building': 0.8,
            'gentle_help_offering': 0.7, 'method_suggestion': 0.6,
            'method_follow_up': 0.8, 'ongoing_support': 0.7,
            'crisis_intervention': 1.0, 'professional_referral': 0.9
        }.get(stage, 0.6)
        quality_bonus = 0.0
        if response_data.get('provides_concrete_help'): quality_bonus += 0.1
        if response_data.get('provides_validation'): quality_bonus += 0.05
        return min(base_confidence + stage_confidence * 0.3 + quality_bonus, 1.0)

    def _determine_follow_up_needs(self, nlu_understanding: Dict[str, Any],
                                   response_data: Dict[str, Any], conversation_memory) -> Dict[str, Any]:
        """Determine what follow-up is needed"""
        needs = {'immediate': False, 'check_in_hours': None, 'type': None}
        if response_data.get('is_crisis_response'):
            needs.update({'immediate': True, 'check_in_hours': 1, 'type': 'crisis'})
        elif response_data.get('method_suggested'):
            needs.update({'check_in_hours': 72, 'type': 'method'})
        elif response_data.get('professional_referral'):
            needs.update({'check_in_hours': 168, 'type': 'referral'})
        elif nlu_understanding.get('severity_score', 0) > 0.7:
            needs.update({'check_in_hours': 48, 'type': 'severity_check'})
        return needs