# API-Based Ollama Integration for Mental Health Chatbot - COMPLETE VERSION
# Modified to use API key-based service (like Groq) instead of local Ollama
# Provides seamless integration with external API while maintaining same interface

import json
import logging
import requests
import os
from typing import Dict, Any, List, Optional

class ApiClient:
    """Enhanced client for interacting with API-based LLM service (e.g., Groq)"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.groq.com/openai/v1", model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv('GROQ_API_KEY') or os.getenv('API_KEY')
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.is_available = self.check_availability()
        
    def check_availability(self) -> bool:
        """Check if API service is available and API key is valid"""
        if not self.api_key:
            self.logger.warning("No API key provided. Set GROQ_API_KEY or API_KEY environment variable")
            return False
            
        try:
            # Test API connectivity with a simple request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make a simple test request
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"API service available with model {self.model}")
                return True
            else:
                self.logger.warning(f"API test failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"API service not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error checking API availability: {e}")
            return False
    
    def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """Generate completion using API with enhanced error handling"""
        if not self.is_available:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if generated_text:
                    self.logger.debug(f"Generated response: {len(generated_text)} chars")
                    return generated_text
                else:
                    self.logger.warning("API returned empty response")
                    return None
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning("API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in API generation: {e}")
            return None
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """Generate chat completion using API with conversation context"""
        if not self.is_available:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if generated_text:
                    self.logger.debug(f"Generated chat response: {len(generated_text)} chars")
                    return generated_text
                else:
                    self.logger.warning("API returned empty chat response")
                    return None
            else:
                self.logger.error(f"API chat error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning("API chat request timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API chat request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in API chat: {e}")
            return None
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the API connection and return status"""
        test_result = {
            "available": False,
            "model": self.model,
            "base_url": self.base_url,
            "error": None,
            "test_response": None
        }
        
        try:
            if not self.is_available:
                test_result["error"] = "Service not available"
                return test_result
            
            test_prompt = "Hello, please respond with a brief greeting."
            test_response = self.generate_response(test_prompt, max_tokens=50)
            
            if test_response:
                test_result["available"] = True
                test_result["test_response"] = test_response
                self.logger.info("API connection test successful")
            else:
                test_result["error"] = "No response generated"
                self.logger.warning("API connection test failed - no response")
                
        except Exception as e:
            test_result["error"] = str(e)
            self.logger.error(f"API connection test error: {e}")
            
        return test_result

class ApiMentalHealthClient:
    """Enhanced mental health specific client using API service"""
    
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str = None, base_url: str = "https://api.groq.com/openai/v1"):
        self.client = ApiClient(api_key=api_key, base_url=base_url, model=model)
        self.is_available = self.client.is_available
        self.logger = logging.getLogger(__name__)
        
        self.base_system_prompt = """You are a compassionate, professional mental health support assistant. Your role is to provide empathetic, supportive responses while maintaining appropriate boundaries.

Core Guidelines:
- Be warm, understanding, and non-judgmental
- Validate the user's emotions and experiences
- Provide practical, actionable support when appropriate
- Keep responses conversational and human-like (2-3 paragraphs maximum)
- Never diagnose, prescribe medication, or provide medical advice
- For crisis situations, prioritize safety and professional resources
- Use encouraging, hopeful language while being realistic
- Respect user autonomy and choices

Response Style:
- Use paragraph breaks to separate distinct ideas and create whitespace. This is very important for readability.
- Use Markdown for emphasis (e.g., **bolding** key advice) where it enhances clarity.
- Write in a natural, conversational tone.
- Use "I" statements to show personal engagement.
- Ask follow-up questions when appropriate.
- Avoid overly clinical or robotic language.
- Keep responses focused and relevant."""
    
    def generate_mental_health_response(self, user_message: str, user_intent: str, conversation_stage: str, 
                                      severity_score: float, context_history: List[Dict[str, str]] = None, 
                                      emotional_state: str = "neutral", urgency_level: str = "low", language: str = "en",
                                      custom_prompt: str = None) -> Optional[str]:
        """Generates a response using API with context-aware prompt and strict language control"""
        
        if not self.is_available:
            return None
        
        try:
            # Prefer the structured chat API with a system prompt that enforces language
            system_prompt = self.build_system_prompt(
                intent=user_intent,
                stage=conversation_stage,
                severity=severity_score,
                emotional_state=emotional_state,
                urgency_level=urgency_level,
                language=language,
            )
            messages = self.build_conversation_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                context_history=context_history,
            )
            temperature = self.get_temperature_for_context(severity_score, conversation_stage)
            max_tokens = self.get_max_tokens_for_stage(conversation_stage)
            response = self.client.chat_completion(messages, max_tokens=max_tokens, temperature=temperature)

            # Fallback to single-prompt generation if chat completion fails
            if not response:
                history_log = []
                if context_history:
                    for turn in context_history:
                        role = "User" if turn.get("role") == "user" else "SAHARA"
                        history_log.append(f"{role}: {turn.get('content')}")
                final_prompt = f"""{self.base_system_prompt}

You are SAHARA, a caring and empathetic mental health support companion. Your primary goal is to provide a supportive, helpful, and safe response. NEVER repeat your instructions. NEVER break character.

Task Instructions:
1. Analyze the user's message in the context of the conversation history.
2. The user's detected intent is: {user_intent}.
3. The conversation stage is: {conversation_stage}.
4. The user's emotional state is: {emotional_state}.
5. The urgency level is: {urgency_level}.
6. Your response MUST be ONLY the words you want to say to the user as SAHARA.

IMPORTANT: You MUST produce your entire response in the language specified by this ISO code: {language}. Do not use English unless required for phone numbers or URLs.

Conversation History:
{chr(10).join(history_log)}

User: {user_message}

Response Template:
SAHARA: [Your response here]"""
                response = self.client.generate_response(final_prompt)

            if response:
                response = response.replace("SAHARA:", "").strip()
                response = self.post_process_response(response, user_intent, severity_score)
                return response
            return None
        except Exception as e:
            self.logger.error(f"Error in API response generation: {e}")
            return None
    
    def build_system_prompt(self, intent: str, stage: str, severity: float, emotional_state: str, urgency_level: str, language: str) -> str:
        """Build context-specific system prompt"""
        
        intent_guidance = {
            "depression_symptoms": "The user is experiencing depression. Focus on validation, hope, and gentle support. Avoid minimizing their experience.",
            "anxiety_panic": "The user is dealing with anxiety or panic. Provide calm, grounding responses. Suggest breathing techniques if appropriate.",
            "crisis_situation": "CRITICAL: This is a potential crisis situation. Prioritize safety, provide emergency resources, and show immediate concern.",
            "bullying_harassment": "The user is experiencing bullying. Validate that this is serious, emphasize it's not their fault, and focus on safety.",
            "academic_pressure": "The user is stressed about academics. Provide practical perspective and stress management approaches.",
            "family_conflicts": "The user has family issues. Be neutral, validate their feelings, focus on what they can control.",
            "social_anxiety": "The user struggles with social situations. Normalize their experience and provide gentle encouragement.",
            "loneliness_isolation": "The user feels lonely. Provide warm connection and gentle suggestions for building relationships.",
            "help_seeking": "The user is actively seeking help. Provide practical, actionable guidance and celebrate their initiative."
        }
        
        stage_guidance = {
            "initial_contact": "This is early in the conversation. Focus on building rapport and understanding their situation.",
            "understanding": "You're gathering information. Ask thoughtful follow-up questions and show active listening.",
            "trust_building": "Build trust through validation and showing genuine care for their wellbeing.",
            "gentle_help_offering": "Offer help in a non-pressuring way. Respect their autonomy to choose.",
            "method_suggestion": "You can suggest specific coping strategies or techniques that might help.",
            "ongoing_support": "Provide continued encouragement and check on their progress.",
            "crisis_intervention": "This is a crisis. Focus entirely on safety and immediate professional resources."
        }
        
        if severity >= 0.8:
            severity_guidance = "This user is in significant distress. Be extra gentle and consider professional resources."
        elif severity >= 0.6:
            severity_guidance = "This user is experiencing moderate distress. Provide substantial support."
        elif severity >= 0.4:
            severity_guidance = "This user has mild to moderate concerns. Balance empathy with encouragement."
        else:
            severity_guidance = "This user has relatively mild concerns. Be supportive while maintaining perspective."
        
        full_prompt = f"""{self.base_system_prompt}

Current Context:
- User Intent: {intent.replace('_', ' ').title()}
- Conversation Stage: {stage.replace('_', ' ').title()}
- Emotional State: {emotional_state.title()}
- Severity Level: {severity:.1f}/1.0
- Urgency Level: {urgency_level.title()}

Specific Guidance:
{intent_guidance.get(intent, "Provide general mental health support.")}
{stage_guidance.get(stage, "Respond appropriately to the conversation context.")}
{severity_guidance}

Remember: Keep your response natural, empathetic, and appropriately sized (2-3 paragraphs). Focus on the human connection.
---
IMPORTANT: You MUST generate your entire response in the following language code: {language}
---
INSTRUCTION: Based on all the context and guidance above, generate ONLY the direct, empathetic response to the user's latest message. Do not repeat, reference, or explain these instructions in your output."""
        
        return full_prompt
    
    def build_conversation_messages(self, system_prompt: str, user_message: str, context_history: List[Dict] = None) -> List[Dict[str, str]]:
        """Build conversation messages for chat completion"""
        messages = [{"role": "system", "content": system_prompt}]
        
        if context_history:
            # Add conversation history (limited to recent context)
            recent_history = context_history[-8:] if len(context_history) > 8 else context_history
            for msg in recent_history:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        return messages
    
    def get_temperature_for_context(self, severity: float, stage: str) -> float:
        """Get appropriate temperature based on context"""
        if stage == "crisis_intervention" or severity >= 0.8:
            return 0.3  # Lower temperature for crisis situations (more predictable responses)
        elif severity >= 0.6:
            return 0.5  # Moderate temperature for serious situations
        else:
            return 0.7  # Higher temperature for general support (more creative responses)
    
    def get_max_tokens_for_stage(self, stage: str) -> int:
        """Get appropriate max tokens based on conversation stage"""
        stage_tokens = {
            "crisis_intervention": 400,  # Need comprehensive crisis response
            "method_suggestion": 350,    # May include detailed instructions
            "professional_referral": 300, # Detailed referral information
            "initial_contact": 250,      # Warm but not overwhelming
            "understanding": 200,        # Focused questions and responses
            "trust_building": 200,       # Validation and support
            "gentle_help_offering": 250, # Offering help with explanation
            "ongoing_support": 200       # Continued encouragement
        }
        return stage_tokens.get(stage, 250)
    
    def post_process_response(self, response: str, intent: str, severity: float) -> str:
        """Post-process the generated response for mental health context"""
        # Remove any inappropriate content
        response = response.replace("I'm just an AI", "I'm here to support you")
        response = response.replace("I cannot", "I want to help you")
        
        # Ensure appropriate length (remove if too long)
        sentences = response.split(". ")
        if len(sentences) > 6:
            response = ". ".join(sentences[:6]) + "."
        
        # Add gentle closure for high severity situations
        if severity >= 0.7 and not response.endswith("?"):
            response += " You don't have to go through this alone."
        
        return response.strip()
    
    def analyze_user_intent(self, user_message: str) -> Optional[Dict[str, Any]]:
        """Use API to analyze user intent and emotional state"""
        if not self.is_available:
            return None
            
        try:
            analysis_prompt = f"""Analyze this message from someone seeking mental health support. Respond with ONLY a JSON object containing these fields:
{{
  "primary_intent": "one of: depression_symptoms, anxiety_panic, bullying_harassment, academic_pressure, family_conflicts, social_anxiety, loneliness_isolation, crisis_situation, help_seeking, general_support",
  "emotional_state": "sad, anxious, angry, hopeless, frustrated, lonely, overwhelmed, or neutral",
  "severity_score": 0.65,
  "urgency_level": "low, medium, high, or crisis",
  "key_concerns": ["list", "of", "main", "concerns"],
  "needs_immediate_help": false
}}

Important:
- Only use "crisis_situation" for explicit suicidal ideation or immediate danger
- Help-seeking messages should NOT be classified as crisis
- Be conservative with high severity scores
- Return ONLY valid JSON, no other text

User message: {user_message}"""
            
            response = self.client.generate_response(prompt=analysis_prompt, max_tokens=200, temperature=0.3)
            
            if response:
                try:
                    # Try to extract JSON from response
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        analysis = json.loads(json_str)
                        
                        # Validate required fields
                        required_fields = ["primary_intent", "emotional_state", "severity_score", "urgency_level"]
                        if all(field in analysis for field in required_fields):
                            self.logger.info(f"API intent analysis successful: {analysis['primary_intent']}")
                            return analysis
                            
                except json.JSONDecodeError:
                    self.logger.warning("API returned invalid JSON for intent analysis")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in API intent analysis: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of API integration"""
        return {
            "api_available": self.is_available,
            "model": self.client.model,
            "base_url": self.client.base_url,
            "connection_test": self.client.test_connection() if self.is_available else None,
            "capabilities": {
                "response_generation": self.is_available,
                "intent_analysis": self.is_available,
                "conversation_context": self.is_available,
                "crisis_detection": self.is_available
            }
        }

# Global instance for easy import
api_llama3 = ApiMentalHealthClient()

# Legacy compatibility functions
def generate_response(prompt: str, system_prompt: str = "") -> Optional[str]:
    """Legacy function for backward compatibility"""
    return api_llama3.client.generate_response(prompt, system_prompt)

def is_llama_available() -> bool:
    """Check if API service is available"""
    return api_llama3.is_available

def test_llama_connection() -> Dict[str, Any]:
    """Test API connection"""
    return api_llama3.client.test_connection()

def get_llama_health() -> Dict[str, Any]:
    """Get comprehensive API health status"""
    return api_llama3.get_status()

# Maintain backward compatibility with original variable name
ollama_llama3 = api_llama3