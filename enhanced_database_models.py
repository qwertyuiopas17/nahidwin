"""
Enhanced Database Models - Complete database schema for mental health chatbot
Includes user management, conversation tracking, method effectiveness, and counselor integration
"""

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import uuid

db = SQLAlchemy()

class User(db.Model):
    """Enhanced user model with comprehensive tracking"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(15), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(100), nullable=True)
    password_hash = db.Column(db.String(256), nullable=False)
    college = db.Column(db.String(100), nullable=True)
    
    # Account management
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=False)
    
    # Mental health tracking
    total_conversations = db.Column(db.Integer, default=0)
    current_risk_level = db.Column(db.String(20), default='low')  # low, medium, high, crisis
    current_conversation_stage = db.Column(db.String(50), default='initial_contact')
    primary_concerns = db.Column(db.Text)  # JSON array of main mental health concerns
    
    # Method and intervention tracking
    methods_tried_count = db.Column(db.Integer, default=0)
    effective_methods_count = db.Column(db.Integer, default=0)
    current_method = db.Column(db.String(100))
    method_check_due = db.Column(db.DateTime)
    
    # Professional help tracking
    counselor_referral_status = db.Column(db.String(50), default='none')  # none, suggested, interested, booked, completed
    referral_date = db.Column(db.DateTime)
    last_crisis_date = db.Column(db.DateTime)
    crisis_count = db.Column(db.Integer, default=0)
    
    # Progress tracking
    first_severity_score = db.Column(db.Float)
    latest_severity_score = db.Column(db.Float)
    average_severity_score = db.Column(db.Float)
    improvement_trend = db.Column(db.String(20), default='stable')  # improving, stable, declining
    
    # Enhanced fields
    timezone = db.Column(db.String(50), default='UTC')
    preferred_language = db.Column(db.String(10), default='en')
    notification_preferences = db.Column(db.Text)  # JSON object for notification settings
    
    # Relationships
    conversation_turns = db.relationship('ConversationTurn', backref='user', lazy=True, cascade='all, delete-orphan')
    method_feedback = db.relationship('MethodFeedback', backref='user', lazy=True, cascade='all, delete-orphan')
    counselor_interactions = db.relationship('CounselorInteraction', backref='user', lazy=True, cascade='all, delete-orphan')
    crisis_events = db.relationship('CrisisEvent', backref='user', lazy=True, cascade='all, delete-orphan')
    user_sessions = db.relationship('UserSession', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_primary_concerns(self):
        """Get primary concerns as list"""
        if self.primary_concerns:
            try:
                return json.loads(self.primary_concerns)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_primary_concerns(self, concerns_list):
        """Set primary concerns from list"""
        self.primary_concerns = json.dumps(concerns_list)
    
    def get_notification_preferences(self):
        """Get notification preferences as dict"""
        if self.notification_preferences:
            try:
                return json.loads(self.notification_preferences)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_notification_preferences(self, prefs_dict):
        """Set notification preferences from dict"""
        self.notification_preferences = json.dumps(prefs_dict)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.now()
    
    def calculate_method_success_rate(self):
        """Calculate method success rate"""
        if self.methods_tried_count == 0:
            return 0.0
        return self.effective_methods_count / self.methods_tried_count
    
    def is_due_for_check_in(self):
        """Check if user is due for a method check-in"""
        if not self.method_check_due:
            return False
        return datetime.now() >= self.method_check_due
    
    def get_crisis_risk_level(self):
        """Get current crisis risk level based on recent activity"""
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_crises = CrisisEvent.query.filter(
            CrisisEvent.user_id == self.id,
            CrisisEvent.crisis_detected_date >= recent_cutoff
        ).count()
        
        if recent_crises > 2:
            return 'high'
        elif recent_crises > 0:
            return 'medium'
        return 'low'
    
    def __repr__(self):
        return f'<User {self.student_id}>'

class ConversationTurn(db.Model):
    """Enhanced conversation turn with detailed analysis and tracking"""
    __tablename__ = 'conversation_turns'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Conversation content
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now, index=True)
    
    # NLU Analysis results
    detected_intent = db.Column(db.String(50), index=True)  # primary intent detected
    intent_confidence = db.Column(db.Float, default=0.0)
    severity_score = db.Column(db.Float, default=0.0)
    emotional_state = db.Column(db.String(50))  # primary emotion detected
    emotional_intensity = db.Column(db.Float, default=0.0)
    
    # Crisis detection results
    is_crisis = db.Column(db.Boolean, default=False, index=True)
    crisis_level = db.Column(db.Float, default=0.0)
    crisis_risk_level = db.Column(db.String(20), default='low')  # low, medium, high, crisis
    
    # Response characteristics
    conversation_stage = db.Column(db.String(50))  # stage of conversation
    response_type = db.Column(db.String(50))  # type of response given
    provides_concrete_help = db.Column(db.Boolean, default=False)
    method_suggested = db.Column(db.String(100))  # method suggested in this turn
    is_follow_up = db.Column(db.Boolean, default=False)
    
    # Context and tracking
    urgency_level = db.Column(db.String(20), default='low')
    context_entities = db.Column(db.Text)  # JSON of extracted entities
    user_needs_identified = db.Column(db.Text)  # JSON of identified user needs
    
    # Performance metrics
    response_time_ms = db.Column(db.Integer)  # response generation time
    user_satisfaction_rating = db.Column(db.Integer)  # 1-5 if provided by user
    
    # Enhanced fields
    turn_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    language_detected = db.Column(db.String(10), default='en')
    sentiment_score = db.Column(db.Float, default=0.0)  # -1 to 1
    topic_tags = db.Column(db.Text)  # JSON array of topic tags
    
    def get_context_entities(self):
        """Get context entities as dict"""
        if self.context_entities:
            try:
                return json.loads(self.context_entities)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_context_entities(self, entities_dict):
        """Set context entities from dict"""
        self.context_entities = json.dumps(entities_dict)
    
    def get_user_needs(self):
        """Get user needs as list"""
        if self.user_needs_identified:
            try:
                return json.loads(self.user_needs_identified)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_user_needs(self, needs_list):
        """Set user needs from list"""
        self.user_needs_identified = json.dumps(needs_list)
    
    def get_topic_tags(self):
        """Get topic tags as list"""
        if self.topic_tags:
            try:
                return json.loads(self.topic_tags)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_topic_tags(self, tags_list):
        """Set topic tags from list"""
        self.topic_tags = json.dumps(tags_list)
    
    def get_conversation_summary(self):
        """Get a brief summary of the conversation turn"""
        user_preview = self.user_message[:100] + "..." if len(self.user_message) > 100 else self.user_message
        return {
            'timestamp': self.timestamp.isoformat(),
            'user_message_preview': user_preview,
            'detected_intent': self.detected_intent,
            'severity_score': self.severity_score,
            'response_provided': bool(self.bot_response)
        }
    
    def __repr__(self):
        return f'<ConversationTurn {self.id}: {self.detected_intent}>'

class MethodFeedback(db.Model):
    """Track user feedback on therapeutic methods and techniques"""
    __tablename__ = 'method_feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Method information
    method_id = db.Column(db.String(100), nullable=False, index=True)
    method_name = db.Column(db.String(200), nullable=False)
    method_category = db.Column(db.String(50))  # anxiety_management, depression_support, etc.
    
    # Suggestion and usage tracking
    first_suggested_date = db.Column(db.DateTime, default=datetime.now)
    last_used_date = db.Column(db.DateTime)
    times_suggested = db.Column(db.Integer, default=1)
    times_used = db.Column(db.Integer, default=0)
    
    # Effectiveness tracking
    effectiveness_rating = db.Column(db.String(20), default='unknown')  # effective, ineffective, partially_effective, unknown
    effectiveness_score = db.Column(db.Float, default=0.5)  # 0.0 to 1.0
    user_feedback_text = db.Column(db.Text)
    feedback_date = db.Column(db.DateTime)
    
    # Usage context
    severity_when_suggested = db.Column(db.Float, default=0.5)
    intent_when_suggested = db.Column(db.String(50))
    
    # Outcome tracking
    improvement_noted = db.Column(db.Boolean, default=False)
    side_effects_reported = db.Column(db.Boolean, default=False)
    would_recommend = db.Column(db.Boolean)
    
    # Enhanced tracking
    usage_frequency = db.Column(db.String(20), default='unknown')  # daily, weekly, monthly, rarely
    difficulty_level = db.Column(db.Integer, default=3)  # 1-5 scale
    time_to_effect = db.Column(db.Integer)  # minutes until user felt effect
    
    def mark_as_used(self):
        """Mark method as used and update timestamp"""
        self.times_used += 1
        self.last_used_date = datetime.now()
    
    def update_effectiveness(self, rating, score, feedback_text=None):
        """Update effectiveness based on user feedback"""
        self.effectiveness_rating = rating
        self.effectiveness_score = score
        if feedback_text:
            self.user_feedback_text = feedback_text
        self.feedback_date = datetime.now()
        
        # Set improvement flag based on rating
        self.improvement_noted = rating in ['effective', 'partially_effective']
    
    def get_usage_pattern(self):
        """Analyze usage pattern"""
        if not self.last_used_date or not self.first_suggested_date:
            return 'no_usage'
        
        days_since_suggestion = (datetime.now() - self.first_suggested_date).days
        if days_since_suggestion == 0:
            days_since_suggestion = 1
        
        usage_rate = self.times_used / days_since_suggestion
        
        if usage_rate >= 1:
            return 'daily'
        elif usage_rate >= 0.5:
            return 'frequent'
        elif usage_rate >= 0.1:
            return 'occasional'
        else:
            return 'rare'
    
    def __repr__(self):
        return f'<MethodFeedback {self.method_id}: {self.effectiveness_rating}>'

class CounselorInteraction(db.Model):
    """Track counselor referrals and professional help interactions"""
    __tablename__ = 'counselor_interactions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Counselor information
    counselor_id = db.Column(db.String(50))
    counselor_name = db.Column(db.String(100))
    counselor_specialties = db.Column(db.Text)  # JSON array
    counselor_contact_info = db.Column(db.Text)  # JSON object
    
    # Referral tracking
    referral_date = db.Column(db.DateTime, default=datetime.now)
    referral_reason = db.Column(db.Text)  # Why was referral made
    referral_urgency = db.Column(db.String(20), default='standard')  # standard, urgent, emergency
    user_concerns_matched = db.Column(db.Text)  # JSON array of matched concerns
    
    # Status tracking
    referral_status = db.Column(db.String(50), default='suggested')  # suggested, contacted, booked, attended, completed, declined
    status_updated_date = db.Column(db.DateTime, default=datetime.now)
    
    # Appointment details
    appointment_date = db.Column(db.DateTime)
    appointment_type = db.Column(db.String(50))  # in_person, video_call, phone_call
    booking_reference = db.Column(db.String(100))
    
    # Outcome tracking
    appointment_attended = db.Column(db.Boolean)
    user_satisfaction = db.Column(db.Integer)  # 1-5 rating
    follow_up_recommended = db.Column(db.Boolean)
    outcome_notes = db.Column(db.Text)
    
    # Enhanced tracking
    referral_source = db.Column(db.String(50), default='chatbot')  # chatbot, user_request, crisis
    counselor_response_time = db.Column(db.Integer)  # hours to respond
    cost_information = db.Column(db.Text)  # JSON object with cost details
    
    def get_specialties(self):
        """Get counselor specialties as list"""
        if self.counselor_specialties:
            try:
                return json.loads(self.counselor_specialties)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_specialties(self, specialties_list):
        """Set counselor specialties from list"""
        self.counselor_specialties = json.dumps(specialties_list)
    
    def get_matched_concerns(self):
        """Get matched concerns as list"""
        if self.user_concerns_matched:
            try:
                return json.loads(self.user_concerns_matched)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_matched_concerns(self, concerns_list):
        """Set matched concerns from list"""
        self.user_concerns_matched = json.dumps(concerns_list)
    
    def get_contact_info(self):
        """Get contact info as dict"""
        if self.counselor_contact_info:
            try:
                return json.loads(self.counselor_contact_info)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_contact_info(self, contact_dict):
        """Set contact info from dict"""
        self.counselor_contact_info = json.dumps(contact_dict)
    
    def get_cost_info(self):
        """Get cost information as dict"""
        if self.cost_information:
            try:
                return json.loads(self.cost_information)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_cost_info(self, cost_dict):
        """Set cost information from dict"""
        self.cost_information = json.dumps(cost_dict)
    
    def update_status(self, new_status):
        """Update referral status with timestamp"""
        self.referral_status = new_status
        self.status_updated_date = datetime.now()
    
    def days_since_referral(self):
        """Calculate days since referral was made"""
        return (datetime.now() - self.referral_date).days
    
    def is_overdue(self):
        """Check if referral follow-up is overdue"""
        if self.referral_status in ['completed', 'declined']:
            return False
        
        days_since = self.days_since_referral()
        
        if self.referral_urgency == 'emergency':
            return days_since > 1
        elif self.referral_urgency == 'urgent':
            return days_since > 3
        else:
            return days_since > 7
    
    def __repr__(self):
        return f'<CounselorInteraction {self.counselor_name}: {self.referral_status}>'

class CrisisEvent(db.Model):
    """Track crisis situations and interventions"""
    __tablename__ = 'crisis_events'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Crisis details
    crisis_detected_date = db.Column(db.DateTime, default=datetime.now, index=True)
    crisis_level = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    risk_assessment = db.Column(db.String(20), nullable=False)  # low, medium, high, critical
    
    # Crisis content
    user_message = db.Column(db.Text, nullable=False)
    detected_indicators = db.Column(db.Text)  # JSON array of crisis indicators found
    crisis_type = db.Column(db.String(50))  # suicide, self_harm, severe_depression
    
    # Intervention provided
    intervention_type = db.Column(db.String(50), nullable=False)  # crisis_response, emergency_referral, safety_planning
    bot_response = db.Column(db.Text, nullable=False)
    resources_provided = db.Column(db.Text)  # JSON array of resources/hotlines provided
    
    # Follow-up tracking
    follow_up_needed = db.Column(db.Boolean, default=True)
    follow_up_date = db.Column(db.DateTime)
    follow_up_completed = db.Column(db.Boolean, default=False)
    crisis_resolved = db.Column(db.Boolean, default=False)
    
    # Professional intervention
    emergency_services_contacted = db.Column(db.Boolean, default=False)
    professional_help_sought = db.Column(db.Boolean, default=False)
    hospitalization_required = db.Column(db.Boolean, default=False)
    
    # Outcome tracking
    user_safe_confirmed = db.Column(db.Boolean)
    crisis_duration_hours = db.Column(db.Float)
    intervention_effectiveness = db.Column(db.String(20))  # effective, partially_effective, ineffective
    
    # Enhanced tracking
    crisis_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    context_before_crisis = db.Column(db.Text)  # JSON of conversation context
    immediate_triggers = db.Column(db.Text)  # JSON array of identified triggers
    support_network_contacted = db.Column(db.Boolean, default=False)
    
    def get_detected_indicators(self):
        """Get detected crisis indicators as list"""
        if self.detected_indicators:
            try:
                return json.loads(self.detected_indicators)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_detected_indicators(self, indicators_list):
        """Set detected indicators from list"""
        self.detected_indicators = json.dumps(indicators_list)
    
    def get_resources_provided(self):
        """Get provided resources as list"""
        if self.resources_provided:
            try:
                return json.loads(self.resources_provided)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_resources_provided(self, resources_list):
        """Set provided resources from list"""
        self.resources_provided = json.dumps(resources_list)
    
    def get_context_before_crisis(self):
        """Get crisis context as dict"""
        if self.context_before_crisis:
            try:
                return json.loads(self.context_before_crisis)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_context_before_crisis(self, context_dict):
        """Set crisis context from dict"""
        self.context_before_crisis = json.dumps(context_dict)
    
    def get_immediate_triggers(self):
        """Get immediate triggers as list"""
        if self.immediate_triggers:
            try:
                return json.loads(self.immediate_triggers)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_immediate_triggers(self, triggers_list):
        """Set immediate triggers from list"""
        self.immediate_triggers = json.dumps(triggers_list)
    
    def mark_resolved(self):
        """Mark crisis as resolved"""
        self.crisis_resolved = True
        self.follow_up_completed = True
        if self.crisis_detected_date:
            duration = datetime.now() - self.crisis_detected_date
            self.crisis_duration_hours = duration.total_seconds() / 3600
    
    def is_recent(self, hours=24):
        """Check if crisis occurred within specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return self.crisis_detected_date >= cutoff
    
    def requires_immediate_follow_up(self):
        """Check if crisis requires immediate follow-up"""
        if self.crisis_resolved:
            return False
        
        if self.risk_assessment in ['critical', 'high']:
            hours_since = (datetime.now() - self.crisis_detected_date).total_seconds() / 3600
            return hours_since >= 2  # Follow up after 2 hours for high-risk cases
        
        return False
    
    def __repr__(self):
        return f'<CrisisEvent {self.crisis_id}: {self.risk_assessment}>'

class SystemMetrics(db.Model):
    """Track system-wide performance and effectiveness metrics"""
    __tablename__ = 'system_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Time period
    metrics_date = db.Column(db.Date, default=datetime.now().date, index=True)
    period_type = db.Column(db.String(20), default='daily')  # daily, weekly, monthly
    
    # Usage metrics
    total_active_users = db.Column(db.Integer, default=0)
    new_users_registered = db.Column(db.Integer, default=0)
    total_conversations = db.Column(db.Integer, default=0)
    average_conversations_per_user = db.Column(db.Float, default=0.0)
    
    # Response quality metrics
    average_intent_confidence = db.Column(db.Float, default=0.0)
    average_response_time_ms = db.Column(db.Float, default=0.0)
    user_satisfaction_average = db.Column(db.Float, default=0.0)
    
    # Crisis intervention metrics
    crisis_events_detected = db.Column(db.Integer, default=0)
    false_positive_crisis_rate = db.Column(db.Float, default=0.0)
    crisis_interventions_successful = db.Column(db.Integer, default=0)
    emergency_referrals_made = db.Column(db.Integer, default=0)
    
    # Method effectiveness metrics
    methods_suggested_total = db.Column(db.Integer, default=0)
    methods_marked_effective = db.Column(db.Integer, default=0)
    overall_method_success_rate = db.Column(db.Float, default=0.0)
    
    # Professional help metrics
    counselor_referrals_made = db.Column(db.Integer, default=0)
    referrals_resulting_in_booking = db.Column(db.Integer, default=0)
    referral_conversion_rate = db.Column(db.Float, default=0.0)
    
    # User progression metrics
    users_showing_improvement = db.Column(db.Integer, default=0)
    users_with_declining_trend = db.Column(db.Integer, default=0)
    average_user_improvement_score = db.Column(db.Float, default=0.0)
    
    # Enhanced metrics
    system_uptime_hours = db.Column(db.Float, default=24.0)
    error_rate_percentage = db.Column(db.Float, default=0.0)
    peak_concurrent_users = db.Column(db.Integer, default=0)
    
    def calculate_conversion_rate(self):
        """Calculate referral conversion rate"""
        if self.counselor_referrals_made == 0:
            return 0.0
        return self.referrals_resulting_in_booking / self.counselor_referrals_made
    
    def calculate_method_success_rate(self):
        """Calculate overall method success rate"""
        if self.methods_suggested_total == 0:
            return 0.0
        return self.methods_marked_effective / self.methods_suggested_total
    
    def update_averages(self):
        """Update calculated averages"""
        self.referral_conversion_rate = self.calculate_conversion_rate()
        self.overall_method_success_rate = self.calculate_method_success_rate()
        
        if self.total_active_users > 0:
            self.average_conversations_per_user = self.total_conversations / self.total_active_users
    
    def __repr__(self):
        return f'<SystemMetrics {self.metrics_date}>'

class UserSession(db.Model):
    """Track user sessions for analytics and security"""
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Session tracking
    session_start = db.Column(db.DateTime, default=datetime.now)
    session_end = db.Column(db.DateTime)
    session_duration_minutes = db.Column(db.Float)
    
    # Session details
    ip_address = db.Column(db.String(45))  # IPv6 compatible
    user_agent = db.Column(db.Text)
    device_type = db.Column(db.String(50))  # mobile, desktop, tablet
    
    # Conversation metrics for this session
    conversations_in_session = db.Column(db.Integer, default=0)
    methods_suggested_in_session = db.Column(db.Integer, default=0)
    crisis_events_in_session = db.Column(db.Integer, default=0)
    
    # Enhanced session tracking
    session_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    login_method = db.Column(db.String(20), default='password')  # password, oauth, etc.
    last_activity = db.Column(db.DateTime, default=datetime.now)
    is_active = db.Column(db.Boolean, default=True)
    
    def end_session(self):
        """End the session and calculate duration"""
        self.session_end = datetime.now()
        self.is_active = False
        
        if self.session_start:
            duration = self.session_end - self.session_start
            self.session_duration_minutes = duration.total_seconds() / 60
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_expired(self, hours=24):
        """Check if session is expired"""
        if not self.last_activity:
            return True
        
        expiry_time = self.last_activity + timedelta(hours=hours)
        return datetime.now() > expiry_time
    
    def get_session_summary(self):
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'duration_minutes': self.session_duration_minutes,
            'conversations': self.conversations_in_session,
            'methods_suggested': self.methods_suggested_in_session,
            'crisis_events': self.crisis_events_in_session,
            'device_type': self.device_type
        }
    
    def __repr__(self):
        return f'<UserSession {self.session_id}>'

# Additional utility models
class SystemConfiguration(db.Model):
    """Store system configuration settings"""
    __tablename__ = 'system_configuration'
    
    id = db.Column(db.Integer, primary_key=True)
    config_key = db.Column(db.String(100), unique=True, nullable=False)
    config_value = db.Column(db.Text)
    config_type = db.Column(db.String(20), default='string')  # string, json, boolean, integer, float
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    @classmethod
    def get_config(cls, key, default=None):
        """Get configuration value"""
        config = cls.query.filter_by(config_key=key).first()
        if not config:
            return default
        
        if config.config_type == 'json':
            try:
                return json.loads(config.config_value)
            except json.JSONDecodeError:
                return default
        elif config.config_type == 'boolean':
            return config.config_value.lower() in ['true', '1', 'yes']
        elif config.config_type == 'integer':
            try:
                return int(config.config_value)
            except ValueError:
                return default
        elif config.config_type == 'float':
            try:
                return float(config.config_value)
            except ValueError:
                return default
        
        return config.config_value
    
    @classmethod
    def set_config(cls, key, value, config_type='string', description=None):
        """Set configuration value"""
        config = cls.query.filter_by(config_key=key).first()
        
        if config_type == 'json' and not isinstance(value, str):
            value = json.dumps(value)
        elif config_type in ['boolean', 'integer', 'float']:
            value = str(value)
        
        if config:
            config.config_value = value
            config.config_type = config_type
            config.updated_at = datetime.now()
            if description:
                config.description = description
        else:
            config = cls(
                config_key=key,
                config_value=value,
                config_type=config_type,
                description=description
            )
        
        db.session.add(config)
        db.session.commit()
        return config

# Database initialization and utility functions
def init_database(app):
    """Initialize database with app context"""
    with app.app_context():
        db.create_all()
        
        # Create default system configurations
        default_configs = [
            ('crisis_hotlines', json.dumps([
                {'name': 'Emergency Services', 'number': '112', 'description': 'Immediate emergency response'},
                {'name': 'Crisis Text Line', 'number': 'Text HOME to 741741', 'description': '24/7 crisis support via text'},
                {'name': 'National Suicide Prevention', 'number': '1860-266-2345', 'description': 'Suicide prevention hotline'}
            ]), 'json', 'Emergency crisis hotlines'),
            ('max_session_duration_hours', '24', 'integer', 'Maximum session duration in hours'),
            ('enable_crisis_detection', 'true', 'boolean', 'Enable automatic crisis detection'),
            ('method_check_interval_days', '3', 'integer', 'Days between method effectiveness check-ins'),
            ('system_maintenance_mode', 'false', 'boolean', 'System maintenance mode flag')
        ]
        
        for key, value, config_type, description in default_configs:
            if not SystemConfiguration.query.filter_by(config_key=key).first():
                SystemConfiguration.set_config(key, value, config_type, description)
        
        print("✅ Database tables created successfully")
        print("✅ Default system configurations initialized")

def get_user_statistics(user_id: int) -> dict:
    """Get comprehensive user statistics"""
    user = User.query.get(user_id)
    if not user:
        return {}
    
    # Get conversation statistics
    total_turns = ConversationTurn.query.filter_by(user_id=user_id).count()
    crisis_events = CrisisEvent.query.filter_by(user_id=user_id).count()
    
    # Get method effectiveness
    effective_methods = MethodFeedback.query.filter_by(
        user_id=user_id,
        effectiveness_rating='effective'
    ).count()
    
    total_methods = MethodFeedback.query.filter_by(user_id=user_id).count()
    
    # Get recent severity trend
    recent_turns = ConversationTurn.query.filter_by(user_id=user_id)\
        .order_by(ConversationTurn.timestamp.desc())\
        .limit(10).all()
    
    severity_trend = [turn.severity_score for turn in reversed(recent_turns)]
    
    # Get session statistics
    total_sessions = UserSession.query.filter_by(user_id=user_id).count()
    avg_session_duration = db.session.query(db.func.avg(UserSession.session_duration_minutes))\
        .filter_by(user_id=user_id).scalar() or 0
    
    return {
        'user_info': {
            'student_id': user.student_id,
            'total_conversations': total_turns,
            'member_since': user.created_at,
            'last_active': user.last_login,
            'total_sessions': total_sessions,
            'avg_session_duration': round(avg_session_duration, 2)
        },
        'mental_health': {
            'current_risk_level': user.current_risk_level,
            'primary_concerns': user.get_primary_concerns(),
            'severity_trend': severity_trend,
            'improvement_trend': user.improvement_trend,
            'crisis_risk_level': user.get_crisis_risk_level()
        },
        'interventions': {
            'crisis_events': crisis_events,
            'methods_tried': total_methods,
            'effective_methods': effective_methods,
            'method_success_rate': user.calculate_method_success_rate()
        },
        'professional_help': {
            'referral_status': user.counselor_referral_status,
            'referral_date': user.referral_date
        }
    }

def cleanup_expired_sessions():
    """Clean up expired user sessions"""
    expired_cutoff = datetime.now() - timedelta(hours=24)
    
    expired_sessions = UserSession.query.filter(
        UserSession.last_activity < expired_cutoff,
        UserSession.is_active == True
    ).all()
    
    for session in expired_sessions:
        session.end_session()
    
    db.session.commit()
    return len(expired_sessions)

def get_system_health():
    """Get overall system health metrics"""
    try:
        # Basic counts
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        total_conversations = ConversationTurn.query.count()
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_conversations = ConversationTurn.query.filter(
            ConversationTurn.timestamp >= recent_cutoff
        ).count()
        
        recent_crises = CrisisEvent.query.filter(
            CrisisEvent.crisis_detected_date >= recent_cutoff
        ).count()
        
        # Active sessions
        active_sessions = UserSession.query.filter_by(is_active=True).count()
        
        return {
            'database_healthy': True,
            'total_users': total_users,
            'active_users': active_users,
            'total_conversations': total_conversations,
            'recent_conversations': recent_conversations,
            'recent_crises': recent_crises,
            'active_sessions': active_sessions,
            'last_checked': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'database_healthy': False,
            'error': str(e),
            'last_checked': datetime.now().isoformat()
        }