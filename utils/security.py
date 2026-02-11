"""
Security utilities for the I.T Task Manager
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from flask import current_app, request
import jwt
from functools import wraps

def hash_password(password):
    """
    Hash a password using SHA-256 with salt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    salt = current_app.config.get('SECRET_KEY', 'default-secret-key')
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

def verify_password(password, hashed):
    """
    Verify a password against its hash
    
    Args:
        password: Plain text password
        hashed: Hashed password
        
    Returns:
        Boolean indicating if password matches
    """
    return hash_password(password) == hashed

def generate_api_key(user_id):
    """
    Generate API key for a user
    
    Args:
        user_id: User ID
        
    Returns:
        API key string
    """
    timestamp = datetime.utcnow().isoformat()
    data = f"{user_id}:{timestamp}"
    
    secret = current_app.config.get('SECRET_KEY', 'default-secret-key')
    signature = hmac.new(
        secret.encode(),
        data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return f"rfss_{user_id}_{signature[:32]}"

def validate_api_key(api_key):
    """
    Validate API key
    
    Args:
        api_key: API key to validate
        
    Returns:
        Tuple (is_valid, user_id)
    """
    if not api_key or not api_key.startswith('rfss_'):
        return False, None
    
    try:
        parts = api_key.split('_')
        if len(parts) != 3:
            return False, None
        
        prefix, user_id, signature = parts
        
        # Reconstruct and verify signature
        timestamp = "2024-01-01T00:00:00"  # Placeholder timestamp
        data = f"{user_id}:{timestamp}"
        secret = current_app.config.get('SECRET_KEY', 'default-secret-key')
        expected_signature = hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()[:32]
        
        if hmac.compare_digest(signature, expected_signature):
            return True, user_id
        else:
            return False, None
    
    except Exception:
        return False, None

def generate_jwt_token(user_id, expires_in=3600):
    """
    Generate JWT token
    
    Args:
        user_id: User ID
        expires_in: Token expiration in seconds
        
    Returns:
        JWT token string
    """
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(seconds=expires_in),
        'iat': datetime.utcnow()
    }
    
    secret = current_app.config.get('SECRET_KEY', 'default-secret-key')
    return jwt.encode(payload, secret, algorithm='HS256')

def decode_jwt_token(token):
    """
    Decode and validate JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Tuple (is_valid, payload)
    """
    try:
        secret = current_app.config.get('SECRET_KEY', 'default-secret-key')
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        return True, payload
    except jwt.ExpiredSignatureError:
        return False, {'error': 'Token expired'}
    except jwt.InvalidTokenError:
        return False, {'error': 'Invalid token'}

def sanitize_input(input_string):
    """
    Sanitize user input to prevent XSS
    
    Args:
        input_string: User input string
        
    Returns:
        Sanitized string
    """
    if not input_string:
        return ""
    
    # Remove HTML tags
    import re
    cleaned = re.sub(r'<[^>]*>', '', input_string)
    
    # Escape special characters
    escapes = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '/': '&#x2F;'
    }
    
    for char, escape in escapes.items():
        cleaned = cleaned.replace(char, escape)
    
    return cleaned

def validate_csrf_token(token):
    """
    Validate CSRF token
    
    Args:
        token: CSRF token from form
        
    Returns:
        Boolean indicating if token is valid
    """
    session_token = request.cookies.get('csrf_token')
    
    if not session_token or not token:
        return False
    
    return hmac.compare_digest(session_token, token)

def generate_csrf_token():
    """
    Generate CSRF token
    
    Returns:
        CSRF token string
    """
    return secrets.token_hex(32)

def rate_limit_check(identifier, limit=10, window=60):
    """
    Check if request is within rate limit
    
    Args:
        identifier: User identifier (IP or user ID)
        limit: Maximum requests per window
        window: Time window in seconds
        
    Returns:
        Tuple (is_allowed, remaining, reset_time)
    """
    from werkzeug.contrib.cache import SimpleCache
    cache = SimpleCache()
    
    key = f"rate_limit:{identifier}"
    current = cache.get(key)
    
    now = datetime.utcnow()
    
    if current is None:
        # First request in window
        cache.set(key, 1, timeout=window)
        return True, limit - 1, now + timedelta(seconds=window)
    
    if current >= limit:
        # Rate limit exceeded
        return False, 0, cache.get(key + '_reset') or now + timedelta(seconds=window)
    
    # Increment counter
    cache.set(key, current + 1, timeout=window)
    return True, limit - current - 1, now + timedelta(seconds=window)

def require_api_key(f):
    """
    Decorator to require API key for endpoint
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return {'error': 'API key required'}, 401
        
        is_valid, user_id = validate_api_key(api_key)
        
        if not is_valid:
            return {'error': 'Invalid API key'}, 401
        
        # Add user_id to request context
        request.user_id = user_id
        
        return f(*args, **kwargs)
    
    return decorated_function

def require_role(required_role):
    """
    Decorator to require specific user role
    
    Args:
        required_role: Required role (Admin, Supervisor, etc.)
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask_login import current_user
            
            if not current_user.is_authenticated:
                return {'error': 'Authentication required'}, 401
            
            if current_user.role != required_role:
                return {'error': f'{required_role} role required'}, 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator

def encrypt_data(data, key=None):
    """
    Encrypt sensitive data
    
    Args:
        data: Data to encrypt
        key: Encryption key (uses app secret if not provided)
        
    Returns:
        Encrypted data
    """
    if key is None:
        key = current_app.config.get('SECRET_KEY', 'default-secret-key')
    
    # Simple XOR encryption for demonstration
    # In production, use proper encryption like AES
    key_bytes = key.encode()
    data_bytes = str(data).encode()
    
    encrypted = bytearray()
    for i, byte in enumerate(data_bytes):
        encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
    
    return encrypted.hex()

def decrypt_data(encrypted_hex, key=None):
    """
    Decrypt encrypted data
    
    Args:
        encrypted_hex: Encrypted data in hex
        key: Decryption key (uses app secret if not provided)
        
    Returns:
        Decrypted data
    """
    if key is None:
        key = current_app.config.get('SECRET_KEY', 'default-secret-key')
    
    try:
        encrypted = bytes.fromhex(encrypted_hex)
        key_bytes = key.encode()
        
        decrypted = bytearray()
        for i, byte in enumerate(encrypted):
            decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return decrypted.decode()
    
    except Exception:
        return None

def generate_secure_filename(filename):
    """
    Generate secure filename to prevent path traversal
    
    Args:
        filename: Original filename
        
    Returns:
        Secure filename
    """
    import os
    from werkzeug.utils import secure_filename
    
    secure = secure_filename(filename)
    
    # Add timestamp to prevent collisions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name, ext = os.path.splitext(secure)
    
    return f"{name}_{timestamp}{ext}"

def validate_session():
    """
    Validate user session
    
    Returns:
        Tuple (is_valid, user_id)
    """
    from flask_login import current_user
    
    if not current_user.is_authenticated:
        return False, None
    
    # Check session timeout
    session_timeout = current_app.config.get('PERMANENT_SESSION_LIFETIME', timedelta(hours=2))
    
    if 'login_time' in session:
        login_time = datetime.fromisoformat(session['login_time'])
        if datetime.utcnow() - login_time > session_timeout:
            return False, None
    
    return True, current_user.id

def audit_log(action, entity_type=None, entity_id=None, details=None):
    """
    Create audit log entry
    
    Args:
        action: Action performed
        entity_type: Type of entity affected
        entity_id: ID of entity affected
        details: Additional details
    """
    from models.database import create_audit_log
    from flask_login import current_user
    
    user_id = current_user.id if current_user.is_authenticated else 'system'
    
    # Get request info
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent')
    
    create_audit_log(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details or {},
        ip_address=ip_address,
        user_agent=user_agent
    )