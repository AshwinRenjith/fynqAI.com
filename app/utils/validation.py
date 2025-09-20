"""
Validation Utilities
Common validation functions and decorators
"""

import re
import uuid
from typing import Any, Dict, Optional
from datetime import datetime
from email_validator import validate_email, EmailNotValidError
import phonenumbers
from phonenumbers import NumberParseException

from app.exceptions import ValidationError


class Validator:
    """Collection of validation utilities"""
    
    @staticmethod
    def validate_email_address(email: str) -> str:
        """
        Validate and normalize email address
        
        Args:
            email: Email address to validate
            
        Returns:
            Normalized email address
            
        Raises:
            ValidationError: If email is invalid
        """
        try:
            validated_email = validate_email(email)
            return validated_email.email.lower()
        except EmailNotValidError as e:
            raise ValidationError(f"Invalid email address: {str(e)}")
    
    @staticmethod
    def validate_phone_number(phone: str, region: Optional[str] = None) -> str:
        """
        Validate and format phone number
        
        Args:
            phone: Phone number to validate
            region: Optional region code (e.g., 'IN', 'US')
            
        Returns:
            Formatted phone number in E.164 format
            
        Raises:
            ValidationError: If phone number is invalid
        """
        try:
            parsed_number = phonenumbers.parse(phone, region)
            
            if not phonenumbers.is_valid_number(parsed_number):
                raise ValidationError("Invalid phone number")
            
            return phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
            
        except NumberParseException as e:
            raise ValidationError(f"Invalid phone number format: {str(e)}")
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """
        Validate password strength
        
        Args:
            password: Password to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "is_valid": True,
            "score": 0,
            "errors": [],
            "suggestions": []
        }
        
        # Length check
        if len(password) < 8:
            result["errors"].append("Password must be at least 8 characters long")
            result["is_valid"] = False
        else:
            result["score"] += 1
        
        # Uppercase check
        if not re.search(r"[A-Z]", password):
            result["suggestions"].append("Add uppercase letters")
        else:
            result["score"] += 1
        
        # Lowercase check
        if not re.search(r"[a-z]", password):
            result["suggestions"].append("Add lowercase letters")
        else:
            result["score"] += 1
        
        # Number check
        if not re.search(r"\d", password):
            result["suggestions"].append("Add numbers")
        else:
            result["score"] += 1
        
        # Special character check
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            result["suggestions"].append("Add special characters")
        else:
            result["score"] += 1
        
        # Common password check
        common_passwords = [
            "password", "123456", "qwerty", "abc123", "password123",
            "admin", "letmein", "welcome", "monkey", "dragon"
        ]
        
        if password.lower() in common_passwords:
            result["errors"].append("Password is too common")
            result["is_valid"] = False
        
        # Calculate strength level
        if result["score"] >= 4:
            result["strength"] = "strong"
        elif result["score"] >= 3:
            result["strength"] = "medium"
        else:
            result["strength"] = "weak"
        
        return result
    
    @staticmethod
    def validate_uuid(value: str) -> uuid.UUID:
        """
        Validate UUID string
        
        Args:
            value: UUID string to validate
            
        Returns:
            UUID object
            
        Raises:
            ValidationError: If UUID is invalid
        """
        try:
            return uuid.UUID(value)
        except (ValueError, TypeError):
            raise ValidationError("Invalid UUID format")
    
    @staticmethod
    def validate_academic_year(year: str) -> bool:
        """
        Validate academic year format (e.g., "2024-25")
        
        Args:
            year: Academic year string
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If format is invalid
        """
        pattern = r"^\d{4}-\d{2}$"
        
        if not re.match(pattern, year):
            raise ValidationError("Academic year must be in format YYYY-YY (e.g., 2024-25)")
        
        # Parse years
        start_year, end_year_suffix = year.split("-")
        start_year = int(start_year)
        end_year = int(f"{str(start_year)[:2]}{end_year_suffix}")
        
        # Validate year progression
        if end_year != start_year + 1:
            raise ValidationError("Academic year end must be consecutive to start year")
        
        # Validate reasonable range
        current_year = datetime.now().year
        if start_year < current_year - 5 or start_year > current_year + 2:
            raise ValidationError("Academic year is outside reasonable range")
        
        return True
    
    @staticmethod
    def validate_grade(grade: int) -> bool:
        """
        Validate grade level
        
        Args:
            grade: Grade level
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If grade is invalid
        """
        if grade not in [11, 12]:
            raise ValidationError("Grade must be 11 or 12")
        
        return True
    
    @staticmethod
    def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate JSON data against a simple schema
        
        Args:
            data: Data to validate
            schema: Schema definition
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        def validate_field(value, field_schema, field_name):
            field_type = field_schema.get("type")
            required = field_schema.get("required", False)
            
            if value is None:
                if required:
                    raise ValidationError(f"Field '{field_name}' is required")
                return
            
            if field_type == "string" and not isinstance(value, str):
                raise ValidationError(f"Field '{field_name}' must be a string")
            elif field_type == "number" and not isinstance(value, (int, float)):
                raise ValidationError(f"Field '{field_name}' must be a number")
            elif field_type == "boolean" and not isinstance(value, bool):
                raise ValidationError(f"Field '{field_name}' must be a boolean")
            elif field_type == "array" and not isinstance(value, list):
                raise ValidationError(f"Field '{field_name}' must be an array")
            elif field_type == "object" and not isinstance(value, dict):
                raise ValidationError(f"Field '{field_name}' must be an object")
            
            # Validate string constraints
            if field_type == "string" and isinstance(value, str):
                min_length = field_schema.get("min_length")
                max_length = field_schema.get("max_length")
                pattern = field_schema.get("pattern")
                
                if min_length and len(value) < min_length:
                    raise ValidationError(f"Field '{field_name}' must be at least {min_length} characters")
                
                if max_length and len(value) > max_length:
                    raise ValidationError(f"Field '{field_name}' must be at most {max_length} characters")
                
                if pattern and not re.match(pattern, value):
                    raise ValidationError(f"Field '{field_name}' does not match required pattern")
            
            # Validate number constraints
            if field_type == "number" and isinstance(value, (int, float)):
                minimum = field_schema.get("minimum")
                maximum = field_schema.get("maximum")
                
                if minimum is not None and value < minimum:
                    raise ValidationError(f"Field '{field_name}' must be at least {minimum}")
                
                if maximum is not None and value > maximum:
                    raise ValidationError(f"Field '{field_name}' must be at most {maximum}")
        
        # Validate each field in schema
        for field_name, field_schema in schema.items():
            value = data.get(field_name)
            validate_field(value, field_schema, field_name)
        
        return True
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """
        Basic HTML sanitization
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        clean_text = re.sub(r"<[^>]+>", "", text)
        
        # Remove script content
        clean_text = re.sub(r"<script.*?</script>", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove style content
        clean_text = re.sub(r"<style.*?</style>", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Decode HTML entities
        import html
        clean_text = html.unescape(clean_text)
        
        return clean_text.strip()
    
    @staticmethod
    def validate_image_url(url: str) -> bool:
        """
        Validate image URL format
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If URL is invalid
        """
        if not url:
            raise ValidationError("Image URL cannot be empty")
        
        # Basic URL format check
        url_pattern = r"^https?:\/\/[^\s\/$.?#].[^\s]*$"
        if not re.match(url_pattern, url):
            raise ValidationError("Invalid URL format")
        
        # Check image extension
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
        if not any(url.lower().endswith(ext) for ext in image_extensions):
            raise ValidationError("URL must point to a valid image file")
        
        return True
    
    @staticmethod
    def validate_subject_code(code: str) -> bool:
        """
        Validate subject code format
        
        Args:
            code: Subject code to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If code is invalid
        """
        valid_codes = ["MATH", "PHY", "CHEM", "BIO", "ENG", "COMP"]
        
        if code.upper() not in valid_codes:
            raise ValidationError(f"Invalid subject code. Must be one of: {', '.join(valid_codes)}")
        
        return True
    
    @staticmethod
    def validate_exam_type(exam: str) -> bool:
        """
        Validate exam type
        
        Args:
            exam: Exam type to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If exam type is invalid
        """
        valid_exams = ["JEE_MAIN", "JEE_ADVANCED", "NEET", "BITSAT", "BOARD_EXAM"]
        
        if exam not in valid_exams:
            raise ValidationError(f"Invalid exam type. Must be one of: {', '.join(valid_exams)}")
        
        return True


class DataSanitizer:
    """Data sanitization utilities"""
    
    @staticmethod
    def sanitize_user_input(text: str, max_length: int = 1000) -> str:
        """
        Sanitize user input text
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove HTML
        sanitized = Validator.sanitize_html(text)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Remove excessive whitespace
        sanitized = re.sub(r"\s+", " ", sanitized)
        
        return sanitized
    
    @staticmethod
    def sanitize_file_name(filename: str) -> str:
        """
        Sanitize file name for safe storage
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        if not filename:
            return "unnamed_file"
        
        # Remove path separators
        sanitized = filename.replace("/", "_").replace("\\", "_")
        
        # Remove or replace special characters
        sanitized = re.sub(r"[<>:\"|?*]", "", sanitized)
        
        # Replace spaces with underscores
        sanitized = sanitized.replace(" ", "_")
        
        # Remove multiple underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
            max_name_length = 250 - len(ext)
            sanitized = f"{name[:max_name_length]}.{ext}" if ext else name[:255]
        
        return sanitized.strip("_")
    
    @staticmethod
    def sanitize_search_query(query: str) -> str:
        """
        Sanitize search query for safe processing
        
        Args:
            query: Search query
            
        Returns:
            Sanitized query
        """
        if not query:
            return ""
        
        # Remove HTML
        sanitized = Validator.sanitize_html(query)
        
        # Remove special regex characters
        sanitized = re.sub(r"[.*+?^${}()|[\]\\]", "", sanitized)
        
        # Trim and normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized.strip())
        
        # Limit length
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        
        return sanitized


# Validation decorators
def validate_required(*fields):
    """Decorator to validate required fields in request data"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Assume first argument after self is request data
            data = args[1] if len(args) > 1 else kwargs.get('data', {})
            
            if isinstance(data, dict):
                missing_fields = [field for field in fields if not data.get(field)]
                if missing_fields:
                    raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_uuid_param(param_name: str):
    """Decorator to validate UUID parameters"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if param_name in kwargs:
                try:
                    kwargs[param_name] = uuid.UUID(str(kwargs[param_name]))
                except (ValueError, TypeError):
                    raise ValidationError(f"Invalid UUID format for parameter: {param_name}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Export main classes and functions
__all__ = [
    "Validator",
    "DataSanitizer", 
    "validate_required",
    "validate_uuid_param"
]
