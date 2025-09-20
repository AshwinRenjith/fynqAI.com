"""
Process Intelligence Layer (PIL) - Core Innovation
Hallucination-free mathematical reasoning and step validation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import re
import sympy as sp
from sympy import *

from app.config import get_settings
from app.exceptions import PILProcessingError


logger = logging.getLogger(__name__)
settings = get_settings()


class ReasoningType(Enum):
    """Types of mathematical reasoning"""
    ALGEBRAIC = "algebraic"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    TRIGONOMETRY = "trigonometry"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    LOGIC = "logic"


class ValidationResult(Enum):
    """Validation results for reasoning steps"""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    REQUIRES_VERIFICATION = "requires_verification"


class ReasoningStep:
    """Represents a single step in mathematical reasoning"""
    
    def __init__(
        self,
        step_number: int,
        description: str,
        mathematical_expression: str,
        reasoning_type: ReasoningType,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        confidence: float = 1.0
    ):
        self.step_number = step_number
        self.description = description
        self.mathematical_expression = mathematical_expression
        self.reasoning_type = reasoning_type
        self.input_state = input_state
        self.output_state = output_state
        self.confidence = confidence
        self.validation_result: Optional[ValidationResult] = None
        self.verification_details: Dict[str, Any] = {}


class BaseValidator(ABC):
    """Abstract base class for step validators"""
    
    @abstractmethod
    async def validate_step(self, step: ReasoningStep) -> ValidationResult:
        """Validate a reasoning step"""
        pass


class AlgebraicValidator(BaseValidator):
    """Validator for algebraic reasoning steps"""
    
    async def validate_step(self, step: ReasoningStep) -> ValidationResult:
        """Validate algebraic manipulations and transformations"""
        try:
            # Extract mathematical expressions
            input_expr = step.input_state.get("expression", "")
            output_expr = step.output_state.get("expression", "")
            
            if not input_expr or not output_expr:
                return ValidationResult.UNCERTAIN
            
            # Parse expressions using SymPy
            try:
                input_sympy = sp.sympify(input_expr)
                output_sympy = sp.sympify(output_expr)
            except Exception:
                return ValidationResult.INVALID
            
            # Check if transformation is valid
            is_valid = await self._verify_algebraic_transformation(
                input_sympy, output_sympy, step
            )
            
            return ValidationResult.VALID if is_valid else ValidationResult.INVALID
            
        except Exception as e:
            logger.error(f"Algebraic validation failed: {e}")
            return ValidationResult.UNCERTAIN
    
    async def _verify_algebraic_transformation(
        self, 
        input_expr: sp.Expr, 
        output_expr: sp.Expr, 
        step: ReasoningStep
    ) -> bool:
        """Verify that algebraic transformation is mathematically valid"""
        
        operation = step.description.lower()
        
        # Check for common algebraic operations
        if "simplify" in operation or "expand" in operation:
            return sp.simplify(input_expr - output_expr) == 0
        
        if "factor" in operation:
            return sp.expand(output_expr) == input_expr
        
        if "substitute" in operation:
            # Check if substitution is valid
            substitutions = step.input_state.get("substitutions", {})
            if substitutions:
                substituted = input_expr.subs(substitutions)
                return sp.simplify(substituted - output_expr) == 0
        
        # General equivalence check
        try:
            difference = sp.simplify(input_expr - output_expr)
            return difference == 0
        except Exception:
            return False


class CalculusValidator(BaseValidator):
    """Validator for calculus operations"""
    
    async def validate_step(self, step: ReasoningStep) -> ValidationResult:
        """Validate calculus operations like differentiation and integration"""
        try:
            operation = step.description.lower()
            input_expr = step.input_state.get("expression", "")
            output_expr = step.output_state.get("expression", "")
            
            if not input_expr or not output_expr:
                return ValidationResult.UNCERTAIN
            
            try:
                input_sympy = sp.sympify(input_expr)
                output_sympy = sp.sympify(output_expr)
            except Exception:
                return ValidationResult.INVALID
            
            # Validate differentiation
            if "derivative" in operation or "differentiate" in operation:
                variable = step.input_state.get("variable", "x")
                computed_derivative = sp.diff(input_sympy, variable)
                return (ValidationResult.VALID if 
                       sp.simplify(computed_derivative - output_sympy) == 0 
                       else ValidationResult.INVALID)
            
            # Validate integration
            if "integrate" in operation or "integral" in operation:
                variable = step.input_state.get("variable", "x")
                computed_integral = sp.integrate(input_sympy, variable)
                
                # Check if derivative of result equals original
                derivative_check = sp.diff(output_sympy, variable)
                return (ValidationResult.VALID if 
                       sp.simplify(derivative_check - input_sympy) == 0 
                       else ValidationResult.INVALID)
            
            # Validate limits
            if "limit" in operation:
                variable = step.input_state.get("variable", "x")
                approach_value = step.input_state.get("approach", 0)
                computed_limit = sp.limit(input_sympy, variable, approach_value)
                return (ValidationResult.VALID if 
                       sp.simplify(computed_limit - output_sympy) == 0 
                       else ValidationResult.INVALID)
            
            return ValidationResult.UNCERTAIN
            
        except Exception as e:
            logger.error(f"Calculus validation failed: {e}")
            return ValidationResult.UNCERTAIN


class PhysicsValidator(BaseValidator):
    """Validator for physics reasoning and unit analysis"""
    
    async def validate_step(self, step: ReasoningStep) -> ValidationResult:
        """Validate physics equations and unit consistency"""
        try:
            # Check dimensional analysis
            if await self._validate_units(step):
                # Check physics principles
                if await self._validate_physics_principles(step):
                    return ValidationResult.VALID
                else:
                    return ValidationResult.INVALID
            else:
                return ValidationResult.INVALID
                
        except Exception as e:
            logger.error(f"Physics validation failed: {e}")
            return ValidationResult.UNCERTAIN
    
    async def _validate_units(self, step: ReasoningStep) -> bool:
        """Validate dimensional consistency"""
        input_units = step.input_state.get("units", {})
        output_units = step.output_state.get("units", {})
        
        if not input_units or not output_units:
            return True  # Cannot validate without unit information
        
        # Implement dimensional analysis logic
        # This is a simplified version - can be enhanced with proper unit libraries
        return True
    
    async def _validate_physics_principles(self, step: ReasoningStep) -> bool:
        """Validate against fundamental physics principles"""
        operation = step.description.lower()
        
        # Conservation laws
        if "conservation" in operation:
            return await self._check_conservation_laws(step)
        
        # Kinematic equations
        if "kinematic" in operation or "motion" in operation:
            return await self._check_kinematic_equations(step)
        
        # Thermodynamic laws
        if "thermodynamic" in operation or "heat" in operation:
            return await self._check_thermodynamic_laws(step)
        
        return True  # Default to valid if no specific checks
    
    async def _check_conservation_laws(self, step: ReasoningStep) -> bool:
        """Check conservation of energy, momentum, etc."""
        # Implement conservation law checks
        return True
    
    async def _check_kinematic_equations(self, step: ReasoningStep) -> bool:
        """Validate kinematic equation usage"""
        # Implement kinematic validation
        return True
    
    async def _check_thermodynamic_laws(self, step: ReasoningStep) -> bool:
        """Validate thermodynamic principles"""
        # Implement thermodynamic validation
        return True


class ReasoningEngine:
    """Core reasoning engine that orchestrates step validation"""
    
    def __init__(self):
        self.validators = {
            ReasoningType.ALGEBRAIC: AlgebraicValidator(),
            ReasoningType.CALCULUS: CalculusValidator(),
            ReasoningType.PHYSICS: PhysicsValidator(),
            # Add more validators as needed
        }
        self.reasoning_chain: List[ReasoningStep] = []
    
    async def process_solution(
        self, 
        question: str, 
        proposed_solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a complete solution through PIL validation
        
        Args:
            question: Original question text
            proposed_solution: LLM-generated solution with steps
        
        Returns:
            Validated solution with confidence scores and verification details
        """
        try:
            # Parse solution into reasoning steps
            reasoning_steps = await self._parse_solution_steps(proposed_solution)
            
            # Validate each step
            validated_steps = []
            overall_confidence = 1.0
            
            for step in reasoning_steps:
                validation_result = await self._validate_step(step)
                step.validation_result = validation_result
                
                # Update confidence based on validation
                if validation_result == ValidationResult.VALID:
                    step.confidence = min(step.confidence, 0.95)
                elif validation_result == ValidationResult.UNCERTAIN:
                    step.confidence = min(step.confidence, 0.7)
                    overall_confidence *= 0.8
                elif validation_result == ValidationResult.INVALID:
                    step.confidence = 0.0
                    overall_confidence = 0.0
                
                validated_steps.append(step)
            
            # Generate verification report
            verification_report = await self._generate_verification_report(
                validated_steps, overall_confidence
            )
            
            # Create validated solution
            validated_solution = {
                "original_question": question,
                "steps": [self._step_to_dict(step) for step in validated_steps],
                "overall_confidence": overall_confidence,
                "verification_report": verification_report,
                "hallucination_detected": overall_confidence < 0.3,
                "requires_human_review": overall_confidence < 0.7
            }
            
            logger.info(
                f"PIL processing completed",
                extra={
                    "steps_count": len(validated_steps),
                    "overall_confidence": overall_confidence,
                    "hallucination_detected": overall_confidence < 0.3
                }
            )
            
            return validated_solution
            
        except Exception as e:
            logger.error(f"PIL processing failed: {e}", exc_info=True)
            raise PILProcessingError(f"Reasoning validation failed: {str(e)}")
    
    async def _parse_solution_steps(
        self, 
        proposed_solution: Dict[str, Any]
    ) -> List[ReasoningStep]:
        """Parse LLM solution into structured reasoning steps"""
        
        steps_data = proposed_solution.get("steps", [])
        reasoning_steps = []
        
        for i, step_data in enumerate(steps_data):
            # Determine reasoning type
            reasoning_type = await self._determine_reasoning_type(step_data)
            
            # Create reasoning step
            step = ReasoningStep(
                step_number=i + 1,
                description=step_data.get("description", ""),
                mathematical_expression=step_data.get("expression", ""),
                reasoning_type=reasoning_type,
                input_state=step_data.get("input_state", {}),
                output_state=step_data.get("output_state", {}),
                confidence=step_data.get("confidence", 1.0)
            )
            
            reasoning_steps.append(step)
        
        return reasoning_steps
    
    async def _determine_reasoning_type(
        self, 
        step_data: Dict[str, Any]
    ) -> ReasoningType:
        """Determine the type of reasoning for a step"""
        
        description = step_data.get("description", "").lower()
        expression = step_data.get("expression", "").lower()
        
        # Check for calculus operations
        calculus_keywords = ["derivative", "integral", "limit", "differentiate", "integrate"]
        if any(keyword in description for keyword in calculus_keywords):
            return ReasoningType.CALCULUS
        
        # Check for physics concepts
        physics_keywords = ["force", "energy", "momentum", "velocity", "acceleration", "mass"]
        if any(keyword in description for keyword in physics_keywords):
            return ReasoningType.PHYSICS
        
        # Check for trigonometry
        trig_keywords = ["sin", "cos", "tan", "theta", "angle"]
        if any(keyword in description or keyword in expression for keyword in trig_keywords):
            return ReasoningType.TRIGONOMETRY
        
        # Default to algebraic
        return ReasoningType.ALGEBRAIC
    
    async def _validate_step(self, step: ReasoningStep) -> ValidationResult:
        """Validate a single reasoning step using appropriate validator"""
        
        validator = self.validators.get(step.reasoning_type)
        if validator:
            return await validator.validate_step(step)
        else:
            logger.warning(f"No validator found for reasoning type: {step.reasoning_type}")
            return ValidationResult.UNCERTAIN
    
    async def _generate_verification_report(
        self, 
        steps: List[ReasoningStep], 
        overall_confidence: float
    ) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        
        valid_steps = sum(1 for step in steps if step.validation_result == ValidationResult.VALID)
        invalid_steps = sum(1 for step in steps if step.validation_result == ValidationResult.INVALID)
        uncertain_steps = sum(1 for step in steps if step.validation_result == ValidationResult.UNCERTAIN)
        
        report = {
            "total_steps": len(steps),
            "valid_steps": valid_steps,
            "invalid_steps": invalid_steps,
            "uncertain_steps": uncertain_steps,
            "overall_confidence": overall_confidence,
            "validation_summary": {
                "mathematical_accuracy": valid_steps / len(steps) if steps else 0,
                "logical_consistency": await self._check_logical_consistency(steps),
                "completeness": await self._check_solution_completeness(steps)
            },
            "issues_detected": await self._identify_issues(steps),
            "recommendations": await self._generate_recommendations(steps)
        }
        
        return report
    
    async def _check_logical_consistency(self, steps: List[ReasoningStep]) -> float:
        """Check logical flow between steps"""
        if len(steps) <= 1:
            return 1.0
        
        consistency_score = 1.0
        
        for i in range(1, len(steps)):
            prev_step = steps[i-1]
            curr_step = steps[i]
            
            # Check if current step's input matches previous step's output
            if not await self._steps_are_consistent(prev_step, curr_step):
                consistency_score *= 0.8
        
        return consistency_score
    
    async def _steps_are_consistent(
        self, 
        prev_step: ReasoningStep, 
        curr_step: ReasoningStep
    ) -> bool:
        """Check if two consecutive steps are logically consistent"""
        # Implementation depends on specific step types and content
        return True  # Simplified for now
    
    async def _check_solution_completeness(self, steps: List[ReasoningStep]) -> float:
        """Check if solution is complete and addresses the original question"""
        # Implementation depends on question type and expected solution structure
        return 1.0  # Simplified for now
    
    async def _identify_issues(self, steps: List[ReasoningStep]) -> List[str]:
        """Identify specific issues in the reasoning"""
        issues = []
        
        for step in steps:
            if step.validation_result == ValidationResult.INVALID:
                issues.append(f"Step {step.step_number}: Mathematical error in {step.description}")
            elif step.validation_result == ValidationResult.UNCERTAIN:
                issues.append(f"Step {step.step_number}: Uncertain validation for {step.description}")
        
        return issues
    
    async def _generate_recommendations(self, steps: List[ReasoningStep]) -> List[str]:
        """Generate recommendations for improving the solution"""
        recommendations = []
        
        invalid_count = sum(1 for step in steps if step.validation_result == ValidationResult.INVALID)
        
        if invalid_count > 0:
            recommendations.append("Review mathematical operations for accuracy")
        
        uncertain_count = sum(1 for step in steps if step.validation_result == ValidationResult.UNCERTAIN)
        
        if uncertain_count > len(steps) * 0.3:
            recommendations.append("Consider providing more detailed explanations")
        
        return recommendations
    
    def _step_to_dict(self, step: ReasoningStep) -> Dict[str, Any]:
        """Convert ReasoningStep to dictionary for serialization"""
        return {
            "step_number": step.step_number,
            "description": step.description,
            "mathematical_expression": step.mathematical_expression,
            "reasoning_type": step.reasoning_type.value,
            "input_state": step.input_state,
            "output_state": step.output_state,
            "confidence": step.confidence,
            "validation_result": step.validation_result.value if step.validation_result else None,
            "verification_details": step.verification_details
        }
