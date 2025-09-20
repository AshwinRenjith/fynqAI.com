"""
Retrieval-Augmented Generation (RAG) System
Advanced knowledge retrieval for exam-specific content
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from abc import ABC, abstractmethod

from app.config import get_settings
from app.core.rag.embeddings import EmbeddingGenerator
from app.core.rag.knowledge_base import KnowledgeBaseManager
from app.database.vector_db import VectorDatabase


logger = logging.getLogger(__name__)
settings = get_settings()


class BaseRetriever(ABC):
    """Abstract base class for document retrievers"""
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query"""
        pass


class VectorRetriever(BaseRetriever):
    """Vector-based semantic retrieval using embeddings"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        self.knowledge_base = KnowledgeBaseManager()
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        subject: Optional[str] = None,
        difficulty_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar documents using vector embeddings
        
        Args:
            query: Search query text
            top_k: Number of documents to retrieve
            subject: Filter by subject (mathematics, physics, chemistry, biology)
            difficulty_level: Filter by difficulty (basic, intermediate, advanced)
        
        Returns:
            List of relevant documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Build metadata filters
            filters = {}
            if subject:
                filters["subject"] = subject
            if difficulty_level:
                filters["difficulty_level"] = difficulty_level
            
            # Search vector database
            search_results = await self.vector_db.search(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            # Enhance results with additional context
            enhanced_results = await self._enhance_results(search_results, query)
            
            logger.info(
                f"Retrieved {len(enhanced_results)} documents for query",
                extra={
                    "query_length": len(query),
                    "subject": subject,
                    "difficulty_level": difficulty_level,
                    "results_count": len(enhanced_results)
                }
            )
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}", exc_info=True)
            return []
    
    async def _enhance_results(
        self, 
        search_results: List[Dict], 
        original_query: str
    ) -> List[Dict[str, Any]]:
        """Enhance search results with additional context and metadata"""
        enhanced_results = []
        
        for result in search_results:
            enhanced_result = {
                "content": result.get("content", ""),
                "title": result.get("title", ""),
                "source": result.get("source", ""),
                "subject": result.get("subject", ""),
                "topic": result.get("topic", ""),
                "difficulty_level": result.get("difficulty_level", ""),
                "relevance_score": result.get("score", 0.0),
                "chapter": result.get("chapter", ""),
                "page_number": result.get("page_number", ""),
                "concepts": result.get("concepts", []),
                "examples": result.get("examples", []),
                "formulas": result.get("formulas", [])
            }
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining vector search with keyword matching"""
    
    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.knowledge_base = KnowledgeBaseManager()
    
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Combine vector search with keyword-based retrieval
        
        Uses both semantic similarity and exact keyword matching
        to ensure comprehensive coverage of relevant content
        """
        try:
            # Perform vector search
            vector_results = await self.vector_retriever.retrieve(query, top_k)
            
            # Perform keyword search
            keyword_results = await self._keyword_search(query, top_k)
            
            # Merge and rank results
            combined_results = await self._merge_results(
                vector_results, 
                keyword_results, 
                top_k
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}", exc_info=True)
            return await self.vector_retriever.retrieve(query, top_k)
    
    async def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search for exact matches"""
        # Extract important keywords from query
        keywords = await self._extract_keywords(query)
        
        # Search knowledge base for keyword matches
        keyword_matches = await self.knowledge_base.search_by_keywords(
            keywords, top_k
        )
        
        return keyword_matches
    
    async def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        # Simple keyword extraction (can be enhanced with NLP)
        import re
        
        # Mathematical terms, formulas, concepts
        math_patterns = [
            r"∫|∑|∏|√|π|θ|α|β|γ|δ|λ|μ|σ|φ|ψ|ω",  # Mathematical symbols
            r"\b(?:sin|cos|tan|log|ln|exp|lim|max|min)\b",  # Mathematical functions
            r"\b(?:derivative|integral|limit|matrix|vector|equation)\b",  # Mathematical concepts
        ]
        
        keywords = []
        
        # Extract mathematical terms
        for pattern in math_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            keywords.extend(matches)
        
        # Extract general keywords (length > 3, not common words)
        common_words = {"the", "and", "or", "but", "for", "with", "from", "what", "how", "when", "where"}
        words = re.findall(r'\b\w{4,}\b', query.lower())
        keywords.extend([word for word in words if word not in common_words])
        
        return list(set(keywords))
    
    async def _merge_results(
        self, 
        vector_results: List[Dict], 
        keyword_results: List[Dict], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Merge and rank results from different retrieval methods"""
        
        # Create a scoring system that combines vector similarity and keyword relevance
        all_results = {}
        
        # Add vector results with semantic score
        for result in vector_results:
            result_id = result.get("id", hash(result.get("content", "")))
            all_results[result_id] = result.copy()
            all_results[result_id]["semantic_score"] = result.get("relevance_score", 0.0)
            all_results[result_id]["keyword_score"] = 0.0
        
        # Add keyword results with keyword score
        for result in keyword_results:
            result_id = result.get("id", hash(result.get("content", "")))
            if result_id in all_results:
                all_results[result_id]["keyword_score"] = result.get("relevance_score", 0.0)
            else:
                all_results[result_id] = result.copy()
                all_results[result_id]["semantic_score"] = 0.0
                all_results[result_id]["keyword_score"] = result.get("relevance_score", 0.0)
        
        # Calculate combined score
        for result_id, result in all_results.items():
            semantic_weight = 0.7
            keyword_weight = 0.3
            
            combined_score = (
                semantic_weight * result["semantic_score"] + 
                keyword_weight * result["keyword_score"]
            )
            result["combined_score"] = combined_score
            result["relevance_score"] = combined_score
        
        # Sort by combined score and return top_k
        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        return sorted_results[:top_k]


class ExamSpecificRetriever(BaseRetriever):
    """Specialized retriever for competitive exam content"""
    
    def __init__(self):
        self.hybrid_retriever = HybridRetriever()
        self.exam_patterns = {
            "jee": ["JEE Main", "JEE Advanced", "IIT-JEE"],
            "neet": ["NEET", "Medical entrance"],
            "boards": ["CBSE", "ICSE", "State Board"]
        }
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        exam_type: str = "jee"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve content specifically relevant to competitive exams
        
        Filters and ranks content based on exam patterns,
        previous year questions, and syllabus relevance
        """
        try:
            # Get base results from hybrid retriever
            base_results = await self.hybrid_retriever.retrieve(query, top_k * 2)
            
            # Filter and rank for exam relevance
            exam_filtered_results = await self._filter_for_exam(
                base_results, exam_type
            )
            
            # Add exam-specific metadata
            enhanced_results = await self._add_exam_metadata(
                exam_filtered_results, exam_type
            )
            
            return enhanced_results[:top_k]
            
        except Exception as e:
            logger.error(f"Exam-specific retrieval failed: {e}", exc_info=True)
            return await self.hybrid_retriever.retrieve(query, top_k)
    
    async def _filter_for_exam(
        self, 
        results: List[Dict], 
        exam_type: str
    ) -> List[Dict[str, Any]]:
        """Filter results for exam-specific relevance"""
        
        exam_keywords = self.exam_patterns.get(exam_type, [])
        filtered_results = []
        
        for result in results:
            # Check for exam-specific content
            content = result.get("content", "").lower()
            source = result.get("source", "").lower()
            
            # Calculate exam relevance score
            exam_score = 0.0
            
            for keyword in exam_keywords:
                if keyword.lower() in content or keyword.lower() in source:
                    exam_score += 0.2
            
            # Boost score for previous year questions
            if "previous year" in content or "pyq" in content:
                exam_score += 0.3
            
            # Boost score for syllabus-relevant content
            if "syllabus" in content or "curriculum" in content:
                exam_score += 0.2
            
            # Combine with original relevance score
            original_score = result.get("relevance_score", 0.0)
            combined_score = 0.6 * original_score + 0.4 * exam_score
            
            result["exam_relevance_score"] = exam_score
            result["relevance_score"] = combined_score
            
            filtered_results.append(result)
        
        # Sort by combined relevance score
        return sorted(
            filtered_results, 
            key=lambda x: x["relevance_score"], 
            reverse=True
        )
    
    async def _add_exam_metadata(
        self, 
        results: List[Dict], 
        exam_type: str
    ) -> List[Dict[str, Any]]:
        """Add exam-specific metadata to results"""
        
        for result in results:
            result["exam_type"] = exam_type
            result["is_previous_year_question"] = (
                "previous year" in result.get("content", "").lower() or
                "pyq" in result.get("content", "").lower()
            )
            result["difficulty_category"] = await self._categorize_difficulty(
                result, exam_type
            )
        
        return results
    
    async def _categorize_difficulty(
        self, 
        result: Dict, 
        exam_type: str
    ) -> str:
        """Categorize difficulty based on exam type and content analysis"""
        
        content = result.get("content", "").lower()
        
        # JEE-specific difficulty categorization
        if exam_type == "jee":
            if "jee advanced" in content:
                return "advanced"
            elif "jee main" in content:
                return "intermediate"
            else:
                return "basic"
        
        # Default difficulty from existing metadata
        return result.get("difficulty_level", "intermediate")
