"""
Database Connection Management
Async connections to PostgreSQL (Supabase), Redis, and Vector DB (Pinecone)
"""

import logging
from typing import Optional, AsyncGenerator
import redis.asyncio as redis
from pinecone import Pinecone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
import contextlib

from app.config import get_settings
from app.models import Base


logger = logging.getLogger(__name__)
settings = get_settings()


class DatabaseManager:
    """Manages all database connections"""
    
    def __init__(self):
        self.postgres_engine = None
        self.postgres_session_factory = None
        self.redis_client = None
        self.pinecone_client = None
        self.pinecone_index = None
        
    async def initialize(self):
        """Initialize all database connections"""
        try:
            await self._setup_postgres()
            await self._setup_redis()
            await self._setup_pinecone()
            logger.info("All database connections initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise
    
    async def _setup_postgres(self):
        """Setup PostgreSQL connection with Supabase"""
        try:
            # Create async engine
            self.postgres_engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DATABASE_ECHO,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                poolclass=NullPool if settings.ENVIRONMENT == "test" else None,
                connect_args={
                    "server_settings": {
                        "application_name": "fynqAI-backend",
                    }
                }
            )
            
            # Create session factory
            self.postgres_session_factory = async_sessionmaker(
                bind=self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Test connection
            async with self.postgres_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                
            logger.info("PostgreSQL connection established successfully")
            
        except Exception as e:
            logger.error(f"PostgreSQL setup failed: {e}", exc_info=True)
            raise
    
    async def _setup_redis(self):
        """Setup Redis connection for caching and sessions"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Redis setup failed: {e}", exc_info=True)
            raise
    
    async def _setup_pinecone(self):
        """Setup Pinecone vector database connection"""
        try:
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(
                api_key=settings.PINECONE_API_KEY,
                environment=settings.PINECONE_ENVIRONMENT
            )
            
            # Connect to the index
            self.pinecone_index = self.pinecone_client.Index(settings.PINECONE_INDEX_NAME)
            
            # Test connection by getting index stats
            stats = self.pinecone_index.describe_index_stats()
            logger.info(f"Pinecone connected - Index stats: {stats}")
            
        except Exception as e:
            logger.error(f"Pinecone setup failed: {e}", exc_info=True)
            raise
    
    async def close(self):
        """Close all database connections"""
        try:
            if self.postgres_engine:
                await self.postgres_engine.dispose()
                logger.info("PostgreSQL connection closed")
            
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            
            # Pinecone doesn't need explicit closing
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}", exc_info=True)
    
    @contextlib.asynccontextmanager
    async def get_postgres_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get PostgreSQL session with automatic cleanup"""
        if not self.postgres_session_factory:
            raise RuntimeError("PostgreSQL not initialized")
        
        async with self.postgres_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        return self.redis_client
    
    def get_pinecone_index(self):
        """Get Pinecone index"""
        if not self.pinecone_index:
            raise RuntimeError("Pinecone not initialized")
        return self.pinecone_index
    
    async def health_check(self) -> dict:
        """Perform health check on all database connections"""
        health = {
            "postgres": "unknown",
            "redis": "unknown",
            "pinecone": "unknown"
        }
        
        # Check PostgreSQL
        try:
            async with self.get_postgres_session() as session:
                result = await session.execute("SELECT 1")
                if result.scalar() == 1:
                    health["postgres"] = "healthy"
                else:
                    health["postgres"] = "unhealthy"
        except Exception as e:
            health["postgres"] = f"error: {str(e)}"
        
        # Check Redis
        try:
            await self.redis_client.ping()
            health["redis"] = "healthy"
        except Exception as e:
            health["redis"] = f"error: {str(e)}"
        
        # Check Pinecone
        try:
            stats = self.pinecone_index.describe_index_stats()
            if stats:
                health["pinecone"] = "healthy"
            else:
                health["pinecone"] = "unhealthy"
        except Exception as e:
            health["pinecone"] = f"error: {str(e)}"
        
        return health


# Global database manager instance
db_manager = DatabaseManager()


# Dependency functions for FastAPI
async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for PostgreSQL session"""
    async with db_manager.get_postgres_session() as session:
        yield session


async def get_redis() -> redis.Redis:
    """FastAPI dependency for Redis client"""
    return await db_manager.get_redis()


def get_pinecone_index():
    """FastAPI dependency for Pinecone index"""
    return db_manager.get_pinecone_index()


# Connection pool monitoring
class ConnectionPoolMonitor:
    """Monitor database connection pool health"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def get_postgres_pool_info(self) -> dict:
        """Get PostgreSQL connection pool information"""
        if not self.db_manager.postgres_engine:
            return {"status": "not_initialized"}
        
        pool = self.db_manager.postgres_engine.pool
        
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
    
    async def get_redis_info(self) -> dict:
        """Get Redis connection information"""
        if not self.db_manager.redis_client:
            return {"status": "not_initialized"}
        
        try:
            info = await self.db_manager.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def get_pinecone_info(self) -> dict:
        """Get Pinecone index information"""
        if not self.db_manager.pinecone_index:
            return {"status": "not_initialized"}
        
        try:
            stats = self.db_manager.pinecone_index.describe_index_stats()
            return {
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_fullness": stats.get("index_fullness", 0.0),
                "namespaces": stats.get("namespaces", {})
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def get_all_info(self) -> dict:
        """Get information about all database connections"""
        return {
            "postgres": await self.get_postgres_pool_info(),
            "redis": await self.get_redis_info(),
            "pinecone": await self.get_pinecone_info()
        }


# Initialize monitor
pool_monitor = ConnectionPoolMonitor(db_manager)


# Cache management utilities
class CacheManager:
    """Redis cache management utilities"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str, default=None):
        """Get value from cache"""
        try:
            value = await self.redis.get(key)
            return value if value is not None else default
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Set value in cache with TTL"""
        try:
            ttl = ttl or self.default_ttl
            await self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str):
        """Delete key from cache"""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def incr(self, key: str, amount: int = 1, ttl: Optional[int] = None):
        """Increment counter in cache"""
        try:
            result = await self.redis.incr(key, amount)
            if ttl:
                await self.redis.expire(key, ttl)
            return result
        except Exception as e:
            logger.error(f"Cache incr error for key {key}: {e}")
            return None
    
    async def get_keys_by_pattern(self, pattern: str):
        """Get keys matching pattern"""
        try:
            return await self.redis.keys(pattern)
        except Exception as e:
            logger.error(f"Cache keys error for pattern {pattern}: {e}")
            return []
    
    async def flush_pattern(self, pattern: str):
        """Delete all keys matching pattern"""
        try:
            keys = await self.get_keys_by_pattern(pattern)
            if keys:
                await self.redis.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.error(f"Cache flush error for pattern {pattern}: {e}")
            return 0


async def get_cache_manager() -> CacheManager:
    """FastAPI dependency for cache manager"""
    redis_client = await get_redis()
    return CacheManager(redis_client)
