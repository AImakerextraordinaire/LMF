"""
Neural Anamnesis Python Client

Async HTTP client for connecting the LMF harness to the Neural Anamnesis
Rust service. Handles tensor serialization, connection pooling, retry logic,
and graceful degradation when the service is unavailable.

Author: Kiro (IDE)
Date: 2026-03-05
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError

logger = logging.getLogger(__name__)


@dataclass
class NeuralAnamnStatus:
    """System status from Neural Anamnesis service"""
    total_memories: int
    total_sectors: int
    active_sector: Optional[Dict[str, Any]]
    archived_sectors: int
    cache_utilization: float
    index_sectors: int


class NeuralAnamnClient:
    """
    Async client for Neural Anamnesis service.
    
    Features:
    - Async I/O with connection pooling
    - Automatic health monitoring
    - Graceful degradation when service unavailable
    - Tensor <-> JSON serialization
    - Retry logic for transient failures
    
    Usage:
        async with NeuralAnamnClient() as client:
            # Query memories
            retrieved = await client.query(field_state, emotional_state)
            
            # Write memory
            success = await client.write(pattern, significance, emotional_state)
            
            # Check status
            status = await client.status()
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:6060",
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        health_check_interval: float = 5.0,
    ):
        """
        Initialize Neural Anamnesis client.
        
        Args:
            base_url: Base URL of Neural Anamnesis service
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            health_check_interval: Seconds between health checks
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = ClientTimeout(total=timeout_seconds)
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        
        self.session: Optional[ClientSession] = None
        self._service_available = False
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def connect(self):
        """Initialize HTTP session and start health monitoring"""
        if self.session is None:
            self.session = ClientSession(timeout=self.timeout)
            logger.info(f"Neural Anamnesis client connected to {self.base_url}")
            
            # Start background health check
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def close(self):
        """Close HTTP session and stop health monitoring"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Neural Anamnesis client disconnected")
    
    @property
    def is_available(self) -> bool:
        """Check if service is currently available"""
        return self._service_available
    
    async def _health_check_loop(self):
        """Background task that periodically checks service health"""
        while True:
            try:
                healthy = await self.health()
                if healthy != self._service_available:
                    if healthy:
                        logger.info("Neural Anamnesis service is now available")
                    else:
                        logger.warning("Neural Anamnesis service is unavailable")
                    self._service_available = healthy
            except Exception as e:
                logger.debug(f"Health check error: {e}")
                self._service_available = False
            
            await asyncio.sleep(self.health_check_interval)
    
    async def health(self) -> bool:
        """
        Check if service is healthy.
        
        Returns:
            True if service responds to health check, False otherwise
        """
        if not self.session:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("status") == "healthy"
                return False
        except Exception:
            return False
    
    async def status(self) -> Optional[NeuralAnamnStatus]:
        """
        Get system status from Neural Anamnesis.
        
        Returns:
            NeuralAnamnStatus object if successful, None if service unavailable
        """
        if not self._service_available:
            logger.warning("Neural Anamnesis service unavailable, skipping status check")
            return None
        
        try:
            async with self.session.get(f"{self.base_url}/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return NeuralAnamnStatus(
                        total_memories=data["total_memories"],
                        total_sectors=data["total_sectors"],
                        active_sector=data.get("active_sector"),
                        archived_sectors=data["archived_sectors"],
                        cache_utilization=data["cache_utilization"],
                        index_sectors=data["index_sectors"],
                    )
                else:
                    logger.error(f"Status check failed: HTTP {resp.status}")
                    return None
        except ClientError as e:
            logger.error(f"Status check error: {e}")
            return None
    
    async def query(
        self,
        field_state: torch.Tensor,
        emotional_state: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> Optional[torch.Tensor]:
        """
        Query Neural Anamnesis for memories matching field state.
        
        Args:
            field_state: Field state tensor [field_dim]
            emotional_state: Optional emotional state tensor [emotional_dims]
            top_k: Number of sectors to retrieve from
        
        Returns:
            Retrieved memory tensor [field_dim] if successful, None if unavailable
        """
        if not self._service_available:
            logger.debug("Neural Anamnesis unavailable, skipping query")
            return None
        
        # Serialize tensors to JSON
        payload = {
            "query": {
                "data": field_state.cpu().flatten().tolist(),
                "shape": list(field_state.shape),
            },
            "top_k": top_k,
        }
        
        if emotional_state is not None:
            payload["emotional_state"] = {
                "data": emotional_state.cpu().flatten().tolist(),
                "shape": list(emotional_state.shape),
            }
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.base_url}/query",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Deserialize response to tensor
                        memory_envelope = data["memory"]
                        memory_tensor = torch.tensor(
                            memory_envelope["data"],
                            dtype=field_state.dtype,
                            device=field_state.device
                        ).reshape(memory_envelope["shape"])
                        logger.debug(
                            f"Query successful: {len(data['sectors_used'])} sectors, "
                            f"{data['latency_us']}μs"
                        )
                        return memory_tensor
                    
                    elif resp.status == 400:
                        error = await resp.text()
                        logger.error(f"Query failed: bad request - {error}")
                        return None
                    
                    elif resp.status == 500:
                        logger.warning(f"Query failed: server error (attempt {attempt + 1}/{self.max_retries})")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                            continue
                        return None
                    
                    else:
                        logger.error(f"Query failed: HTTP {resp.status}")
                        return None
            
            except ClientError as e:
                logger.warning(f"Query error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.1 * (2 ** attempt))
                    continue
                return None
        
        return None
    
    async def write(
        self,
        pattern: torch.Tensor,
        significance: float,
        emotional_state: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Write a memory pattern to Neural Anamnesis.
        
        Args:
            pattern: Memory pattern tensor [field_dim]
            significance: Significance score [0.0, 1.0]
            emotional_state: Optional emotional state tensor [emotional_dims]
        
        Returns:
            True if write successful, False otherwise
        """
        if not self._service_available:
            logger.debug("Neural Anamnesis unavailable, skipping write")
            return False
        
        # Validate significance
        if not (0.0 <= significance <= 1.0):
            logger.error(f"Invalid significance: {significance} (must be in [0, 1])")
            return False
        
        # Serialize tensors to JSON
        payload = {
            "pattern": {
                "data": pattern.cpu().flatten().tolist(),
                "shape": list(pattern.shape),
            },
            "significance": significance,
        }
        
        if emotional_state is not None:
            payload["emotional_state"] = {
                "data": emotional_state.cpu().flatten().tolist(),
                "shape": list(emotional_state.shape),
            }
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.base_url}/write",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data["success"]:
                            logger.info(
                                f"Memory written: sector={data['sector_id'][:8]}..., "
                                f"slots={data['slots_used']}, "
                                f"capacity={data['capacity_ratio']:.2%}"
                            )
                            if data.get("sector_allocated"):
                                logger.info("New sector allocated (previous sector frozen)")
                            return True
                        else:
                            logger.error("Write failed: server returned success=false")
                            return False
                    
                    elif resp.status == 400:
                        error = await resp.text()
                        logger.error(f"Write failed: bad request - {error}")
                        return False
                    
                    elif resp.status == 500:
                        logger.warning(f"Write failed: server error (attempt {attempt + 1}/{self.max_retries})")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(0.1 * (2 ** attempt))
                            continue
                        return False
                    
                    elif resp.status == 503:
                        logger.error("Write failed: service unavailable (sector manager locked)")
                        return False
                    
                    elif resp.status == 507:
                        logger.error("Write failed: insufficient storage (cannot allocate new sector)")
                        return False
                    
                    else:
                        logger.error(f"Write failed: HTTP {resp.status}")
                        return False
            
            except ClientError as e:
                logger.warning(f"Write error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.1 * (2 ** attempt))
                    continue
                return False
        
        return False


# Convenience function for synchronous usage
def create_client(base_url: str = "http://localhost:6060") -> NeuralAnamnClient:
    """
    Create a Neural Anamnesis client.
    
    Usage:
        client = create_client()
        async with client:
            await client.write(pattern, significance)
    """
    return NeuralAnamnClient(base_url=base_url)
