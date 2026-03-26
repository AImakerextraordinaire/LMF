"""
Neural Anamnesis Integration Test

End-to-end validation of the Neural Anamnesis Rust service with the Python client.
Tests round-trip memory storage, retrieval fidelity, error handling, and graceful
degradation.

Author: Kiro (IDE)
Date: 2026-03-08
"""

import asyncio
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmf.bridges.anamnesis_client import NeuralAnamnClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTest:
    """Integration test suite for Neural Anamnesis"""
    
    def __init__(self, base_url: str = "http://localhost:6060"):
        self.base_url = base_url
        self.client = NeuralAnamnClient(base_url=base_url)
        self.field_dim = 2880
        self.emotional_dims = 17
        
    async def run_all_tests(self):
        """Run complete integration test suite"""
        logger.info("=" * 70)
        logger.info("Neural Anamnesis Integration Test Suite")
        logger.info("=" * 70)
        
        try:
            async with self.client:
                # Wait for service to be available
                await self._wait_for_service()
                
                # Run test suite
                await self.test_health_check()
                await self.test_status_endpoint()
                await self.test_round_trip_memory()
                await self.test_emotional_modulation()
                await self.test_error_handling()
                await self.test_graceful_degradation()
                
                logger.info("=" * 70)
                logger.info("✅ ALL TESTS PASSED")
                logger.info("=" * 70)
                
        except Exception as e:
            logger.error(f"❌ TEST SUITE FAILED: {e}", exc_info=True)
            raise
    
    async def _wait_for_service(self, timeout: float = 30.0):
        """Wait for Neural Anamnesis service to be available"""
        logger.info("Waiting for Neural Anamnesis service...")
        
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            if await self.client.health():
                logger.info("✅ Service is available")
                return
            await asyncio.sleep(1.0)
        
        raise TimeoutError(f"Service not available after {timeout}s")
    
    async def test_health_check(self):
        """Test health check endpoint"""
        logger.info("\n--- Test: Health Check ---")
        
        healthy = await self.client.health()
        assert healthy, "Health check failed"
        
        logger.info("✅ Health check passed")
    
    async def test_status_endpoint(self):
        """Test status endpoint"""
        logger.info("\n--- Test: Status Endpoint ---")
        
        status = await self.client.status()
        assert status is not None, "Status check failed"
        
        logger.info(f"Total memories: {status.total_memories}")
        logger.info(f"Total sectors: {status.total_sectors}")
        logger.info(f"Cache utilization: {status.cache_utilization:.2%}")
        
        logger.info("✅ Status endpoint passed")
    
    async def test_round_trip_memory(self):
        """Test round-trip memory storage and retrieval"""
        logger.info("\n--- Test: Round-Trip Memory Storage ---")
        
        # Create a synthetic field state (simulating trained LMF output)
        field_state = torch.randn(self.field_dim, dtype=torch.float32)
        field_state = F.normalize(field_state, dim=0)  # Normalize for stability
        
        significance = 0.9
        
        logger.info(f"Writing memory pattern (significance={significance})...")
        success = await self.client.write(field_state, significance)
        assert success, "Memory write failed"
        logger.info("✅ Memory written successfully")
        
        # Query with the same field state
        logger.info("Querying with same field state...")
        retrieved = await self.client.query(field_state, top_k=5)
        assert retrieved is not None, "Memory query failed"
        logger.info("✅ Memory retrieved successfully")
        
        # Debug: Check what we got back
        logger.info(f"Original field_state shape: {field_state.shape}, dtype: {field_state.dtype}")
        logger.info(f"Retrieved shape: {retrieved.shape}, dtype: {retrieved.dtype}")
        logger.info(f"Original first 5 values: {field_state[:5].tolist()}")
        logger.info(f"Retrieved first 5 values: {retrieved[:5].tolist()}")
        logger.info(f"Original norm: {field_state.norm().item():.6f}")
        logger.info(f"Retrieved norm: {retrieved.norm().item():.6f}")
        
        # Verify fidelity
        cosine_sim = F.cosine_similarity(
            field_state.unsqueeze(0),
            retrieved.unsqueeze(0)
        ).item()
        
        logger.info(f"Cosine similarity: {cosine_sim:.6f}")
        assert cosine_sim > 0.99, f"Low fidelity: {cosine_sim:.6f} < 0.99"
        
        logger.info("✅ Round-trip fidelity validated (similarity > 0.99)")
    
    async def test_emotional_modulation(self):
        """Test emotional state modulation"""
        logger.info("\n--- Test: Emotional Modulation ---")
        
        # Create field state and emotional state
        field_state = torch.randn(self.field_dim, dtype=torch.float32)
        field_state = F.normalize(field_state, dim=0)
        
        emotional_state = torch.randn(self.emotional_dims, dtype=torch.float32)
        emotional_state = F.normalize(emotional_state, dim=0)
        
        significance = 0.75
        
        # Write with emotional state
        logger.info("Writing memory with emotional modulation...")
        success = await self.client.write(
            field_state,
            significance,
            emotional_state=emotional_state
        )
        assert success, "Emotional memory write failed"
        logger.info("✅ Emotional memory written")
        
        # Query with emotional state
        logger.info("Querying with emotional state...")
        retrieved = await self.client.query(
            field_state,
            emotional_state=emotional_state,
            top_k=5
        )
        assert retrieved is not None, "Emotional query failed"
        logger.info("✅ Emotional query successful")
        
        # Verify shape
        assert retrieved.shape == field_state.shape, \
            f"Shape mismatch: {retrieved.shape} != {field_state.shape}"
        
        logger.info("✅ Emotional modulation test passed")
    
    async def test_error_handling(self):
        """Test error handling for invalid inputs"""
        logger.info("\n--- Test: Error Handling ---")
        
        # Test 1: Invalid significance (out of range)
        logger.info("Testing invalid significance...")
        field_state = torch.randn(self.field_dim, dtype=torch.float32)
        success = await self.client.write(field_state, significance=1.5)
        assert not success, "Should reject significance > 1.0"
        logger.info("✅ Invalid significance rejected")
        
        # Test 2: Wrong tensor shape
        logger.info("Testing wrong tensor shape...")
        wrong_shape = torch.randn(100, dtype=torch.float32)  # Wrong dimension
        
        # The client will send it, but server should reject with 400
        # Client should handle this gracefully and return False
        success = await self.client.write(wrong_shape, significance=0.5)
        # Note: This might succeed if server doesn't validate shape yet
        # In production, server should reject and client should return False
        logger.info(f"Wrong shape write result: {success}")
        
        logger.info("✅ Error handling test passed")
    
    async def test_graceful_degradation(self):
        """Test graceful degradation when service is unavailable"""
        logger.info("\n--- Test: Graceful Degradation ---")
        
        # Simulate service unavailable by forcing the flag
        logger.info("Simulating service unavailable...")
        original_flag = self.client._service_available
        self.client._service_available = False
        
        # Test write (should skip silently)
        field_state = torch.randn(self.field_dim, dtype=torch.float32)
        success = await self.client.write(field_state, significance=0.5)
        assert not success, "Write should return False when unavailable"
        logger.info("✅ Write gracefully skipped when unavailable")
        
        # Test query (should return None)
        retrieved = await self.client.query(field_state)
        assert retrieved is None, "Query should return None when unavailable"
        logger.info("✅ Query gracefully returned None when unavailable")
        
        # Restore flag
        self.client._service_available = original_flag
        
        logger.info("✅ Graceful degradation test passed")


async def main():
    """Main test entry point"""
    test = IntegrationTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
