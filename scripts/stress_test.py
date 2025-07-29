"""
Production HTTP load testing

Tests realistic HTTP load patterns and provides deployment recommendations:
- Finds single instance capacity limits
- Calculates required instances for target load
- Provides production deployment guidance
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from scripts.test_data import generate_test_data

# Configuration constants
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TARGET_RPS = 1000
TIMEOUT_MS = 500
CONNECT_TIMEOUT_MS = 2000  # 2s connection timeout
MAX_CONCURRENT_DEFAULT = 500
CONNECTOR_LIMIT_BUFFER = 50
CONNECTOR_PER_HOST_BUFFER = 25
DNS_CACHE_TTL = 300

# Performance thresholds
SUCCESS_RATE_THRESHOLD = 0.95  # 95% success rate required
AVG_LATENCY_THRESHOLD_MS = 50  # 50ms average latency target
P99_LATENCY_THRESHOLD_MS = 200  # 200ms P99 latency limit
RPS_ACHIEVEMENT_THRESHOLD = 0.9  # 90% of target RPS required

# Test configuration
TEST_SCENARIOS = [
    (100, 1),  # Warm-up
    (200, 1),  # Light load
    (300, 2),  # Medium load
    (400, 2),  # Heavy load
    (500, 2),  # Stress test
    (600, 2),  # Find breaking point
    (750, 2),  # Push further
    (1000, 2),  # Maximum test
]

# Progress reporting
PROGRESS_REPORT_INTERVAL = 500  # Report every 500 requests
PROGRESS_TIME_INTERVAL = 30  # Report every 30 seconds

# Recovery and timing
RECOVERY_DELAY_SECONDS = 5

# Cost estimation
ESTIMATED_COST_PER_INSTANCE_MONTHLY = 50  # $50/instance/month estimate

# Deployment thresholds
EXCELLENT_CAPACITY_RPS = 1000
MIN_ACCEPTABLE_CAPACITY_RPS = 200


@dataclass
class TestResult:
    success: bool
    latency_ms: float
    status_code: int
    timestamp: float
    error: Optional[str] = None


class ProductionLoadTester:
    """HTTP load tester for backchannel detection production capacity planning"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    async def single_request(
        self, session: aiohttp.ClientSession, text: str, context: Optional[str] = None
    ) -> TestResult:
        """Make a single HTTP request with conversational timeout requirements"""
        payload = {"utterance": text, "previous_utterance": context}
        start_time = time.perf_counter()
        timestamp = time.time()

        try:
            async with session.post(
                f"{self.base_url}/api/v1/predict",
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=TIMEOUT_MS / 1000
                ),  # Conversational latency limit
            ) as response:
                latency = (time.perf_counter() - start_time) * 1000

                if response.status == 200:
                    await response.json()  # Consume response
                    return TestResult(True, latency, response.status, timestamp)
                else:
                    error_text = await response.text()
                    return TestResult(
                        False,
                        latency,
                        response.status,
                        timestamp,
                        f"HTTP {response.status}: {error_text}",
                    )

        except asyncio.TimeoutError:
            latency = (time.perf_counter() - start_time) * 1000
            return TestResult(
                False,
                latency,
                0,
                timestamp,
                f"Timeout (>{TIMEOUT_MS}ms) - Too slow for conversational use",
            )
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return TestResult(
                False, latency, 0, timestamp, f"{type(e).__name__}: {str(e)}"
            )

    async def capacity_test(
        self,
        target_rps: int,
        duration_minutes: int,
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Test sustained load capacity"""

        if max_concurrent is None:
            max_concurrent = min(target_rps, MAX_CONCURRENT_DEFAULT)

        print(f"\nTesting {target_rps} req/s for {duration_minutes} minute(s)")
        print(f"Max concurrent: {max_concurrent}")

        total_requests = target_rps * duration_minutes * 60
        request_interval = 1.0 / target_rps
        test_samples = generate_test_data(total_requests)

        # Configure HTTP session
        connector = aiohttp.TCPConnector(
            limit=max_concurrent + CONNECTOR_LIMIT_BUFFER,
            limit_per_host=max_concurrent + CONNECTOR_PER_HOST_BUFFER,
            ttl_dns_cache=DNS_CACHE_TTL,
            enable_cleanup_closed=True,
        )

        results = []
        start_time = time.perf_counter()

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=TIMEOUT_MS / 1000, connect=CONNECT_TIMEOUT_MS / 1000
            ),
            headers={"Connection": "keep-alive"},
        ) as session:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def rate_limited_request(
                sample: Tuple[str, Optional[str]], delay: float
            ):
                await asyncio.sleep(delay)
                async with semaphore:
                    text, context = sample
                    return await self.single_request(session, text, context)

            # Create tasks with proper rate limiting
            tasks = [
                asyncio.create_task(rate_limited_request(sample, i * request_interval))
                for i, sample in enumerate(test_samples)
            ]

            # Execute with progress tracking
            completed = 0
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    results.append(result)
                    completed += 1

                    # Progress reporting
                    if (
                        completed % PROGRESS_REPORT_INTERVAL == 0
                        or (time.perf_counter() - start_time) % PROGRESS_TIME_INTERVAL
                        < 1
                    ):
                        elapsed = time.perf_counter() - start_time
                        current_rps = completed / elapsed if elapsed > 0 else 0
                        if completed % PROGRESS_REPORT_INTERVAL == 0:
                            print(
                                f"Progress: {completed}/{total_requests} "
                                f"({current_rps:.1f} req/s)"
                            )

                except Exception as e:
                    self.logger.error(f"Task failed: {e}")
                    completed += 1

        total_time = time.perf_counter() - start_time
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        return {
            "target_rps": target_rps,
            "duration_minutes": duration_minutes,
            "total_requests": total_requests,
            "completed_requests": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "actual_rps": len(results) / total_time,
            "total_time": total_time,
            "results": results,
        }

    def analyze_performance(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results for conversational requirements"""
        results = test_result["results"]
        successful = [r for r in results if r.success]

        if not successful:
            return {
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "error_summary": self._summarize_errors(
                    [r for r in results if not r.success]
                ),
            }

        latencies = [r.latency_ms for r in successful]
        sorted_latencies = sorted(latencies)

        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": (
                sorted_latencies[int(len(latencies) * 0.95)]
                if len(latencies) > 20
                else max(latencies)
            ),
            "p99_latency_ms": (
                sorted_latencies[int(len(latencies) * 0.99)]
                if len(latencies) > 100
                else max(latencies)
            ),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "error_summary": self._summarize_errors(
                [r for r in results if not r.success]
            ),
        }

    def _summarize_errors(self, failed_results: List[TestResult]) -> Dict[str, int]:
        """Summarize error types"""
        error_summary: Dict[str, int] = {}
        for result in failed_results:
            if "timeout" in (result.error or "").lower():
                error_type = "Timeout (>500ms)"
            elif result.status_code > 0:
                error_type = f"HTTP {result.status_code}"
            else:
                error_type = "Network/Connection Error"
            error_summary[error_type] = error_summary.get(error_type, 0) + 1
        return error_summary

    def is_conversational_ready(
        self, test_result: Dict[str, Any], analysis: Dict[str, Any]
    ) -> bool:
        """Check if performance meets conversational requirements"""
        return (
            test_result["success_rate"] >= SUCCESS_RATE_THRESHOLD
            and analysis["avg_latency_ms"] <= AVG_LATENCY_THRESHOLD_MS
            and analysis["p99_latency_ms"] <= P99_LATENCY_THRESHOLD_MS
            and test_result["actual_rps"]
            >= test_result["target_rps"] * RPS_ACHIEVEMENT_THRESHOLD
        )

    def print_test_results(self, test_result: Dict[str, Any], analysis: Dict[str, Any]):
        """Print formatted test results"""
        target_rps = test_result["target_rps"]
        actual_rps = test_result["actual_rps"]
        success_rate = test_result["success_rate"]

        print("\nRESULTS:")
        print(f"Target: {target_rps} req/s")
        print(f"Achieved: {actual_rps:.1f} req/s ({actual_rps/target_rps*100:.1f}%)")
        print(f"Success rate: {success_rate:.1%}")
        print(
            f"Requests: {test_result['successful']}/{test_result['completed_requests']}"
        )

        if analysis["avg_latency_ms"] > 0:
            print(
                f"Latency: {analysis['avg_latency_ms']:.1f}ms avg, "
                f"{analysis['p99_latency_ms']:.1f}ms P99"
            )

        # Conversational assessment
        if self.is_conversational_ready(test_result, analysis):
            print(f"CONVERSATIONAL READY at {target_rps} req/s")
        else:
            print("CONVERSATIONAL LIMITS REACHED")
            if success_rate < SUCCESS_RATE_THRESHOLD:
                threshold_pct = SUCCESS_RATE_THRESHOLD * 100
                print(
                    f"Success rate {success_rate:.1%} < {threshold_pct:.0f}% required"
                )
            if analysis["avg_latency_ms"] > AVG_LATENCY_THRESHOLD_MS:
                print(
                    f"Avg latency {analysis['avg_latency_ms']:.1f}ms > "
                    f"{AVG_LATENCY_THRESHOLD_MS}ms target"
                )
            if analysis["p99_latency_ms"] > P99_LATENCY_THRESHOLD_MS:
                print(
                    f"P99 latency {analysis['p99_latency_ms']:.1f}ms > "
                    f"{P99_LATENCY_THRESHOLD_MS}ms limit"
                )

        # Show errors if significant
        if (
            analysis["error_summary"]
            and test_result["failed"] > test_result["completed_requests"] * 0.05
        ):
            print("Errors:")
            for error_type, count in analysis["error_summary"].items():
                percentage = (count / test_result["completed_requests"]) * 100
                print(f"  {error_type}: {count} ({percentage:.1f}%)")

    async def find_capacity_limits(self) -> Dict[str, Any]:
        """Find the maximum capacity of a single instance"""
        print("FINDING SINGLE INSTANCE CAPACITY LIMITS")
        print("=" * 60)

        # Progressive load testing to find limits
        test_scenarios = TEST_SCENARIOS

        results = []
        max_working_rps = 0

        for target_rps, duration in test_scenarios:
            try:
                test_result = await self.capacity_test(target_rps, duration)
                analysis = self.analyze_performance(test_result)

                self.print_test_results(test_result, analysis)

                results.append((target_rps, test_result, analysis))

                # Check if this load level works
                if self.is_conversational_ready(test_result, analysis):
                    max_working_rps = target_rps
                    print(f"{target_rps} req/s: Working capacity confirmed")
                else:
                    print(f"{target_rps} req/s: Over capacity - stopping tests")
                    break

                # Brief recovery between tests
                if target_rps < 1000:
                    print(f"Waiting {RECOVERY_DELAY_SECONDS}s...")
                    await asyncio.sleep(RECOVERY_DELAY_SECONDS)

            except Exception as e:
                print(f"Test failed at {target_rps} req/s: {e}")
                break

        return {"max_working_rps": max_working_rps, "all_results": results}

    def calculate_deployment_requirements(
        self, capacity_limits: Dict[str, Any], target_rps: int = 1000
    ):
        """Calculate production deployment requirements"""
        max_capacity = capacity_limits["max_working_rps"]

        print("\nPRODUCTION DEPLOYMENT REQUIREMENTS")
        print("=" * 60)
        print(f"Single instance capacity: {max_capacity} req/s")
        print(f"Target requirement: {target_rps} req/s")

        if max_capacity >= target_rps:
            print("SINGLE INSTANCE SUFFICIENT")
            print("Recommended: 1 instance + load balancer for reliability")
            print(f"Capacity utilization: {target_rps/max_capacity*100:.1f}%")
            instances_needed = 1
        else:
            instances_needed = max(2, int(target_rps / max_capacity) + 1)
            total_capacity = instances_needed * max_capacity
            print("HORIZONTAL SCALING REQUIRED")
            print(f"Instances needed: {instances_needed}")
            print(f"Total capacity: {total_capacity} req/s")
            print(f"Capacity utilization: {target_rps/total_capacity*100:.1f}%")

        print("\nDEPLOYMENT CHECKLIST:")
        print("  Load balancer (nginx/HAProxy/cloud LB)")
        print(f"  {instances_needed} FastAPI instances")
        print("  Container orchestration (Kubernetes recommended)")
        print("  Auto-scaling (target: 70% CPU utilization)")
        print("  Monitoring & alerting")

        if instances_needed > 1:
            print("\nKUBERNETES CONFIG EXAMPLE:")
            print(f"  replicas: {instances_needed}")
            print("  resources.requests.cpu: 500m")
            print("  resources.limits.cpu: 2000m")
            print(f"  HPA: min={instances_needed}, max={instances_needed*2}")

        print("\nESTIMATED COSTS (rough):")
        monthly_cost = instances_needed * ESTIMATED_COST_PER_INSTANCE_MONTHLY
        print(f"Infrastructure: ~${monthly_cost}/month")
        cost_per_req = monthly_cost / (target_rps * 60 * 60 * 24 * 30) * 1000
        print(f"Cost per req: ~${cost_per_req:.3f}/1k requests")

        return {
            "instances_needed": instances_needed,
            "total_capacity": instances_needed * max_capacity,
            "utilization_percent": target_rps / (instances_needed * max_capacity) * 100,
            "estimated_monthly_cost": monthly_cost,
        }


async def main():
    """Main production testing function"""
    print("PRODUCTION LOAD TESTING & CAPACITY PLANNING")
    print("=" * 60)
    print("Purpose: Find single instance limits and calculate deployment requirements")
    print("Target: 1,000 req/s (2,000 concurrent conversational sessions)")
    print("=" * 60)

    tester = ProductionLoadTester(DEFAULT_BASE_URL)

    try:
        # Find capacity limits
        capacity_limits = await tester.find_capacity_limits()

        # Calculate deployment requirements
        deployment_req = tester.calculate_deployment_requirements(
            capacity_limits, target_rps=DEFAULT_TARGET_RPS
        )

        print("\nFINAL RECOMMENDATIONS:")
        max_capacity = capacity_limits["max_working_rps"]
        instances = deployment_req["instances_needed"]

        if max_capacity == 0:
            print("CRITICAL: API cannot handle any significant load")
            print("Next steps: Run simple_test.py to debug basic functionality")
        elif max_capacity >= EXCELLENT_CAPACITY_RPS:
            print("EXCELLENT: Single instance meets requirements")
            print("Next steps: Deploy with load balancer for reliability")
        elif max_capacity >= MIN_ACCEPTABLE_CAPACITY_RPS:
            print(f"SCALING NEEDED: Deploy {instances} instances")
            print("Next steps: Set up Kubernetes + load balancer")
        else:
            print("OPTIMIZATION NEEDED: Low single-instance capacity")
            print("Next steps: Optimize model performance first")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
        print("Check that your API is running: uvicorn app.main:app --reload")

    print("\nProduction testing completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    asyncio.run(main())
