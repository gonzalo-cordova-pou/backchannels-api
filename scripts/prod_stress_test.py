"""
HTTP-based stress test for production environment testing
Tests the full API stack including HTTP overhead, serialization, etc.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp

from .test_data import generate_test_data


@dataclass
class APITestResult:
    success: bool
    latency_ms: float
    status_code: int
    error: Optional[str] = None
    response_data: Optional[dict] = None


class HTTPStressTest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def single_request(
        self, session: aiohttp.ClientSession, text: str, context: Optional[str] = None
    ) -> APITestResult:
        """Make a single HTTP request to the API"""
        payload = {"utterance": text, "previous_utterance": context}

        start_time = time.perf_counter()
        try:
            async with session.post(
                f"{self.base_url}/api/v1/predict",  # Adjust endpoint as needed
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as response:
                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000

                if response.status == 200:
                    data = await response.json()
                    return APITestResult(
                        success=True,
                        latency_ms=latency,
                        status_code=response.status,
                        response_data=data,
                    )
                else:
                    error_text = await response.text()
                    return APITestResult(
                        success=False,
                        latency_ms=latency,
                        status_code=response.status,
                        error=f"HTTP {response.status}: {error_text}",
                    )

        except asyncio.TimeoutError:
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000
            return APITestResult(
                success=False,
                latency_ms=latency,
                status_code=0,
                error="Request timeout",
            )
        except Exception as e:
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000
            return APITestResult(
                success=False,
                latency_ms=latency,
                status_code=0,
                error=f"{type(e).__name__}: {str(e)}",
            )

    async def run_concurrent_test(
        self, num_requests: int, max_concurrent: int
    ) -> Dict[str, Any]:
        """Run concurrent HTTP requests (simulating multiple voice agents)"""
        print(
            f"Running HTTP test: {num_requests} requests, "
            f"{max_concurrent} concurrent..."
        )

        # Generate test data
        test_samples = generate_test_data(num_requests)

        # Use the full test samples with both text and context
        requests_data = test_samples

        # Configure session for concurrent requests
        connector = aiohttp.TCPConnector(
            limit=max_concurrent + 10,
            limit_per_host=max_concurrent + 5,
            ttl_dns_cache=300,
        )

        timeout = aiohttp.ClientTimeout(total=10.0, connect=2.0)

        results = []
        start_time = time.perf_counter()

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)

            async def limited_request(
                sample: tuple[str, Optional[str]]
            ) -> APITestResult:
                async with semaphore:
                    text, context = sample
                    return await self.single_request(session, text, context)

            # Execute all requests
            tasks = [limited_request(sample) for sample in requests_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.perf_counter() - start_time

        # Process results
        valid_results = [r for r in results if isinstance(r, APITestResult)]
        successful = [r for r in valid_results if r.success]
        failed = [r for r in valid_results if not r.success]

        return {
            "test_type": "http_concurrent",
            "num_requests": num_requests,
            "max_concurrent": max_concurrent,
            "total_time": total_time,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": (
                len(successful) / len(valid_results) if valid_results else 0
            ),
            "throughput": num_requests / total_time,
            "results": valid_results,
        }

    def analyze_results(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze HTTP test results"""
        results = test_result["results"]
        successful = [r for r in results if r.success]

        if not successful:
            return {
                "avg_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "error_summary": {},
            }

        latencies = [r.latency_ms for r in successful]

        # Error analysis
        failed = [r for r in results if not r.success]
        error_summary: dict[str, int] = {}
        for result in failed:
            error_key = (
                f"Status {result.status_code}"
                if result.status_code > 0
                else "Network Error"
            )
            error_summary[error_key] = error_summary.get(error_key, 0) + 1

        return {
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "error_summary": error_summary,
        }

    def print_results(self, test_result: Dict[str, Any], analysis: Dict[str, Any]):
        """Print formatted results"""
        print(f"\n{'='*60}")
        print("HTTP API STRESS TEST RESULTS")
        print(f"{'='*60}")

        print(f"Total requests: {test_result['num_requests']}")
        print(f"Max concurrent: {test_result['max_concurrent']}")
        print(f"Successful: {test_result['successful']}")
        print(f"Failed: {test_result['failed']}")
        print(f"Success rate: {test_result['success_rate']:.1%}")
        print(f"Total time: {test_result['total_time']:.2f}s")
        print(f"Throughput: {test_result['throughput']:.1f} req/s")

        print("\nLatency (including HTTP overhead):")
        print(f"  Average: {analysis['avg_latency_ms']:.1f}ms")
        print(f"  P50: {analysis['p50_latency_ms']:.1f}ms")
        print(f"  P95: {analysis['p95_latency_ms']:.1f}ms")
        print(f"  P99: {analysis['p99_latency_ms']:.1f}ms")
        print(f"  Min: {analysis['min_latency_ms']:.1f}ms")
        print(f"  Max: {analysis['max_latency_ms']:.1f}ms")

        if analysis["error_summary"]:
            print("\nErrors:")
            for error_type, count in analysis["error_summary"].items():
                print(f"  {error_type}: {count}")


async def main():
    """Run comprehensive HTTP stress test"""
    tester = HTTPStressTest("http://localhost:8000")

    # Test scenarios matching your use case
    scenarios = [
        (100, 4),  # Light load
        (500, 8),  # Medium load
        (1000, 10),  # Heavy load (your target: 10 calls)
        (1000, 16),  # Stress test
    ]

    for num_requests, max_concurrent in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {num_requests} requests, {max_concurrent} concurrent")
        print(f"{'='*80}")

        try:
            result = await tester.run_concurrent_test(num_requests, max_concurrent)
            analysis = tester.analyze_results(result)
            tester.print_results(result, analysis)

            # Assessment
            avg_latency = analysis["avg_latency_ms"]
            if avg_latency < 50:
                print(f"✅ PASSED: {avg_latency:.1f}ms < 50ms target")
            else:
                print(f"❌ FAILED: {avg_latency:.1f}ms > 50ms target")

        except Exception as e:
            print(f"❌ Test failed: {e}")

        print("\nWaiting 5s before next test...")
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
