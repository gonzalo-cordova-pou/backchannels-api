import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from app.models.predictor import BackchannelPredictor
from scripts.test_data import generate_test_data


class StressTest:
    """Stress testing class for backchannel detection model"""

    def __init__(self, predictor: BackchannelPredictor):
        self.predictor = predictor
        self.results: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def get_test_data(self, num_samples: int) -> List[tuple]:
        """Generate test data for stress testing"""
        return generate_test_data(num_samples)

    def single_inference(
        self, text: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a single inference and return timing data"""
        start_time = time.perf_counter()

        try:
            result = self.predictor.predict(text, context)
            end_time = time.perf_counter()

            return {
                "success": True,
                "text": text,
                "context": context,
                "is_backchannel": result["is_backchannel"],
                "confidence": result["confidence"],
                "latency_ms": (end_time - start_time) * 1000,
                "error": None,
            }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "success": False,
                "text": text,
                "context": context,
                "latency_ms": (end_time - start_time) * 1000,
                "error": str(e),
            }

    def run_sequential_test(self, num_samples: int) -> Dict[str, Any]:
        """Run sequential inference test"""
        print(f"Running sequential test with {num_samples} samples...")

        test_data = self.get_test_data(num_samples)
        results = []

        start_time = time.perf_counter()
        for text, context in test_data:
            result = self.single_inference(text, context)
            results.append(result)
        total_time = time.perf_counter() - start_time

        return {
            "test_type": "sequential",
            "num_samples": num_samples,
            "total_time": total_time,
            "throughput": num_samples / total_time,
            "results": results,
        }

    def run_concurrent_test(self, num_samples: int, max_workers: int) -> Dict[str, Any]:
        """Run concurrent inference test"""
        print(
            f"Running concurrent test with {num_samples} samples, {max_workers} workers..."
        )

        test_data = self.get_test_data(num_samples)
        results = []

        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.single_inference, text, context)
                for text, context in test_data
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        total_time = time.perf_counter() - start_time

        return {
            "test_type": "concurrent",
            "num_samples": num_samples,
            "max_workers": max_workers,
            "total_time": total_time,
            "throughput": num_samples / total_time,
            "results": results,
        }

    def run_batch_test(self, num_batches: int, batch_size: int) -> Dict[str, Any]:
        """Run batch inference test"""
        print(f"Running batch test with {num_batches} batches of size {batch_size}...")

        all_results = []
        total_samples = 0

        start_time = time.perf_counter()
        for i in range(num_batches):
            # Generate batch data
            batch_texts = [f"test text {i}_{j}" for j in range(batch_size)]

            batch_start = time.perf_counter()
            try:
                batch_results = self.predictor.predict_batch(batch_texts)
                batch_time = time.perf_counter() - batch_start

                for j, result in enumerate(batch_results):
                    all_results.append(
                        {
                            "success": True,
                            "text": batch_texts[j],
                            "is_backchannel": result["is_backchannel"],
                            "confidence": result["confidence"],
                            "latency_ms": result["latency_ms"],
                            "error": None,
                        }
                    )

                total_samples += batch_size

            except Exception as e:
                batch_time = time.perf_counter() - batch_start
                for j in range(batch_size):
                    all_results.append(
                        {
                            "success": False,
                            "text": batch_texts[j],
                            "latency_ms": batch_time * 1000 / batch_size,
                            "error": str(e),
                        }
                    )

        total_time = time.perf_counter() - start_time

        return {
            "test_type": "batch",
            "num_batches": num_batches,
            "batch_size": batch_size,
            "total_samples": total_samples,
            "total_time": total_time,
            "throughput": total_samples / total_time,
            "results": all_results,
        }

    def analyze_results(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results and compute statistics"""
        results = test_result["results"]
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if not successful:
            return {
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "throughput": 0.0,
                "num_successful": 0,
                "num_failed": len(failed),
                "errors": [r["error"] for r in failed],
            }

        latencies = [r["latency_ms"] for r in successful]

        return {
            "success_rate": len(successful) / len(results),
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "throughput": test_result["throughput"],
            "num_successful": len(successful),
            "num_failed": len(failed),
            "errors": [r["error"] for r in failed] if failed else [],
        }

    def print_test_results(self, test_result: Dict[str, Any], analysis: Dict[str, Any]):
        """Print formatted test results"""
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {test_result['test_type'].upper()}")
        print(f"{'='*60}")

        print(f"Total samples: {len(test_result['results'])}")
        print(f"Successful: {analysis['num_successful']}")
        print(f"Failed: {analysis['num_failed']}")
        print(f"Success rate: {analysis['success_rate']:.2%}")
        print(f"Total time: {test_result['total_time']:.2f}s")
        print(f"Throughput: {analysis['throughput']:.2f} samples/sec")

        print("\nLatency Statistics:")
        print(f"  Average: {analysis['avg_latency_ms']:.1f}ms")
        print(f"  Min: {analysis['min_latency_ms']:.1f}ms")
        print(f"  Max: {analysis['max_latency_ms']:.1f}ms")
        print(f"  P50: {analysis['p50_latency_ms']:.1f}ms")
        print(f"  P95: {analysis['p95_latency_ms']:.1f}ms")
        print(f"  P99: {analysis['p99_latency_ms']:.1f}ms")

        if analysis["errors"]:
            print("\nErrors:")
            for error in set(analysis["errors"]):
                count = analysis["errors"].count(error)
                print(f"  - {error} (x{count})")

    def run_full_stress_test(self):
        """Run a comprehensive stress test suite"""
        print("Backchannel Detection - Stress Test Suite")
        print("=" * 50)

        # Test configurations
        tests = [
            ("Sequential Small", lambda: self.run_sequential_test(100)),
            ("Sequential Medium", lambda: self.run_sequential_test(1000)),
            ("Sequential Large", lambda: self.run_sequential_test(5000)),
            ("Concurrent Small", lambda: self.run_concurrent_test(100, 4)),
            ("Concurrent Medium", lambda: self.run_concurrent_test(1000, 8)),
            ("Concurrent Large", lambda: self.run_concurrent_test(5000, 16)),
            ("Batch Small", lambda: self.run_batch_test(10, 10)),
            ("Batch Medium", lambda: self.run_batch_test(50, 20)),
            ("Batch Large", lambda: self.run_batch_test(100, 50)),
        ]

        all_results = []

        for test_name, test_func in tests:
            print(f"\nRunning {test_name}...")
            try:
                test_result = test_func()
                analysis = self.analyze_results(test_result)
                self.print_test_results(test_result, analysis)

                all_results.append(
                    {
                        "test_name": test_name,
                        "test_result": test_result,
                        "analysis": analysis,
                    }
                )

            except Exception as e:
                print(f"❌ {test_name} failed: {e}")

        # Print overall summary
        self.print_overall_summary(all_results)

    def print_overall_summary(self, all_results: List[Dict[str, Any]]):
        """Print overall summary of all tests"""
        print(f"\n{'='*80}")
        print("OVERALL STRESS TEST SUMMARY")
        print(f"{'='*80}")

        print(
            f"{'Test':<20} {'Samples':<10} {'Success Rate':<15} {'Avg Latency':<15} {'Throughput':<15}"
        )
        print("-" * 80)

        for result in all_results:
            test_name = result["test_name"]
            analysis = result["analysis"]
            test_result = result["test_result"]

            samples = len(test_result["results"])
            success_rate = analysis["success_rate"]
            avg_latency = analysis["avg_latency_ms"]
            throughput = analysis["throughput"]

            print(
                f"{test_name:<20} {samples:<10} {success_rate:>13.1%} {avg_latency:>13.1f}ms {throughput:>13.1f}/s"
            )


def load_model():
    """Load the backchannel detection model"""
    print("Loading backchannel detection model...")
    start_time = time.time()

    try:
        predictor = BackchannelPredictor.from_config({"type": "distilbert"})
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        return predictor
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def main():
    """Main function"""
    # Load the model
    predictor = load_model()
    if predictor is None:
        print("❌ Cannot proceed without a loaded model")
        sys.exit(1)

    # Run stress tests
    stress_test = StressTest(predictor)
    stress_test.run_full_stress_test()

    print("\n✅ Stress test completed!")


if __name__ == "__main__":
    main()
