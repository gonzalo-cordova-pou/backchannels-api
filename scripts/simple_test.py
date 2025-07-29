"""
Simple functionality test for backchannel detection API
Quick validation that the model loads and works correctly
"""

import sys
import time
from typing import Any, Dict

from app.models.predictor import BackchannelPredictor
from scripts.test_data import get_test_samples

# Configuration constants
MODEL_TYPE = "distilbert"
TEST_SAMPLES_COUNT = 10
PERFORMANCE_TEST_COUNT = 100
PROGRESS_REPORT_INTERVAL = 20

# Performance thresholds
CONVERSATIONAL_AVG_LATENCY_MS = 50
CONVERSATIONAL_P95_LATENCY_MS = 100
SUCCESS_RATE_THRESHOLD = 0.95

# Display formatting
SEPARATOR_LENGTH = 60


def load_model() -> BackchannelPredictor:
    """Load the backchannel detection model"""
    print("Loading backchannel detection model...")
    start_time = time.time()

    try:
        predictor = BackchannelPredictor.from_config({"type": MODEL_TYPE})
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f}s")
        print(f"Model: {predictor.model.model_name}")

        # Show worker configuration
        if hasattr(predictor, "_executor"):
            max_workers = predictor._executor._max_workers
            print(f"ThreadPool workers: {max_workers}")

        return predictor
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)


def run_basic_tests(predictor: BackchannelPredictor) -> Dict[str, Any]:
    """Run basic functionality tests"""
    print("\n" + "=" * SEPARATOR_LENGTH)
    print("BASIC FUNCTIONALITY TESTS")
    print("=" * SEPARATOR_LENGTH)

    samples = get_test_samples()
    results = []
    total_latency = 0

    for i, (text, context) in enumerate(samples[:TEST_SAMPLES_COUNT], 1):
        print(f"\nTest {i:2d}: '{text}'")
        if context:
            print(f"Context: '{context}'")

        try:
            result = predictor.predict(text, context)

            status = "BACKCHANNEL" if result["is_backchannel"] else "NOT BACKCHANNEL"
            confidence = result["confidence"]
            latency = result["latency_ms"]
            total_latency += latency

            print(f"Result: {status}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Latency: {latency:.1f}ms")

            results.append(
                {
                    "text": text,
                    "context": context,
                    "is_backchannel": result["is_backchannel"],
                    "confidence": confidence,
                    "latency_ms": latency,
                }
            )

        except Exception as e:
            print(f"Error: {e}")
            results.append({"text": text, "context": context, "error": str(e)})

    return {
        "results": results,
        "avg_latency_ms": total_latency / len([r for r in results if "error" not in r]),
        "success_count": len([r for r in results if "error" not in r]),
        "error_count": len([r for r in results if "error" in r]),
    }


def run_quick_performance_test(predictor: BackchannelPredictor) -> Dict[str, Any]:
    """Quick performance test - sequential predictions"""
    print("\n" + "=" * SEPARATOR_LENGTH)
    print(f"QUICK PERFORMANCE TEST ({PERFORMANCE_TEST_COUNT} predictions)")
    print("=" * SEPARATOR_LENGTH)

    samples = get_test_samples()
    latencies = []
    errors = 0

    start_time = time.perf_counter()

    for i in range(PERFORMANCE_TEST_COUNT):
        text, context = samples[i % len(samples)]

        try:
            result = predictor.predict(text, context)
            latencies.append(result["latency_ms"])

            if (i + 1) % PROGRESS_REPORT_INTERVAL == 0:
                current_avg = sum(latencies) / len(latencies)
                print(
                    f"Progress: {i+1}/{PERFORMANCE_TEST_COUNT}, "
                    f"Avg latency: {current_avg:.1f}ms"
                )

        except Exception as e:
            errors += 1
            print(f"Error {errors}: {e}")

    total_time = time.perf_counter() - start_time

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    else:
        avg_latency = min_latency = max_latency = p95_latency = 0

    return {
        "total_predictions": PERFORMANCE_TEST_COUNT,
        "successful": len(latencies),
        "errors": errors,
        "total_time_s": total_time,
        "throughput_per_s": PERFORMANCE_TEST_COUNT / total_time,
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "p95_latency_ms": p95_latency,
    }


def print_summary(basic_results: Dict[str, Any], perf_results: Dict[str, Any]):
    """Print comprehensive test summary"""
    print("\n" + "=" * SEPARATOR_LENGTH)
    print("TEST SUMMARY")
    print("=" * SEPARATOR_LENGTH)

    # Basic functionality
    print("FUNCTIONALITY:")
    print(f"Successful predictions: {basic_results['success_count']}")
    print(f"Errors: {basic_results['error_count']}")
    print(f"Average latency: {basic_results['avg_latency_ms']:.1f}ms")

    # Performance
    print("\nPERFORMANCE:")
    print(f"Throughput: {perf_results['throughput_per_s']:.1f} predictions/sec")
    avg_lat = perf_results["avg_latency_ms"]
    p95_lat = perf_results["p95_latency_ms"]
    print(f"Latency: {avg_lat:.1f}ms avg, {p95_lat:.1f}ms P95")
    min_lat = perf_results["min_latency_ms"]
    max_lat = perf_results["max_latency_ms"]
    print(f"Range: {min_lat:.1f}ms - {max_lat:.1f}ms")
    success_rate = perf_results["successful"] / perf_results["total_predictions"]
    print(
        "Success rate: "
        f"{perf_results['successful']}/{perf_results['total_predictions']} "
        f"({success_rate*100:.1f}%)"
    )

    # Conversational Assessment
    print("\nCONVERSATIONAL READINESS:")
    meets_avg = avg_lat <= CONVERSATIONAL_AVG_LATENCY_MS
    meets_p95 = p95_lat <= CONVERSATIONAL_P95_LATENCY_MS
    meets_success = success_rate >= SUCCESS_RATE_THRESHOLD

    if meets_avg and meets_p95 and meets_success:
        print("READY: Model meets conversational latency requirements")
        print(f"Average {avg_lat:.1f}ms < {CONVERSATIONAL_AVG_LATENCY_MS}ms target")
        print(f"P95 {p95_lat:.1f}ms < {CONVERSATIONAL_P95_LATENCY_MS}ms target")
    else:
        print("NEEDS OPTIMIZATION:")
        if avg_lat > CONVERSATIONAL_AVG_LATENCY_MS:
            print(
                f"Average latency {avg_lat:.1f}ms > "
                f"{CONVERSATIONAL_AVG_LATENCY_MS}ms target"
            )
        if p95_lat > CONVERSATIONAL_P95_LATENCY_MS:
            print(
                f"P95 latency {p95_lat:.1f}ms > "
                f"{CONVERSATIONAL_P95_LATENCY_MS}ms target"
            )
        if success_rate < SUCCESS_RATE_THRESHOLD:
            threshold_pct = SUCCESS_RATE_THRESHOLD * 100
            print(f"Success rate {success_rate:.1%} < {threshold_pct:.0f}% target")

    print("\nNEXT STEPS:")
    if basic_results["error_count"] > 0:
        print("1. Fix prediction errors before load testing")
    elif avg_lat > CONVERSATIONAL_AVG_LATENCY_MS:
        print("1. Optimize model inference speed")
        print("2. Consider model caching or smaller model")
    else:
        print("1. Run production_test.py for capacity planning")
        print("2. Model is ready for HTTP load testing")


def main():
    """Main test function"""
    print("SIMPLE BACKCHANNEL DETECTION TEST")
    print("=" * 50)
    print("Purpose: Quick validation of model functionality and basic performance")
    print("For production capacity planning, run: python -m scripts.production_test")
    print("=" * 50)

    # Load model
    predictor = load_model()

    # Run basic functionality tests
    basic_results = run_basic_tests(predictor)

    # Run quick performance test
    perf_results = run_quick_performance_test(predictor)

    # Print summary with recommendations
    print_summary(basic_results, perf_results)

    print("\nSimple test completed!")


if __name__ == "__main__":
    main()
