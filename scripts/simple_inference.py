import sys
import time

from app.models.predictor import BackchannelPredictor
from scripts.test_data import get_test_samples


def load_model():
    """Load the backchannel detection model"""
    print("Loading backchannel detection model...")
    start_time = time.time()

    try:
        predictor = BackchannelPredictor.from_config({"type": "distilbert"})
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print(f"   Model: {predictor.model.model_name}")
        device = (
            predictor.model.device if hasattr(predictor.model, "device") else "Unknown"
        )
        print(f"   Device: {device}")
        return predictor
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def test_samples():
    """Sample inputs for testing with real-world conversation examples"""
    return get_test_samples()


def run_inference(predictor: BackchannelPredictor):
    """Run inference on sample inputs"""
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE TESTS")
    print("=" * 60)

    samples = test_samples()
    results = []

    for i, (text, context) in enumerate(samples, 1):
        print(f"\nTest {i:2d}: '{text}'")
        if context:
            print(f"        Context: '{context}'")

        try:
            result = predictor.predict(text, context)

            # Format the output
            status = (
                "✅ BACKCHANNEL" if result["is_backchannel"] else "❌ NOT BACKCHANNEL"
            )
            confidence = result["confidence"]
            latency = result["latency_ms"]

            print(f"        Result: {status}")
            print(f"        Confidence: {confidence:.3f}")
            print(f"        Latency: {latency:.1f}ms")

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
            print(f"        ❌ Error: {e}")
            results.append({"text": text, "context": context, "error": str(e)})

    return results


def print_summary(results):
    """Print a summary of the inference results"""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Errors: {len(errors)}")

    if successful:
        backchannels = [r for r in successful if r["is_backchannel"]]
        non_backchannels = [r for r in successful if not r["is_backchannel"]]

        print(f"\nBackchannels detected: {len(backchannels)}")
        print(f"Non-backchannels detected: {len(non_backchannels)}")

        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)

        print(f"\nAverage confidence: {avg_confidence:.3f}")
        print(f"Average latency: {avg_latency:.1f}ms")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - '{error['text']}': {error['error']}")


def main():
    """Main function"""
    print("Backchannel Detection - Simple Inference Test")
    print("=" * 50)

    # Load the model
    predictor = load_model()
    if predictor is None:
        print("❌ Cannot proceed without a loaded model")
        sys.exit(1)

    # Run inference tests
    results = run_inference(predictor)

    # Print summary
    print_summary(results)

    print("\n✅ Inference test completed!")


if __name__ == "__main__":
    main()
