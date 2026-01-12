"""Quick test script to verify model setup."""
import sys
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, 'c:/Users/vinod/Desktop/cyberbullying_project1/Cyberbullying-detection/03_models')
sys.path.insert(0, 'c:/Users/vinod/Desktop/cyberbullying_project1/Cyberbullying-detection')

print("=" * 60)
print("CYBERBULLYING DETECTION - SETUP VERIFICATION")
print("=" * 60)

# Test 1: Import model_loader
print("\n✅ Test 1: Importing model_loader...")
try:
    from model_loader import CyberbullyingDetector, list_available_models
    print("   Import successful!")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: List available models
print("\n✅ Test 2: Available models:")
models = list_available_models()
for name, metrics in models.items():
    print(f"   • {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

# Test 3: Load BERT model
print("\n✅ Test 3: Loading BERT model (this may take a moment)...")
try:
    detector = CyberbullyingDetector(model_type='bert')
    print(f"   Model loaded on device: {detector.device}")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    sys.exit(1)

# Test 4: Run predictions
print("\n✅ Test 4: Running predictions...")
test_texts = [
    "You're such a loser, nobody likes you!",
    "Great job on the presentation today!",
    "I hate you so much!",
    "Thanks for helping me 😊"
]

for text in test_texts:
    result = detector.predict(text)
    icon = "🚨" if result['is_cyberbullying'] else "✅"
    pred = result['prediction']
    conf = result['confidence']
    print(f"   {icon} \"{text[:40]}...\"")
    print(f"      → {pred} (Confidence: {conf:.2%})")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Model setup is correct!")
print("=" * 60)
