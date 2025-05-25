from transformers import AutoModelForTokenClassification

try:
    AutoModelForTokenClassification.from_pretrained('LogSensitiveResearcher/SDLog_main')
    print("Model SDLog_main loaded successfully.")
except Exception as e:
    print(f"Error loading model SDLog_main: {e}")

try:
    AutoModelForTokenClassification.from_pretrained('LogSensitiveResearcher/SDLog_net')
    print("Model SDLog_net loaded successfully.")
except Exception as e:
    print(f"Error loading model SDLog_net: {e}")