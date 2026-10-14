import os
import sys

# Set before src.ai_server is imported: load_dotenv() does not override existing
# variables, so tests never touch the real Supabase/Cloudinary or the model file.
os.environ.update({
    "AI_SERVICE_TOKEN": "test-token",
    "SUPABASE_URL": "",
    "SUPABASE_SECRET_KEY": "",
    "SUPABASE_SERVICE_ROLE_KEY": "",
    "CLOUDINARY_CLOUD_NAME": "",
    "CLOUDINARY_API_KEY": "",
    "CLOUDINARY_API_SECRET": "",
    "DETECTOR": "yolo",
    "MODEL_PATH": "tests/no-model-in-tests.onnx",
    "MODEL_VERSION": "test-model",
    "COLLECT_MODE": "false",
})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
