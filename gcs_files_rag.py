import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/Users/bhawna/Desktop/agentic_ai_day/agenticai/agenticaigcp-b6166246b712.json"
from google.cloud import storage
client = storage.Client()
# from google.auth import default
# creds, project = default()
# print("✅ Authenticated project:", project)
project_id = "agenticaigcp"  # your actual project_id
bucket_name = "test-bucket-bhawna-unique-123456"  # make sure this is globally unique!

# Create bucket
try:
    bucket = client.create_bucket(bucket_name, location="us-central1", project=project_id)
    print(f"✅ Bucket created: {bucket.name}")
except Exception as e:
    print("❌ Error:", e)

