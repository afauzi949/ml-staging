import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

qdrant_host = os.getenv('QDRANT_HOST', 'qdrant.staytech.xyz')
qdrant_port = int(os.getenv('QDRANT_PORT', '443'))
qdrant_https = os.getenv('QDRANT_HTTPS', 'true').lower() == 'true'
qdrant_api_key = os.getenv('QDRANT_API_KEY')

client = QdrantClient(
    host=qdrant_host,
    port=qdrant_port,
    https=qdrant_https,
    api_key=qdrant_api_key
)

print(client.get_collections())