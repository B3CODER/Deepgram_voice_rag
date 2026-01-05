"""
Setup Milvus Collection for PDF RAG
====================================
Creates a collection in TEST_VOICE database with proper schema for PDF storage.
"""

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import os
from dotenv import load_dotenv

load_dotenv()

# Milvus Configuration
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_DB = "TEST_VOICE"  # New database
COLLECTION_NAME = "pdf_knowledge_base"  # New collection name

def create_collection():
    """Create Milvus collection with schema for PDF RAG."""
    
    print("\n" + "="*60)
    print("🗄️  MILVUS COLLECTION SETUP")
    print("="*60)
    
    # Connect to Milvus
    print(f"\n📡 Connecting to Milvus...")
    print(f"   Host: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"   Database: {MILVUS_DB}")
    
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        db_name=MILVUS_DB
    )
    print("   ✅ Connected!")
    
    # Check if collection already exists
    from pymilvus import utility
    if utility.has_collection(COLLECTION_NAME):
        print(f"\n⚠️  Collection '{COLLECTION_NAME}' already exists!")
        choice = input("   Do you want to drop and recreate it? (yes/no): ").strip().lower()
        if choice == 'yes':
            Collection(COLLECTION_NAME).drop()
            print("   ✅ Dropped existing collection")
        else:
            print("   ℹ️  Using existing collection")
            connections.disconnect("default")
            return
    
    # Define schema
    print(f"\n📋 Creating collection schema...")
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="upload_timestamp", dtype=DataType.VARCHAR, max_length=100),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="PDF Knowledge Base for Voice RAG"
    )
    
    print("   Schema fields:")
    for field in fields:
        print(f"   - {field.name}: {field.dtype}")
    
    # Create collection
    print(f"\n🔨 Creating collection '{COLLECTION_NAME}'...")
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using='default'
    )
    print("   ✅ Collection created!")
    
    # Create index for vector field
    print(f"\n🔍 Creating vector index...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    collection.create_index(
        field_name="vector",
        index_params=index_params
    )
    print("   ✅ Index created!")
    
    # Load collection
    print(f"\n📥 Loading collection into memory...")
    collection.load()
    print("   ✅ Collection loaded!")
    
    # Display summary
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE")
    print("="*60)
    print(f"Database: {MILVUS_DB}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vector Dimension: 1024")
    print(f"Index Type: IVF_FLAT")
    print(f"Metric: L2")
    print("="*60 + "\n")
    
    # Disconnect
    connections.disconnect("default")
    
    print("✅ You can now use this collection in voice_pipeline.py")
    print(f"   Update MILVUS_DB to '{MILVUS_DB}'")
    print(f"   Update MILVUS_COLLECTION to '{COLLECTION_NAME}'")
    print()


if __name__ == "__main__":
    try:
        create_collection()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
