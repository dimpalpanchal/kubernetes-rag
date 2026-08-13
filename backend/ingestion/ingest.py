import os
import sys
import asyncio
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Add backend directory to sys.path to allow importing app modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.models.document import DocumentChunk
from app.core.database import Base, create_database_if_not_exists

# Load environment variables explicitly for the script
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

TARGET_DIR = str(Path(__file__).resolve().parent.parent.parent / "data")

async def init_db(engine):
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("TRUNCATE TABLE document_chunks;"))

async def main():
    print("Initializing database...")
    await create_database_if_not_exists()
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    await init_db(engine)
    
    AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    if not os.path.exists(TARGET_DIR):
        raise FileNotFoundError(
            f"Kubernetes docs directory not found at: {TARGET_DIR}. "
            "Please ensure you have placed your markdown files there."
        )

    print(f"Loading markdown files from {TARGET_DIR}...")
    docs = []
    skipped_count = 0
    all_files = list(Path(TARGET_DIR).rglob("*.md"))
    for file_path in all_files:
        filename = file_path.name.lower()
        is_changelog = "changelog" in filename
        is_version = any(filename.startswith(prefix) for prefix in ["v1.", "v2.", "v3.", "v4.", "v5."])
        is_template = any(token in filename for token in ["bug-report", "pullrequesttemplate", "code-of-conduct", "contributing", "license", "security-policy"])
        is_large = file_path.stat().st_size > 40 * 1024
        
        if is_changelog or is_version or is_template or is_large:
            skipped_count += 1
            continue
            
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    print(f"Loaded {len(docs)} documents (skipped {skipped_count} large/changelog/version files).")

    print("Splitting documents using MarkdownHeaderTextSplitter...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    header_splits = []
    for doc in docs:
        doc_splits = header_splitter.split_text(doc.page_content)
        for split in doc_splits:
            # Preserve original metadata and combine with header metadata
            metadata = doc.metadata.copy()
            metadata.update(split.metadata)
            
            # Generate a human-readable chunk title from the hierarchy
            headers = []
            for lvl in ["Header 1", "Header 2", "Header 3", "Header 4"]:
                if lvl in split.metadata:
                    headers.append(split.metadata[lvl])
            metadata["chunk_title"] = " > ".join(headers) if headers else "Untitled Chunk"
            
            split.metadata = metadata
            header_splits.append(split)
            
    print("Splitting sub-chunks using MarkdownTextSplitter...")
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(header_splits)
    print(f"Created {len(splits)} chunks from {len(docs)} documents.")

    print("Deduplicating chunks by content hash...")
    import hashlib
    unique_splits = []
    seen_hashes = set()
    for split in splits:
        content_hash = hashlib.sha256(split.page_content.strip().lower().encode("utf-8")).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_splits.append(split)
    print(f"Retained {len(unique_splits)} unique chunks (removed {len(splits) - len(unique_splits)} exact duplicate chunks).")
    splits = unique_splits

    print("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

    print("Embedding and saving to database (this may take a while)...")
    
    BATCH_SIZE = 100  # Configurable batch size
    
    async with AsyncSessionLocal() as session:
        for i in range(0, len(splits), BATCH_SIZE):
            batch = splits[i:i + BATCH_SIZE]
            augmented_contents = []
            
            for split in batch:
                content = split.page_content
                metadata = split.metadata
                
                # Prepend the extracted header hierarchy (chunk_title) to the content sent for embedding
                chunk_title = metadata.get("chunk_title", "")
                if chunk_title and chunk_title != "Untitled Chunk":
                    augmented_content = f"{chunk_title}\n\n{content}"
                else:
                    augmented_content = content
                augmented_contents.append(augmented_content)
                
            try:
                # Generate embeddings for the batch
                batch_embeddings = await embeddings.aembed_documents(augmented_contents)
                
                for j, split in enumerate(batch):
                    chunk = DocumentChunk(
                        content=split.page_content,
                        metadata_=split.metadata,
                        embedding=batch_embeddings[j]
                    )
                    session.add(chunk)
                    
                await session.commit()
                print(f"Inserted {min(i + BATCH_SIZE, len(splits))}/{len(splits)} chunks.")
                
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                await session.rollback()

if __name__ == "__main__":
    asyncio.run(main())
