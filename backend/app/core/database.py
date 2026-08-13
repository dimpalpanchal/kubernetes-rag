import urllib.parse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def create_database_if_not_exists():
    url = settings.DATABASE_URL
    prefix = ""
    for p in ["postgresql+asyncpg://", "postgresql://"]:
        if url.startswith(p):
            prefix = p
            url = url[len(p):]
            break
            
    if prefix and "@" in url:
        creds, host_part = url.rsplit("@", 1)
        if "/" in host_part:
            host_and_port, dbname = host_part.split("/", 1)
            if "?" in dbname:
                dbname = dbname.split("?", 1)[0]
        else:
            host_and_port = host_part
            dbname = ""
        
        if dbname:
            postgres_url = f"{prefix}{creds}@{host_and_port}/postgres"
            temp_engine = create_async_engine(postgres_url, isolation_level="AUTOCOMMIT")
            async with temp_engine.connect() as conn:
                result = await conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{dbname}'"))
                exists = result.scalar()
                if not exists:
                    print(f"Database '{dbname}' does not exist. Creating...")
                    await conn.execute(text(f"CREATE DATABASE {dbname}"))
                    print(f"Database '{dbname}' created successfully.")
                else:
                    print(f"Database '{dbname}' already exists.")
            await temp_engine.dispose()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
