import os
import logging
import asyncio
import aiofiles
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import hashlib
import mimetypes
from pathlib import Path
import json
import yaml
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import time

# Document processing imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document as DocxDocument
import PyPDF2
import pandas as pd

# ML and vector DB imports
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Batch
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm.asyncio import tqdm

class ConfigManager:
    """Configuration management with validation"""
    
    DEFAULT_CONFIG = {
        'qdrant': {
            'host': 'localhost',
            'port': 6333,
            'prefer_grpc': False,
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 1.0
        },
        'model': {
            'name': 'all-MiniLM-L6-v2',
            'device': 'cpu',
            'cache_folder': './models',
            'max_seq_length': 512
        },
        'processing': {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'batch_size': 32,
            'max_file_size_mb': 100,
            'supported_extensions': ['.txt', '.md', '.csv', '.json', '.py', '.pdf', '.docx'],
            'max_workers': 4,
            'memory_limit_mb': 1024
        },
        'search': {
            'default_limit': 5,
            'max_limit': 50,
            'cache_size': 1000,
            'cache_ttl': 3600
        },
        'logging': {
            'level': 'INFO',
            'file': 'qdrant_operations.log',
            'max_size_mb': 10,
            'backup_count': 5,
            'console_output': True
        },
        'security': {
            'max_payload_size_mb': 50,
            'allowed_mime_types': [
                'text/plain', 'text/markdown', 'text/csv',
                'application/json', 'text/x-python',
                'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ],
            'sanitize_content': True
        }
    }
    
    @classmethod
    def load_config(cls, config_path: str = "config.yaml") -> Dict:
        """Load and validate configuration"""
        config = cls.DEFAULT_CONFIG.copy()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                config = cls._deep_merge(config, user_config)
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}")
        
        # Validate configuration
        cls._validate_config(config)
        return config
    
    @classmethod
    def _deep_merge(cls, base: Dict, override: Dict) -> Dict:
        """Deep merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = cls._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    @classmethod
    def _validate_config(cls, config: Dict):
        """Validate configuration values"""
        if config['processing']['chunk_size'] <= 0:
            raise ValueError("chunk_size must be positive")
        if config['processing']['chunk_overlap'] >= config['processing']['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if config['processing']['max_file_size_mb'] <= 0:
            raise ValueError("max_file_size_mb must be positive")

class SecurityValidator:
    """Security validation and sanitization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_payload_size = config['security']['max_payload_size_mb'] * 1024 * 1024
        self.allowed_mime_types = set(config['security']['allowed_mime_types'])
    
    def validate_file(self, file_path: str) -> tuple[bool, str]:
        """Validate file for security concerns"""
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_payload_size:
                return False, f"File too large: {file_size} bytes"
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type not in self.allowed_mime_types:
                return False, f"Unsupported MIME type: {mime_type}"
            
            # Check for directory traversal
            normalized_path = os.path.normpath(file_path)
            if '..' in normalized_path or normalized_path.startswith('/'):
                return False, "Invalid file path detected"
            
            return True, "Valid"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def sanitize_content(self, content: str) -> str:
        """Sanitize text content"""
        if not self.config['security']['sanitize_content']:
            return content
        
        # Remove potential script tags and suspicious content
        import re
        content = re.sub(r'<script.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
        content = re.sub(r'on\w+\s*=', '', content, flags=re.IGNORECASE)
        
        return content

class DocumentProcessor:
    """Enhanced document processing with multiple file type support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.supported_extensions = set(config['processing']['supported_extensions'])
        self.max_file_size = config['processing']['max_file_size_mb'] * 1024 * 1024
        self.security_validator = SecurityValidator(config)
    
    async def process_file(self, file_path: str) -> Optional[Dict]:
        """Process a single file with async support"""
        try:
            # Security validation
            is_valid, message = self.security_validator.validate_file(file_path)
            if not is_valid:
                logging.warning(f"Security check failed for {file_path}: {message}")
                return None
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_extensions:
                logging.warning(f"Unsupported file type: {file_ext}")
                return None
            
            # Extract text based on file type
            if file_ext == '.pdf':
                text = await self._extract_pdf_text(file_path)
            elif file_ext == '.docx':
                text = await self._extract_docx_text(file_path)
            elif file_ext == '.json':
                text = await self._extract_json_text(file_path)
            elif file_ext == '.csv':
                text = await self._extract_csv_text(file_path)
            else:
                text = await self._extract_text_file(file_path)
            
            if not text or not text.strip():
                logging.warning(f"No text extracted from {file_path}")
                return None
            
            # Sanitize content
            text = self.security_validator.sanitize_content(text)
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['processing']['chunk_size'],
                chunk_overlap=self.config['processing']['chunk_overlap']
            )
            chunks = splitter.split_text(text)
            
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'file_type': file_ext,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'processed_at': datetime.now().isoformat(),
                'content_hash': hashlib.md5(text.encode()).hexdigest()
            }
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return None
    
    async def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text files with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    return await f.read()
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError(f"Could not decode {file_path} with any supported encoding")
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF files"""
        def extract():
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract)
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        def extract():
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract)
    
    async def _extract_json_text(self, file_path: str) -> str:
        """Extract text from JSON files"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)
            return json.dumps(data, indent=2)
    
    async def _extract_csv_text(self, file_path: str) -> str:
        """Extract text from CSV files"""
        def extract():
            df = pd.read_csv(file_path)
            return df.to_string()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract)

class HealthMonitor:
    """System health monitoring"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.processed_files = 0
        self.failed_files = 0
        self.total_chunks = 0
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'uptime': str(datetime.now() - self.start_time),
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'processing_stats': {
                'processed_files': self.processed_files,
                'failed_files': self.failed_files,
                'total_chunks': self.total_chunks,
                'success_rate': self.processed_files / (self.processed_files + self.failed_files) if (self.processed_files + self.failed_files) > 0 else 0
            }
        }
    
    def update_stats(self, processed: bool, chunk_count: int = 0):
        """Update processing statistics"""
        if processed:
            self.processed_files += 1
            self.total_chunks += chunk_count
        else:
            self.failed_files += 1

class QdrantDataInjector:
    """Enhanced QDrant data injection with comprehensive features"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager.load_config(config_path)
        self._setup_logging()
        self.health_monitor = HealthMonitor()
        self.document_processor = DocumentProcessor(self.config)
        
        # Initialize components
        self.model = None
        self.client = None
        self._embedding_cache = {}
    
    async def initialize(self):
        """Initialize async components"""
        try:
            # Initialize model
            self.logger.info(f"Loading embedding model: {self.config['model']['name']}")
            self.model = SentenceTransformer(
                self.config['model']['name'],
                device=self.config['model']['device'],
                cache_folder=self.config['model'].get('cache_folder')
            )
            
            # Initialize Qdrant client with retry logic
            await self._initialize_client()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_logging(self):
        """Enhanced logging setup"""
        log_config = self.config['logging']
        
        handlers = []
        
        # File handler
        if log_config.get('file'):
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config['file'],
                maxBytes=log_config.get('max_size_mb', 10) * 1024 * 1024,
                backupCount=log_config.get('backup_count', 5)
            )
            handlers.append(file_handler)
        
        # Console handler
        if log_config.get('console_output', True):
            handlers.append(logging.StreamHandler())
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def _initialize_client(self):
        """Initialize Qdrant client with connection pooling and retry logic"""
        qdrant_config = self.config['qdrant']
        max_retries = qdrant_config.get('max_retries', 3)
        retry_delay = qdrant_config.get('retry_delay', 1.0)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Connecting to Qdrant at {qdrant_config['host']}:{qdrant_config['port']} (attempt {attempt + 1})")
                
                self.client = QdrantClient(
                    host=qdrant_config['host'],
                    port=qdrant_config['port'],
                    prefer_grpc=qdrant_config.get('prefer_grpc', False),
                    timeout=qdrant_config.get('timeout', 30)
                )
                
                # Test connection
                await asyncio.get_event_loop().run_in_executor(None, self.client.get_collections)
                self.logger.info("Successfully connected to Qdrant")
                return
                
            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'components': {},
            'system': self.health_monitor.get_system_health()
        }
        
        try:
            # Check Qdrant connection
            collections = await asyncio.get_event_loop().run_in_executor(None, self.client.get_collections)
            status['components']['qdrant'] = {
                'status': 'healthy',
                'collections': len(collections.collections),
                'total_vectors': sum(col.vectors_count for col in collections.collections)
            }
        except Exception as e:
            status['components']['qdrant'] = {'status': 'unhealthy', 'error': str(e)}
            status['status'] = 'degraded'
        
        try:
            # Check model
            test_embedding = self.model.encode(["health check test"])
            status['components']['model'] = {
                'status': 'healthy',
                'embedding_dim': len(test_embedding[0])
            }
        except Exception as e:
            status['components']['model'] = {'status': 'unhealthy', 'error': str(e)}
            status['status'] = 'degraded'
        
        return status
    
    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text_hash: str, text: str) -> List[float]:
        """Cache embeddings to avoid recomputation"""
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        self._embedding_cache[text_hash] = embedding
        return embedding
    
    async def _generate_embeddings_batch(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings in batches with caching"""
        def generate_batch(chunk_batch):
            return self.model.encode(
                chunk_batch,
                batch_size=self.config['processing']['batch_size'],
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
        
        # Process in batches to manage memory
        batch_size = self.config['processing']['batch_size']
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, generate_batch, batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    async def _create_collection(self, collection_name: str) -> bool:
        """Create collection with proper error handling"""
        try:
            def create_collection_sync():
                if not self.client.collection_exists(collection_name):
                    self.logger.info(f"Creating new collection: {collection_name}")
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=self.model.get_sentence_embedding_dimension(),
                            distance=Distance.COSINE
                        ),
                        timeout=60
                    )
                    return True
                return False
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, create_collection_sync)
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    async def _upload_to_qdrant(self, collection_name: str, chunks: List[str], 
                               embeddings: List[List[float]], metadata: Dict) -> bool:
        """Upload data with UUID-based IDs and batch processing"""
        try:
            points = []
            base_uuid = uuid.uuid4()
            
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate unique UUID for each point
                point_id = str(uuid.UUID(int=base_uuid.int + idx))
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'text': chunk,
                        'source_file': metadata['file_name'],
                        'file_path': metadata['file_path'],
                        'file_type': metadata.get('file_type', ''),
                        'chunk_index': idx,
                        'total_chunks': metadata['chunk_count'],
                        'content_hash': metadata.get('content_hash', ''),
                        'file_size': metadata['file_size'],
                        'created_at': metadata['processed_at'],
                        'chunk_hash': hashlib.md5(chunk.encode()).hexdigest()
                    }
                ))
            
            # Upload in batches
            upload_batch_size = self.config['processing'].get('upload_batch_size', 100)
            
            def upload_batch(batch_points):
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                    wait=True
                )
            
            for i in range(0, len(points), upload_batch_size):
                batch = points[i:i + upload_batch_size]
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, upload_batch, batch)
            
            self.logger.info(f"Successfully uploaded {len(points)} points to {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error uploading to Qdrant: {e}")
            return False
    
    async def process_directory_async(self, input_dir: str, collection_prefix: str = None) -> Dict[str, Any]:
        """Async directory processing with comprehensive error handling"""
        if not os.path.isdir(input_dir):
            self.logger.error(f"Directory not found: {input_dir}")
            return {'success': False, 'error': 'Directory not found'}
        
        # Get all supported files
        all_files = []
        for ext in self.config['processing']['supported_extensions']:
            pattern = f"*{ext}"
            all_files.extend(Path(input_dir).glob(pattern))
        
        if not all_files:
            self.logger.error(f"No supported files found in directory: {input_dir}")
            return {'success': False, 'error': 'No supported files found'}
        
        self.logger.info(f"Found {len(all_files)} files to process in {input_dir}")
        
        results = {
            'success': True,
            'total_files': len(all_files),
            'processed_files': 0,
            'failed_files': 0,
            'collections_created': [],
            'total_chunks': 0,
            'processing_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        # Process files concurrently with semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.config['processing'].get('max_workers', 4))
        
        async def process_single_file(file_path):
            async with semaphore:
                try:
                    # Process file
                    file_result = await self.document_processor.process_file(str(file_path))
                    if not file_result:
                        results['failed_files'] += 1
                        results['errors'].append(f"Failed to process {file_path.name}")
                        self.health_monitor.update_stats(False)
                        return
                    
                    # Determine collection name
                    if collection_prefix:
                        collection_name = f"{collection_prefix}_{file_path.stem}"
                    else:
                        collection_name = file_path.stem
                    
                    # Clean collection name
                    collection_name = "".join(c for c in collection_name if c.isalnum() or c in '_-').lower()
                    
                    # Create collection and upload
                    await self._create_collection(collection_name)
                    if collection_name not in results['collections_created']:
                        results['collections_created'].append(collection_name)
                    
                    # Generate embeddings and upload
                    embeddings = await self._generate_embeddings_batch(file_result['chunks'])
                    
                    if await self._upload_to_qdrant(collection_name, file_result['chunks'], embeddings, file_result):
                        results['processed_files'] += 1
                        results['total_chunks'] += len(file_result['chunks'])
                        self.health_monitor.update_stats(True, len(file_result['chunks']))
                    else:
                        results['failed_files'] += 1
                        results['errors'].append(f"Failed to upload {file_path.name}")
                        self.health_monitor.update_stats(False)
                
                except Exception as e:
                    results['failed_files'] += 1
                    error_msg = f"Error processing {file_path.name}: {str(e)}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    self.health_monitor.update_stats(False)
        
        # Process all files concurrently
        tasks = [process_single_file(file_path) for file_path in all_files]
        await tqdm.gather(*tasks, desc="Processing files")
        
        results['processing_time'] = time.time() - start_time
        
        self.logger.info(f"Processing complete. Successfully processed {results['processed_files']}/{results['total_files']} files")
        self.logger.info(f"Total processing time: {results['processing_time']:.2f} seconds")
        
        return results
    
    def process_directory(self, input_dir: str, collection_prefix: str = None) -> Dict[str, Any]:
        """Sync wrapper for async directory processing"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.process_directory_async(input_dir, collection_prefix))

async def main():
    """Enhanced main function with comprehensive error handling"""
    try:
        # Initialize injector
        injector = QdrantDataInjector()
        await injector.initialize()  # Explicit initialization
        
        # Perform health check
        health = await injector.health_check()
        print(f"System Health: {health['status']}")
        
        if health['status'] != 'healthy':
            print("Warning: System is not fully healthy. Proceeding with caution...")
            print(json.dumps(health, indent=2))
        
        # Get input directory
        input_dir = injector.config.get('input_dir')
        if not input_dir:
            input_dir = input("Enter the path to the directory containing files: ").strip()
        
        if not input_dir:
            print("No input directory provided. Exiting.")
            return
        
        # Process directory
        results = await injector.process_directory_async(input_dir, collection_prefix="documents")
        
        # Display results
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        print(f"Total files: {results['total_files']}")
        print(f"Successfully processed: {results['processed_files']}")
        print(f"Failed: {results['failed_files']}")
        print(f"Total chunks created: {results['total_chunks']}")
        print(f"Collections created: {len(results['collections_created'])}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
        
        # Final health check
        final_health = await injector.health_check()
        print(f"\nFinal system status: {final_health['status']}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}")
        print(f"Fatal error: {e}")
    finally:
        print("Data injection process completed")

if __name__ == "__main__":
    asyncio.run(main())