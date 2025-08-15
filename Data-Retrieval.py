import os
import logging
import asyncio
import time
import json
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from functools import lru_cache
import yaml
import re

# ML and vector DB imports
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, MatchValue, Range, GeoBoundingBox
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
import spacy
from spacy.cli import download
import psutil

class SearchCache:
    """In-memory search cache with TTL"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_key(self, query: str, collection_name: str, search_type: str, **kwargs) -> str:
        """Generate cache key"""
        key_data = {
            'query': query.lower().strip(),
            'collection': collection_name,
            'type': search_type,
            **kwargs
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result if not expired"""
        if key in self.cache:
            # Check TTL
            if time.time() - self.access_times[key] < self.ttl_seconds:
                return self.cache[key]
            else:
                # Expired, remove
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def put(self, key: str, value: Any):
        """Store result in cache"""
        # Clean up if at max size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_request_count', 1), 1)
        }

class SearchAnalytics:
    """Search analytics and performance tracking"""
    
    def __init__(self):
        self.search_history = []
        self.performance_metrics = defaultdict(list)
        self.query_patterns = Counter()
        self.error_counts = Counter()
    
    def log_search(self, query: str, collection: str, search_type: str, 
                  results_count: int, response_time: float, success: bool = True, error: str = None):
        """Log search operation"""
        entry = {
            'timestamp': datetime.now(),
            'query': query.lower().strip(),
            'collection': collection,
            'search_type': search_type,
            'results_count': results_count,
            'response_time': response_time,
            'success': success,
            'error': error
        }
        
        self.search_history.append(entry)
        self.performance_metrics[search_type].append(response_time)
        
        # Track query patterns
        query_words = len(query.split())
        self.query_patterns[f"{search_type}_{query_words}_words"] += 1
        
        if not success and error:
            self.error_counts[error] += 1
        
        # Keep only last 10000 entries to prevent memory issues
        if len(self.search_history) > 10000:
            self.search_history = self.search_history[-5000:]
    
    def get_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get analytics for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_searches = [s for s in self.search_history if s['timestamp'] >= cutoff_time]
        
        if not recent_searches:
            return {'message': 'No searches in specified time period'}
        
        # Calculate metrics
        total_searches = len(recent_searches)
        successful_searches = len([s for s in recent_searches if s['success']])
        avg_response_time = sum(s['response_time'] for s in recent_searches) / total_searches
        
        # Popular queries
        query_counter = Counter(s['query'] for s in recent_searches)
        popular_queries = query_counter.most_common(10)
        
        # Collection usage
        collection_usage = Counter(s['collection'] for s in recent_searches)
        
        # Search type distribution
        search_type_dist = Counter(s['search_type'] for s in recent_searches)
        
        # Performance by search type
        perf_by_type = {}
        for search_type in search_type_dist.keys():
            type_searches = [s for s in recent_searches if s['search_type'] == search_type]
            perf_by_type[search_type] = {
                'count': len(type_searches),
                'avg_response_time': sum(s['response_time'] for s in type_searches) / len(type_searches),
                'avg_results': sum(s['results_count'] for s in type_searches) / len(type_searches)
            }
        
        return {
            'period_hours': hours,
            'total_searches': total_searches,
            'success_rate': successful_searches / total_searches,
            'avg_response_time': avg_response_time,
            'popular_queries': popular_queries,
            'collection_usage': dict(collection_usage),
            'search_type_distribution': dict(search_type_dist),
            'performance_by_type': perf_by_type,
            'top_errors': dict(self.error_counts.most_common(5)) if self.error_counts else {}
        }

class SecurityValidator:
    """Enhanced security validation for search queries"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'\bUNION\b.*\bSELECT\b',
            r'\bDROP\b.*\bTABLE\b',
            r'\bINSERT\b.*\bINTO\b'
        ]
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate search query for security concerns"""
        if not query or not query.strip():
            return False, "Empty query"
        
        if len(query) > 1000:
            return False, "Query too long"
        
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Potentially malicious pattern detected"
        
        return True, "Valid"
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize query string"""
        # Remove HTML tags
        query = re.sub(r'<[^>]+>', '', query)
        # Remove potentially dangerous characters
        query = re.sub(r'[<>"\';\\]', '', query)
        # Normalize whitespace
        query = ' '.join(query.split())
        return query.strip()

class EnhancedNLPProcessor:
    """Enhanced NLP processing with better entity recognition"""
    
    def __init__(self):
        self._load_spacy_model()
        
    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy language model...")
            try:
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"Failed to download spaCy model: {e}")
                self.nlp = None
    
    def extract_search_terms_advanced(self, query: str) -> Dict[str, List[str]]:
        """Enhanced term extraction with entity categorization"""
        if not self.nlp:
            # Fallback to simple word extraction
            words = query.lower().split()
            return {
                'keywords': [w for w in words if len(w) > 2],
                'entities': [],
                'dates': [],
                'numbers': [],
                'locations': []
            }
        
        doc = self.nlp(query.lower())
        
        result = {
            'keywords': [],
            'entities': [],
            'dates': [],
            'numbers': [],
            'locations': []
        }
        
        # Extract keywords (nouns, adjectives, proper nouns)
        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.lemma_) > 2):
                result['keywords'].append(token.lemma_)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "PRODUCT", "EVENT"]:
                result['entities'].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                result['locations'].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                result['dates'].append(ent.text)
            elif ent.label_ in ["MONEY", "PERCENT", "QUANTITY", "CARDINAL"]:
                result['numbers'].append(ent.text)
        
        # Remove duplicates while preserving order
        for key in result:
            seen = set()
            result[key] = [x for x in result[key] if not (x in seen or seen.add(x))]
        
        return result
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and complexity"""
        if not self.nlp:
            return {'complexity': 'simple', 'intent': 'search', 'confidence': 0.5}
        
        doc = self.nlp(query.lower())
        
        # Question words
        question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
        has_question = any(token.text in question_words for token in doc)
        
        # Action verbs
        action_verbs = {'find', 'show', 'list', 'get', 'search', 'look', 'tell'}
        has_action = any(token.lemma_ in action_verbs for token in doc)
        
        # Determine intent
        intent = 'question' if has_question else 'search'
        if has_action:
            intent = 'action'
        
        # Complexity based on sentence structure
        complexity = 'simple'
        if len(doc) > 10:
            complexity = 'complex'
        elif len([token for token in doc if token.pos_ in ["NOUN", "PROPN"]]) > 3:
            complexity = 'moderate'
        
        return {
            'intent': intent,
            'complexity': complexity,
            'confidence': 0.8 if has_question or has_action else 0.6,
            'has_entities': len(list(doc.ents)) > 0,
            'word_count': len(doc)
        }

class QdrantExplorer:
    """Enhanced QDrant explorer with comprehensive search capabilities"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Initialize components
        self.client = self._initialize_client()
        self.embedding_model = self._initialize_model()
        self.nlp_processor = EnhancedNLPProcessor()
        self.security_validator = SecurityValidator(self.config)
        self.search_cache = SearchCache(
            max_size=self.config.get('search', {}).get('cache_size', 1000),
            ttl_seconds=self.config.get('search', {}).get('cache_ttl', 3600)
        )
        self.analytics = SearchAnalytics()
        
        # Health monitoring
        self.start_time = datetime.now()
        self.total_searches = 0
        self.cache_hits = 0
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            'qdrant': {
                'host': 'localhost',
                'port': 6333,
                'prefer_grpc': False,
                'timeout': 30,
                'max_retries': 3
            },
            'model': {
                'name': 'all-MiniLM-L6-v2',
                'device': 'cpu'
            },
            'search': {
                'default_limit': 5,
                'max_limit': 50,
                'cache_size': 1000,
                'cache_ttl': 3600,
                'default_weights': {
                    'semantic': 0.7,
                    'keyword': 0.3
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'qdrant_search.log'
            },
            'security': {
                'max_query_length': 1000,
                'sanitize_queries': True
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                # Merge configs
                for key, value in user_config.items():
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = self.config.get('logging', {}).get('file', 'qdrant_search.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client with retry logic"""
        qdrant_config = self.config['qdrant']
        max_retries = qdrant_config.get('max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                client = QdrantClient(
                    host=qdrant_config['host'],
                    port=qdrant_config['port'],
                    prefer_grpc=qdrant_config.get('prefer_grpc', False),
                    timeout=qdrant_config.get('timeout', 30)
                )
                
                # Test connection
                client.get_collections()
                self.logger.info(f"Connected to Qdrant at {qdrant_config['host']}:{qdrant_config['port']}")
                return client
                
            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise ConnectionError(f"Failed to connect to Qdrant after {max_retries} attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _initialize_model(self) -> SentenceTransformer:
        """Initialize embedding model"""
        try:
            model_config = self.config['model']
            self.logger.info(f"Loading embedding model: {model_config['name']}")
            return SentenceTransformer(
                model_config['name'],
                device=model_config.get('device', 'cpu')
            )
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - self.start_time),
            'status': 'healthy',
            'components': {}
        }
        
        try:
            # Check Qdrant
            collections = self.client.get_collections()
            total_vectors = 0
            
            # Updated collection info retrieval
            for col in collections.collections:
                try:
                    info = self.client.get_collection(col.name)
                    total_vectors += info.points_count  # Using points_count instead of vectors_count
                except Exception as e:
                    self.logger.warning(f"Couldn't get count for collection {col.name}: {str(e)}")
                    # Fallback to config size if count fails
                    if hasattr(col.config.params, 'vectors'):
                        total_vectors += col.config.params.vectors.size
            
            health_status['components']['qdrant'] = {
                'status': 'healthy',
                'collections_count': len(collections.collections),
                'total_vectors': total_vectors
            }
        except Exception as e:
            health_status['components']['qdrant'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['status'] = 'degraded'
        
        try:
            # Check embedding model
            test_embedding = self.embedding_model.encode(["health check"])
            health_status['components']['embedding_model'] = {
                'status': 'healthy',
                'dimension': len(test_embedding[0])
            }
        except Exception as e:
            health_status['components']['embedding_model'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['status'] = 'degraded'
        
        # System resources
        memory = psutil.virtual_memory()
        health_status['system'] = {
            'memory_percent': memory.percent,
            'total_searches': self.total_searches,
            'cache_hits': self.cache_hits,
            'cache_stats': self.search_cache.stats()
        }
        
        return health_status
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """List all available collections with enhanced info"""
        try:
            collections = self.client.get_collections()
            result = []
            
            for col in collections.collections:
                # Get collection info
                info = self.client.get_collection(col.name)
                result.append({
                    "name": col.name,
                    "vectors_count": info.points_count,  # Updated to use points_count
                    "config": {
                        "vector_size": info.config.params.vectors.size,
                        "distance": info.config.params.vectors.distance.value
                    },
                    "status": info.status
                })
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting collections: {e}")
            return []
    
    def semantic_search(self, query: str, collection_name: str, limit: int = 5, 
                       filters: Optional[Dict] = None) -> List[Dict]:
        """Enhanced semantic search with caching and filtering"""
        start_time = time.time()
        
        try:
            # Security validation
            is_valid, message = self.security_validator.validate_query(query)
            if not is_valid:
                self.logger.warning(f"Invalid query rejected: {message}")
                return []
            
            # Sanitize query
            if self.config.get('security', {}).get('sanitize_queries', True):
                query = self.security_validator.sanitize_query(query)
            
            # Check cache
            cache_key = self.search_cache._generate_key(query, collection_name, 'semantic', 
                                                      limit=limit, filters=filters)
            cached_result = self.search_cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                return cached_result
            
            # Generate embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Build filters
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(FieldCondition(key=key, match=MatchText(text=value)))
                    elif isinstance(value, (int, float)):
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                    elif isinstance(value, dict) and 'range' in value:
                        conditions.append(FieldCondition(
                            key=key, 
                            range=Range(
                                gte=value['range'].get('gte'),
                                lte=value['range'].get('lte')
                            )
                        ))
                
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=min(limit, self.config['search']['max_limit']),
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            formatted_results = [{
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in results]
            
            # Cache results
            self.search_cache.put(cache_key, formatted_results)
            
            # Log analytics
            response_time = time.time() - start_time
            self.analytics.log_search(query, collection_name, 'semantic', 
                                    len(formatted_results), response_time)
            self.total_searches += 1
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Semantic search error: {e}")
            self.analytics.log_search(query, collection_name, 'semantic', 
                                    0, time.time() - start_time, False, str(e))
            return []
    
    def keyword_search(self, query: str, collection_name: str, field: str = "text", 
                      limit: int = 5) -> List[Dict]:
        """Enhanced keyword search with proper error handling"""
        start_time = time.time()
        
        try:
            # Security validation
            is_valid, message = self.security_validator.validate_query(query)
            if not is_valid:
                self.logger.warning(f"Invalid query rejected: {message}")
                return []
            
            # Sanitize query
            if self.config.get('security', {}).get('sanitize_queries', True):
                query = self.security_validator.sanitize_query(query)
            
            # Check cache
            cache_key = self.search_cache._generate_key(query, collection_name, 'keyword', 
                                                      field=field, limit=limit)
            cached_result = self.search_cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                return cached_result
            
            # Fixed the syntax error here
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key=field,
                        match=MatchText(text=query)
                    )]
                ),
                limit=min(limit, self.config['search']['max_limit'])
            )
            
            # Format results
            formatted_results = [{
                "id": hit.id,
                "payload": hit.payload,
                "score": 1.0  # Keyword matches get perfect score
            } for hit in results[0]]
            
            # Cache results
            self.search_cache.put(cache_key, formatted_results)
            
            # Log analytics
            response_time = time.time() - start_time
            self.analytics.log_search(query, collection_name, 'keyword', 
                                    len(formatted_results), response_time)
            self.total_searches += 1
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Keyword search error: {e}")
            self.analytics.log_search(query, collection_name, 'keyword', 
                                    0, time.time() - start_time, False, str(e))
            return []
    
    def advanced_hybrid_search(self, query: str, collection_name: str, 
                             weights: Optional[Dict] = None, limit: int = 5,
                             filters: Optional[Dict] = None) -> List[Dict]:
        """Advanced hybrid search with configurable weights and analytics"""
        start_time = time.time()
        
        try:
            # Use default weights if not provided
            if not weights:
                weights = self.config['search']['default_weights']
            
            # Analyze query for better search strategy
            query_analysis = self.nlp_processor.analyze_query_intent(query)
            search_terms = self.nlp_processor.extract_search_terms_advanced(query)
            
            # Adjust weights based on query analysis
            if query_analysis['complexity'] == 'simple':
                weights = {'semantic': 0.5, 'keyword': 0.5}
            elif query_analysis['intent'] == 'question':
                weights = {'semantic': 0.8, 'keyword': 0.2}
            
            # Perform searches
            semantic_results = self.semantic_search(query, collection_name, limit * 2, filters)
            
            # Keyword search on extracted terms
            keyword_results = []
            for term_type, terms in search_terms.items():
                for term in terms[:3]:  # Limit terms to prevent too many searches
                    term_results = self.keyword_search(term, collection_name, limit=limit)
                    keyword_results.extend(term_results)
            
            # Combine and score results
            combined_scores = {}
            
            # Process semantic results
            if semantic_results:
                max_semantic_score = max(r.get('score', 0) for r in semantic_results)
                for result in semantic_results:
                    doc_id = result['id']
                    normalized_score = (result.get('score', 0) / max_semantic_score) if max_semantic_score > 0 else 0
                    combined_scores[doc_id] = {
                        'result': result,
                        'semantic_score': normalized_score,
                        'keyword_score': 0,
                        'total_score': normalized_score * weights['semantic']
                    }
            
            # Process keyword results
            for result in keyword_results:
                doc_id = result['id']
                keyword_score = 0.9  # High relevance for exact matches
                
                if doc_id in combined_scores:
                    combined_scores[doc_id]['keyword_score'] = keyword_score
                    combined_scores[doc_id]['total_score'] += keyword_score * weights['keyword']
                else:
                    combined_scores[doc_id] = {
                        'result': result,
                        'semantic_score': 0,
                        'keyword_score': keyword_score,
                        'total_score': keyword_score * weights['keyword']
                    }
            
            # Sort by total score and return top results
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['total_score'],
                reverse=True
            )[:limit]
            
            # Format final results
            final_results = []
            for item in sorted_results:
                result = item['result'].copy()
                result['hybrid_score'] = item['total_score']
                result['score_breakdown'] = {
                    'semantic': item['semantic_score'],
                    'keyword': item['keyword_score'],
                    'weights_used': weights
                }
                final_results.append(result)
            
            # Log analytics
            response_time = time.time() - start_time
            self.analytics.log_search(query, collection_name, 'hybrid', 
                                    len(final_results), response_time)
            self.total_searches += 1
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search error: {e}")
            self.analytics.log_search(query, collection_name, 'hybrid', 
                                    0, time.time() - start_time, False, str(e))
            return []
    
    def get_search_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive search analytics"""
        return self.analytics.get_analytics(hours)
    
    def clear_cache(self):
        """Clear search cache"""
        self.search_cache.clear()
        self.logger.info("Search cache cleared")

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_results(results: List[Dict], title: str = "Search Results"):
    """Enhanced results display with scoring breakdown"""
    if not results:
        print("\nNo results found.")
        return
    
    print(f"\n🔍 {title}:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. ID: {result['id']}")
        
        # Display scores
        if 'score' in result:
            print(f"   📊 Similarity Score: {result['score']:.3f}")
        
        if 'hybrid_score' in result:
            print(f"   🔀 Hybrid Score: {result['hybrid_score']:.3f}")
            if 'score_breakdown' in result:
                breakdown = result['score_breakdown']
                print(f"      - Semantic: {breakdown['semantic']:.3f}")
                print(f"      - Keyword: {breakdown['keyword']:.3f}")
        
        # Display payload
        if 'payload' in result and result['payload']:
            print("   📄 Content:")
            for key, value in result['payload'].items():
                if key == 'text':
                    # Truncate long text
                    text_preview = str(value)[:150] + "..." if len(str(value)) > 150 else str(value)
                    print(f"     - {key}: {text_preview}")
                elif key in ['source_file', 'file_type', 'created_at']:
                    print(f"     - {key}: {value}")

def display_analytics(analytics: Dict[str, Any]):
    """Display search analytics"""
    if 'message' in analytics:
        print(f"\n📊 Analytics: {analytics['message']}")
        return
    
    print("\n📊 Search Analytics")
    print("=" * 40)
    print(f"Period: Last {analytics['period_hours']} hours")
    print(f"Total Searches: {analytics['total_searches']}")
    print(f"Success Rate: {analytics['success_rate']:.1%}")
    print(f"Avg Response Time: {analytics['avg_response_time']:.3f}s")
    
    if analytics['popular_queries']:
        print(f"\n🔥 Popular Queries:")
        for query, count in analytics['popular_queries'][:5]:
            print(f"   - '{query}' ({count} times)")
    
    if analytics['performance_by_type']:
        print(f"\n⚡ Performance by Search Type:")
        for search_type, perf in analytics['performance_by_type'].items():
            print(f"   - {search_type}: {perf['avg_response_time']:.3f}s avg, {perf['avg_results']:.1f} results avg")

def main_menu(explorer: QdrantExplorer):
    """Enhanced interactive menu with analytics and monitoring"""
    while True:
        clear_screen()
        
        # Display system status
        health = explorer.health_check()
        status_icon = "🟢" if health['status'] == 'healthy' else "🟡" if health['status'] == 'degraded' else "🔴"
        
        print(f"""
        🔍 QDrant Vector Search Explorer {status_icon}
        ========================================
        System Status: {health['status'].upper()}
        Uptime: {health['uptime']}
        Total Searches: {health['system']['total_searches']}
        Cache Hits: {health['system']['cache_hits']}
        
        📋 MENU OPTIONS:
        ────────────────
        1.  📚 List Collections
        2.  🎯 Semantic Search
        3.  🔤 Keyword Search  
        4.  🔀 Advanced Hybrid Search
        5.  🎛️  Advanced Search with Filters
        6.  📊 View Search Analytics
        7.  🏥 System Health Check
        8.  🧹 Clear Search Cache
        9.  ⚙️  Configuration Info
        0.  🚪 Exit
        """)

        choice = input("Enter your choice (0-9): ").strip()

        if choice == "0":
            print("\n👋 Thank you for using QDrant Vector Search Explorer!")
            break

        elif choice == "1":
            clear_screen()
            print("\n📚 Available Collections:")
            print("=" * 50)
            
            collections = explorer.get_collections()
            if collections:
                for col in collections:
                    status_emoji = "✅" if col['status'] == 'green' else "⚠️"
                    print(f"{status_emoji} {col['name']}")
                    print(f"   └── Vectors: {col['vectors_count']:,}")
                    print(f"   └── Dimension: {col['config']['vector_size']}")
                    print(f"   └── Distance: {col['config']['distance']}")
                    print()
            else:
                print("❌ No collections found or connection error.")
            
            input("\nPress Enter to continue...")

        elif choice in ["2", "3", "4", "5"]:
            clear_screen()
            
            # Get collection name
            collections = explorer.get_collections()
            if not collections:
                print("❌ No collections available.")
                input("Press Enter to continue...")
                continue
            
            print("📚 Available Collections:")
            for i, col in enumerate(collections, 1):
                print(f"{i}. {col['name']} ({col['vectors_count']:,} vectors)")
            
            try:
                col_choice = input(f"\nSelect collection (1-{len(collections)}) or enter name: ").strip()
                if col_choice.isdigit():
                    col_idx = int(col_choice) - 1
                    if 0 <= col_idx < len(collections):
                        collection_name = collections[col_idx]['name']
                    else:
                        print("❌ Invalid selection.")
                        input("Press Enter to continue...")
                        continue
                else:
                    collection_name = col_choice
            except (ValueError, IndexError):
                print("❌ Invalid selection.")
                input("Press Enter to continue...")
                continue
            
            # Get search query
            query = input(f"\n🔍 Enter your search query: ").strip()
            if not query:
                print("❌ Query cannot be empty.")
                input("Press Enter to continue...")
                continue
            
            # Get limit
            try:
                limit_input = input("📊 Number of results (default 5): ").strip()
                limit = int(limit_input) if limit_input else 5
                limit = min(limit, 50)  # Cap at 50
            except ValueError:
                limit = 5
            
            print(f"\n🔍 Searching in '{collection_name}'...")
            start_time = time.time()
            
            # Perform search based on choice
            if choice == "2":  # Semantic Search
                results = explorer.semantic_search(query, collection_name, limit)
                display_results(results, "Semantic Search Results")
                
            elif choice == "3":  # Keyword Search
                field = input("🏷️  Field to search (default 'text'): ").strip() or "text"
                results = explorer.keyword_search(query, collection_name, field, limit)
                display_results(results, f"Keyword Search Results (field: {field})")
                
            elif choice == "4":  # Advanced Hybrid Search
                print("\n⚖️  Search Weights Configuration:")
                try:
                    semantic_weight = input("Semantic weight (0.0-1.0, default 0.7): ").strip()
                    semantic_weight = float(semantic_weight) if semantic_weight else 0.7
                    keyword_weight = 1.0 - semantic_weight
                    
                    weights = {'semantic': semantic_weight, 'keyword': keyword_weight}
                    print(f"Using weights: Semantic={semantic_weight:.1f}, Keyword={keyword_weight:.1f}")
                except ValueError:
                    weights = None
                    print("Using default weights")
                
                results = explorer.advanced_hybrid_search(query, collection_name, weights, limit)
                display_results(results, "Advanced Hybrid Search Results")
                
            elif choice == "5":  # Advanced Search with Filters
                print("\n🎛️  Filter Configuration (optional):")
                filters = {}
                
                while True:
                    filter_key = input("Filter field (or press Enter to skip): ").strip()
                    if not filter_key:
                        break
                    
                    filter_value = input(f"Filter value for '{filter_key}': ").strip()
                    if filter_value:
                        # Try to convert to number
                        try:
                            if '.' in filter_value:
                                filters[filter_key] = float(filter_value)
                            else:
                                filters[filter_key] = int(filter_value)
                        except ValueError:
                            filters[filter_key] = filter_value
                    
                    add_more = input("Add another filter? (y/N): ").strip().lower()
                    if add_more != 'y':
                        break
                
                results = explorer.semantic_search(query, collection_name, limit, filters)
                filter_desc = f" (filters: {filters})" if filters else ""
                display_results(results, f"Filtered Semantic Search Results{filter_desc}")
            
            # Show search timing
            search_time = time.time() - start_time
            print(f"\n⏱️  Search completed in {search_time:.3f} seconds")
            
            input("\nPress Enter to continue...")

        elif choice == "6":  # Analytics
            clear_screen()
            print("📊 Search Analytics Dashboard")
            print("=" * 40)
            
            try:
                hours = input("Analytics period in hours (default 24): ").strip()
                hours = int(hours) if hours else 24
            except ValueError:
                hours = 24
            
            analytics = explorer.get_search_analytics(hours)
            display_analytics(analytics)
            
            input("\nPress Enter to continue...")

        elif choice == "7":  # Health Check
            clear_screen()
            print("🏥 System Health Check")
            print("=" * 30)
            
            health = explorer.health_check()
            
            print(f"🕐 Timestamp: {health['timestamp']}")
            print(f"⏰ Uptime: {health['uptime']}")
            print(f"📊 Overall Status: {health['status'].upper()}")
            
            print(f"\n🔧 Components:")
            for component, status in health['components'].items():
                status_icon = "✅" if status['status'] == 'healthy' else "❌"
                print(f"  {status_icon} {component.title()}: {status['status']}")
                
                if status['status'] == 'healthy':
                    if component == 'qdrant':
                        print(f"      - Collections: {status['collections_count']}")
                        print(f"      - Total Vectors: {status['total_vectors']:,}")
                    elif component == 'embedding_model':
                        print(f"      - Dimension: {status['dimension']}")
                else:
                    print(f"      - Error: {status.get('error', 'Unknown error')}")
            
            print(f"\n💻 System Resources:")
            sys_info = health['system']
            print(f"  - Memory Usage: {sys_info['memory_percent']:.1f}%")
            print(f"  - Total Searches: {sys_info['total_searches']:,}")
            print(f"  - Cache Hits: {sys_info['cache_hits']:,}")
            
            cache_stats = sys_info['cache_stats']
            print(f"  - Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
            print(f"  - Cache Hit Rate: {cache_stats['hit_rate']:.1%}")
            
            input("\nPress Enter to continue...")

        elif choice == "8":  # Clear Cache
            clear_screen()
            confirm = input("🧹 Are you sure you want to clear the search cache? (y/N): ").strip().lower()
            if confirm == 'y':
                explorer.clear_cache()
                print("✅ Search cache cleared successfully!")
            else:
                print("❌ Cache clear cancelled.")
            
            input("\nPress Enter to continue...")

        elif choice == "9":  # Configuration Info
            clear_screen()
            print("⚙️  Current Configuration")
            print("=" * 30)
            
            config = explorer.config
            
            print(f"🔌 Qdrant Connection:")
            print(f"   Host: {config['qdrant']['host']}")
            print(f"   Port: {config['qdrant']['port']}")
            print(f"   Timeout: {config['qdrant']['timeout']}s")
            
            print(f"\n🤖 Model Configuration:")
            print(f"   Name: {config['model']['name']}")
            print(f"   Device: {config['model']['device']}")
            
            print(f"\n🔍 Search Configuration:")
            print(f"   Default Limit: {config['search']['default_limit']}")
            print(f"   Max Limit: {config['search']['max_limit']}")
            print(f"   Cache Size: {config['search']['cache_size']}")
            print(f"   Cache TTL: {config['search']['cache_ttl']}s")
            
            print(f"\n🔒 Security Configuration:")
            print(f"   Max Query Length: {config['security']['max_query_length']}")
            print(f"   Sanitize Queries: {config['security']['sanitize_queries']}")
            
            input("\nPress Enter to continue...")

        else:
            print("\n❌ Invalid choice. Please try again.")
            input("Press Enter to continue...")

async def main():
    """Async main function with comprehensive error handling"""
    try:
        print("🚀 Initializing QDrant Vector Search Explorer...")
        
        # Initialize explorer
        explorer = QdrantExplorer()
        
        # Perform initial health check
        print("🔍 Performing initial health check...")
        health = explorer.health_check()
        
        if health['status'] == 'healthy':
            print("✅ System is healthy and ready!")
        else:
            print(f"⚠️  System status: {health['status']}")
            for component, status in health['components'].items():
                if status['status'] != 'healthy':
                    print(f"   ❌ {component}: {status.get('error', 'Unknown error')}")
            
            proceed = input("\nContinue anyway? (y/N): ").strip().lower()
            if proceed != 'y':
                print("👋 Exiting...")
                return
        
        # Start main menu
        main_menu(explorer)
        
    except KeyboardInterrupt:
        print("\n\n👋 Search session interrupted by user. Goodbye!")
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("Please check if Qdrant is running and accessible.")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        logging.error(f"Fatal error in main: {e}", exc_info=True)
    finally:
        print("🔚 QDrant Vector Search Explorer session ended.")

if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except RuntimeError:
        # Fallback for environments without async support
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()