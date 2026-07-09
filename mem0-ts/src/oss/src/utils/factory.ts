import {
  EmbeddingConfig,
  HistoryStoreConfig,
  LLMConfig,
  VectorStoreConfig,
} from "../types";
import { Embedder } from "../embeddings/base";
import { LLM } from "../llms/base";
import { VectorStore } from "../vector_stores/base";
import { HistoryManager } from "../storage/base";

// Provider modules are loaded lazily via dynamic import() so that each
// provider's heavy external SDK (ollama, groq-sdk, pg, redis, cloudflare,
// chromadb, iovalkey, weaviate-client, @qdrant/js-client-rest, @pinecone-*,
// @azure/*, @google/genai, @langchain/core, @mistralai/mistralai,
// @supabase/supabase-js, @opensearch-project/opensearch, @upstash/vector,
// cassandra-driver, mongodb, mysql2, @zilliz/milvus2-sdk-node, ...) is only
// required when that specific provider is actually configured. Previously
// these were static top-level imports, which forced ALL optional provider
// SDKs to resolve at module-load time — so `import('mem0ai/oss')` threw
// ERR_MODULE_NOT_FOUND on the first missing optional dep even when the caller
// only used the OpenAI LLM/embedder + the in-memory vector store.
// These factory methods are async; all call sites run in async context
// (Memory._autoInitialize / reset / getEntityStore).

export class EmbedderFactory {
  static async create(
    provider: string,
    config: EmbeddingConfig,
  ): Promise<Embedder> {
    switch (provider.toLowerCase()) {
      case "openai": {
        const { OpenAIEmbedder } = await import("../embeddings/openai");
        return new OpenAIEmbedder(config);
      }
      case "ollama": {
        const { OllamaEmbedder } = await import("../embeddings/ollama");
        return new OllamaEmbedder(config);
      }
      case "lmstudio": {
        const { LMStudioEmbedder } = await import("../embeddings/lmstudio");
        return new LMStudioEmbedder(config);
      }
      case "together": {
        const { TogetherEmbedder } = await import("../embeddings/together");
        return new TogetherEmbedder(config);
      }
      case "google":
      case "gemini": {
        const { GoogleEmbedder } = await import("../embeddings/google");
        return new GoogleEmbedder(config);
      }
      case "azure_openai": {
        const { AzureOpenAIEmbedder } = await import("../embeddings/azure");
        return new AzureOpenAIEmbedder(config);
      }
      case "fastembed": {
        const { FastEmbedEmbedder } = await import("../embeddings/fastembed");
        return new FastEmbedEmbedder(config);
      }
      case "langchain": {
        const { LangchainEmbedder } = await import("../embeddings/langchain");
        return new LangchainEmbedder(config);
      }
      case "huggingface": {
        const { HuggingFaceEmbedder } =
          await import("../embeddings/huggingface");
        return new HuggingFaceEmbedder(config);
      }
      default:
        throw new Error(`Unsupported embedder provider: ${provider}`);
    }
  }
}

export class LLMFactory {
  static async create(provider: string, config: LLMConfig): Promise<LLM> {
    switch (provider.toLowerCase()) {
      case "openai": {
        const { OpenAILLM } = await import("../llms/openai");
        return new OpenAILLM(config);
      }
      case "openai_structured": {
        const { OpenAIStructuredLLM } =
          await import("../llms/openai_structured");
        return new OpenAIStructuredLLM(config);
      }
      case "anthropic": {
        const { AnthropicLLM } = await import("../llms/anthropic");
        return new AnthropicLLM(config);
      }
      case "groq": {
        const { GroqLLM } = await import("../llms/groq");
        return new GroqLLM(config);
      }
      case "ollama": {
        const { OllamaLLM } = await import("../llms/ollama");
        return new OllamaLLM(config);
      }
      case "lmstudio": {
        const { LMStudioLLM } = await import("../llms/lmstudio");
        return new LMStudioLLM(config);
      }
      case "google":
      case "gemini": {
        const { GoogleLLM } = await import("../llms/google");
        return new GoogleLLM(config);
      }
      case "azure_openai": {
        const { AzureOpenAILLM } = await import("../llms/azure");
        return new AzureOpenAILLM(config);
      }
      case "mistral": {
        const { MistralLLM } = await import("../llms/mistral");
        return new MistralLLM(config);
      }
      case "langchain": {
        const { LangchainLLM } = await import("../llms/langchain");
        return new LangchainLLM(config);
      }
      case "deepseek": {
        const { DeepSeekLLM } = await import("../llms/deepseek");
        return new DeepSeekLLM(config);
      }
      case "xai": {
        const { XAILLM } = await import("../llms/xai");
        return new XAILLM(config);
      }
      case "sarvam": {
        const { SarvamLLM } = await import("../llms/sarvam");
        return new SarvamLLM(config);
      }
      case "litellm": {
        const { LiteLLM } = await import("../llms/litellm");
        return new LiteLLM(config);
      }
      case "minimax": {
        const { MiniMaxLLM } = await import("../llms/minimax");
        return new MiniMaxLLM(config);
      }
      case "together": {
        const { TogetherLLM } = await import("../llms/together");
        return new TogetherLLM(config);
      }
      case "vllm": {
        const { VllmLLM } = await import("../llms/vllm");
        return new VllmLLM(config);
      }
      default:
        throw new Error(`Unsupported LLM provider: ${provider}`);
    }
  }
}

export class VectorStoreFactory {
  static async create(
    provider: string,
    config: VectorStoreConfig,
  ): Promise<VectorStore> {
    switch (provider.toLowerCase()) {
      case "memory": {
        const { MemoryVectorStore } = await import("../vector_stores/memory");
        return new MemoryVectorStore(config);
      }
      case "qdrant": {
        const { Qdrant } = await import("../vector_stores/qdrant");
        return new Qdrant(config as any);
      }
      case "chroma": {
        const { ChromaDB } = await import("../vector_stores/chroma");
        return new ChromaDB(config as any);
      }
      case "redis": {
        const { RedisDB } = await import("../vector_stores/redis");
        return new RedisDB(config as any);
      }
      case "valkey": {
        const { ValkeyDB } = await import("../vector_stores/valkey");
        return new ValkeyDB(config as any);
      }
      case "supabase": {
        const { SupabaseDB } = await import("../vector_stores/supabase");
        return new SupabaseDB(config as any);
      }
      case "langchain": {
        const { LangchainVectorStore } =
          await import("../vector_stores/langchain");
        return new LangchainVectorStore(config as any);
      }
      case "vectorize": {
        const { VectorizeDB } = await import("../vector_stores/vectorize");
        return new VectorizeDB(config as any);
      }
      case "azure-ai-search": {
        const { AzureAISearch } =
          await import("../vector_stores/azure_ai_search");
        return new AzureAISearch(config as any);
      }
      case "vertex_ai_vector_search": {
        const { VertexAIVectorSearch } =
          await import("../vector_stores/vertex_ai_vector_search");
        return new VertexAIVectorSearch(config as any);
      }
      case "pgvector": {
        const { PGVector } = await import("../vector_stores/pgvector");
        return new PGVector(config as any);
      }
      case "elasticsearch": {
        const { ElasticsearchDB } =
          await import("../vector_stores/elasticsearch");
        return new ElasticsearchDB(config as any);
      }
      case "opensearch": {
        const { OpenSearchDB } = await import("../vector_stores/opensearch");
        return new OpenSearchDB(config as any);
      }
      case "upstash_vector": {
        const { UpstashVector } =
          await import("../vector_stores/upstash_vector");
        return new UpstashVector(config as any);
      }
      case "azure_mysql": {
        const { AzureMySQLDB } = await import("../vector_stores/azure_mysql");
        return new AzureMySQLDB(config as any);
      }
      case "cassandra": {
        const { CassandraDB } = await import("../vector_stores/cassandra");
        return new CassandraDB(config as any);
      }
      case "pinecone": {
        const { PineconeDB } = await import("../vector_stores/pinecone");
        return new PineconeDB(config as any);
      }
      case "s3-vectors":
      case "s3_vectors": {
        const { S3Vectors } = await import("../vector_stores/s3_vectors");
        return new S3Vectors(config as any);
      }
      case "turbopuffer": {
        const { TurbopufferDB } = await import("../vector_stores/turbopuffer");
        return new TurbopufferDB(config as any);
      }
      case "milvus": {
        const { Milvus } = await import("../vector_stores/milvus");
        return new Milvus(config as any);
      }
      case "mongodb": {
        const { MongoDB } = await import("../vector_stores/mongodb");
        return new MongoDB(config as any);
      }
      case "weaviate": {
        const { WeaviateDB } = await import("../vector_stores/weaviate");
        return new WeaviateDB(config as any);
      }
      default:
        throw new Error(`Unsupported vector store provider: ${provider}`);
    }
  }
}

export class HistoryManagerFactory {
  static async create(
    provider: string,
    config: HistoryStoreConfig,
  ): Promise<HistoryManager> {
    switch (provider.toLowerCase()) {
      case "sqlite": {
        const { SQLiteManager } = await import("../storage/SQLiteManager");
        return new SQLiteManager(config.config.historyDbPath || ":memory:");
      }
      case "supabase": {
        const { SupabaseHistoryManager } =
          await import("../storage/SupabaseHistoryManager");
        return new SupabaseHistoryManager({
          supabaseUrl: config.config.supabaseUrl || "",
          supabaseKey: config.config.supabaseKey || "",
          tableName: config.config.tableName || "memory_history",
        });
      }
      case "memory": {
        const { MemoryHistoryManager } =
          await import("../storage/MemoryHistoryManager");
        return new MemoryHistoryManager();
      }
      default:
        throw new Error(`Unsupported history store provider: ${provider}`);
    }
  }
}
