export * from "./memory";
export * from "./memory/memory.types";
export * from "./types";

// Base classes + the always-available OpenAI LLM/embedder are re-exported
// eagerly — their only external deps (openai, uuid, zod, axios) are core
// (non-optional) dependencies of the published package.
export * from "./embeddings/base";
export * from "./embeddings/openai";
export * from "./llms/base";
export * from "./llms/openai";
export * from "./llms/openai_structured";
export * from "./vector_stores/base";
export * from "./utils/factory";

// NOTE: The optional providers are intentionally NOT statically re-exported
// here. Each of those modules statically imports a heavy provider SDK that is
// an OPTIONAL peerDependency; a top-level `export *` would force every one of
// those SDKs to resolve at `import('mem0ai/oss')` time, which breaks loading
// in any install that hasn't added all of the optional SDKs. This includes
// the in-memory vector store (`./vector_stores/memory`), which statically
// imports the native `better-sqlite3` module (also an optional peer). The
// factories (utils/factory.ts) load every provider lazily (dynamic import)
// only when configured. Callers that genuinely need a provider class as a
// named export can deep-import its source module directly (after installing
// that provider's SDK).
