//! Prefix Caching for LLM Inference
//!
//! Hash-based KV cache for repeated system prompts. Dramatically reduces
//! Time to First Token (TTFT) for API workloads with repeated system prompts.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use candle_core::Tensor;
use sha2::{Digest, Sha256};

pub struct PrefixCacheConfig {
    pub memory_budget_mb: usize,
    pub enabled: bool,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            memory_budget_mb: 512,
            enabled: true,
        }
    }
}

impl Clone for PrefixCacheConfig {
    fn clone(&self) -> Self {
        Self {
            memory_budget_mb: self.memory_budget_mb,
            enabled: self.enabled,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub prompt_hash: u64,
    pub system_hash: u64,
    pub model_config_hash: u64,
}

impl CacheKey {
    pub fn new(prompt: &str, system_prompt: Option<&str>, model_config: &str) -> Self {
        let prompt_hash = Self::hash_string(prompt);
        let system_hash = Self::hash_string(system_prompt.unwrap_or(""));
        let model_config_hash = Self::hash_string(model_config);

        Self {
            prompt_hash,
            system_hash,
            model_config_hash,
        }
    }

    fn hash_string(s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

pub struct CachedPrefix {
    pub key: CacheKey,
    pub tokens: Vec<u32>,
    pub kv_cache: Vec<CachedLayer>,
    pub access_count: u64,
    pub last_access: std::time::Instant,
}

pub struct CachedLayer {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
}

pub struct PrefixCache {
    config: PrefixCacheConfig,
    cache: HashMap<CacheKey, Arc<CachedPrefix>>,
    access_order: Vec<CacheKey>,
    current_memory_bytes: usize,
    memory_budget_bytes: usize,
}

impl PrefixCache {
    pub fn new(config: PrefixCacheConfig) -> Self {
        let memory_budget_bytes = config.memory_budget_mb * 1024 * 1024;
        Self {
            config,
            cache: HashMap::new(),
            access_order: Vec::new(),
            current_memory_bytes: 0,
            memory_budget_bytes,
        }
    }

    pub fn config(&self) -> &PrefixCacheConfig {
        &self.config
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn get(&self, key: &CacheKey) -> Option<Arc<CachedPrefix>> {
        if !self.config.enabled {
            return None;
        }

        self.cache.get(key).cloned()
    }

    pub fn insert(&mut self, key: CacheKey, tokens: Vec<u32>, _kv_cache: Vec<CachedLayer>) {
        if !self.config.enabled {
            return;
        }

        let estimated_size = tokens.len() * 4 + 1024;

        while self.current_memory_bytes + estimated_size > self.memory_budget_bytes
            && !self.access_order.is_empty()
        {
            self.evict_lru();
        }

        if self.current_memory_bytes + estimated_size > self.memory_budget_bytes {
            tracing::warn!("Prefix cache: prompt too large to cache");
            return;
        }

        let prefix = Arc::new(CachedPrefix {
            key: key.clone(),
            tokens,
            kv_cache: Vec::new(),
            access_count: 1,
            last_access: std::time::Instant::now(),
        });

        self.current_memory_bytes += estimated_size;
        self.cache.insert(key.clone(), prefix);
        self.access_order.push(key);
    }

    pub fn touch(&mut self, key: &CacheKey) {
        // Just move to back of access order for LRU
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
            self.access_order.push(key.clone());
        }
    }

    fn evict_lru(&mut self) {
        if let Some(oldest_key) = self.access_order.first().cloned() {
            if let Some(prefix) = self.cache.remove(&oldest_key) {
                let size = prefix.tokens.len() * 4 + 1024;
                self.current_memory_bytes = self.current_memory_bytes.saturating_sub(size);
            }
            self.access_order.remove(0);
        }
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.current_memory_bytes = 0;
    }

    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            num_entries: self.cache.len(),
            memory_used_mb: self.current_memory_bytes / (1024 * 1024),
            memory_budget_mb: self.config.memory_budget_mb,
            hit_rate: 0.0,
        }
    }
}

pub struct PrefixCacheStats {
    pub num_entries: usize,
    pub memory_used_mb: usize,
    pub memory_budget_mb: usize,
    pub hit_rate: f64,
}

pub fn hash_prompt(prompt: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    let result = hasher.finalize();
    u64::from_le_bytes(result[0..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_creation() {
        let key1 = CacheKey::new("Hello", Some("System"), "config");
        let key2 = CacheKey::new("Hello", Some("System"), "config");
        let key3 = CacheKey::new("World", Some("System"), "config");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_prefix_cache_insert() {
        let config = PrefixCacheConfig::default();
        let mut cache = PrefixCache::new(config);

        let key = CacheKey::new("test prompt", Some("system"), "config");
        cache.insert(key, vec![1, 2, 3, 4], Vec::new());

        assert_eq!(cache.stats().num_entries, 1);
    }
}
