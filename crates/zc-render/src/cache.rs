use std::collections::HashMap;
use std::sync::Arc;

/// Default finished-frame RGBA8 cache capacity (512 MiB per Spec 04 §5.1).
pub const DEFAULT_FRAME_CACHE_MAX_BYTES: usize = 512 * 1024 * 1024;

/// Hard upper bound on cached frame entries (per Spec 04 §5.1).
pub const MAX_FRAME_CACHE_ENTRIES: usize = 8192;

/// Unique key identifying a rendered finished frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameCacheKey {
    pub frame_index: i64,
    pub width: u32,
    pub height: u32,
    pub revision: u64,
}

impl FrameCacheKey {
    pub const fn new(frame_index: i64, width: u32, height: u32, revision: u64) -> Self {
        Self {
            frame_index,
            width,
            height,
            revision,
        }
    }
}

/// Immutable finished RGBA8 frame stored in [`FrameCache`].
#[derive(Debug, Clone)]
pub struct CachedFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Arc<[u8]>,
}

#[derive(Debug)]
struct LruNode {
    key: FrameCacheKey,
    frame: CachedFrame,
    bytes: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Bounded O(1) LRU finished-frame RGBA8 cache with both a byte cap and a hard
/// entry limit (`8192`).
#[derive(Debug)]
pub struct FrameCache {
    nodes: Vec<Option<LruNode>>,
    free_slots: Vec<usize>,
    index: HashMap<FrameCacheKey, usize>,
    head: Option<usize>,
    tail: Option<usize>,
    current_bytes: usize,
    max_bytes: usize,
    max_entries: usize,
}

impl Default for FrameCache {
    fn default() -> Self {
        Self::new(DEFAULT_FRAME_CACHE_MAX_BYTES)
    }
}

impl FrameCache {
    /// Default capacity constant (`512 MiB`).
    pub const DEFAULT_MAX_BYTES: usize = DEFAULT_FRAME_CACHE_MAX_BYTES;
    /// Hard maximum number of cached entries (`8192`).
    pub const MAX_ENTRIES: usize = MAX_FRAME_CACHE_ENTRIES;

    /// Creates a new bounded LRU frame cache with the specified byte budget.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            nodes: Vec::new(),
            free_slots: Vec::new(),
            index: HashMap::new(),
            head: None,
            tail: None,
            current_bytes: 0,
            max_bytes,
            max_entries: Self::MAX_ENTRIES,
        }
    }

    /// Looks up a cached frame and promotes it to most-recently-used on hit.
    pub fn get(&mut self, key: &FrameCacheKey) -> Option<CachedFrame> {
        let slot = *self.index.get(key)?;
        self.move_to_front(slot);
        self.nodes[slot].as_ref().map(|n| n.frame.clone())
    }

    /// Non-mutating cache lookup that does not alter LRU ordering.
    pub fn peek(&self, key: &FrameCacheKey) -> Option<&CachedFrame> {
        let slot = *self.index.get(key)?;
        self.nodes[slot].as_ref().map(|n| &n.frame)
    }

    /// Inserts or replaces a finished RGBA8 frame, evicting least-recently-used
    /// entries until both `max_bytes` and `MAX_ENTRIES` are satisfied.
    ///
    /// Returns `true` if the frame was stored, or `false` if the frame exceeds
    /// the cache's total byte budget.
    pub fn insert(&mut self, key: FrameCacheKey, width: u32, height: u32, pixels: Vec<u8>) -> bool {
        let bytes = pixels.len();
        if self.max_bytes == 0 || bytes > self.max_bytes || self.max_entries == 0 {
            self.remove(&key);
            return false;
        }

        let frame = CachedFrame {
            width,
            height,
            pixels: Arc::from(pixels),
        };

        if let Some(&slot) = self.index.get(&key) {
            if let Some(node) = self.nodes[slot].as_mut() {
                self.current_bytes = self
                    .current_bytes
                    .saturating_sub(node.bytes)
                    .saturating_add(bytes);
                node.frame = frame;
                node.bytes = bytes;
            }
            self.move_to_front(slot);
        } else {
            let slot = self.alloc_slot(LruNode {
                key,
                frame,
                bytes,
                prev: None,
                next: None,
            });
            self.index.insert(key, slot);
            self.current_bytes = self.current_bytes.saturating_add(bytes);
            self.attach_front(slot);
        }

        self.evict_excess();
        true
    }

    /// Removes a specific frame from the cache if present.
    pub fn remove(&mut self, key: &FrameCacheKey) -> bool {
        let Some(slot) = self.index.remove(key) else {
            return false;
        };
        self.detach(slot);
        if let Some(node) = self.nodes[slot].take() {
            self.current_bytes = self.current_bytes.saturating_sub(node.bytes);
        }
        self.free_slots.push(slot);
        true
    }

    /// Evicts every cached frame immediately.
    pub fn invalidate_all(&mut self) {
        self.nodes.clear();
        self.free_slots.clear();
        self.index.clear();
        self.head = None;
        self.tail = None;
        self.current_bytes = 0;
    }

    /// Shrinks the maximum byte budget to 90% under VRAM/memory pressure and
    /// evicts least-recently-used entries to fit.
    pub fn shrink_for_pressure(&mut self) {
        let shrunk = (self.max_bytes / 10).saturating_mul(9);
        self.set_max_bytes(shrunk);
    }

    /// Updates the maximum byte budget and immediately evicts excess entries.
    pub fn set_max_bytes(&mut self, max_bytes: usize) {
        self.max_bytes = max_bytes;
        self.evict_excess();
    }

    /// Current total RGBA8 pixel bytes stored in the cache.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Configured maximum byte budget.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Current number of cached frames.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns `true` if no frames are currently cached.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn alloc_slot(&mut self, node: LruNode) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            self.nodes[slot] = Some(node);
            slot
        } else {
            let slot = self.nodes.len();
            self.nodes.push(Some(node));
            slot
        }
    }

    fn attach_front(&mut self, slot: usize) {
        let old_head = self.head;
        if let Some(node) = self.nodes[slot].as_mut() {
            node.prev = None;
            node.next = old_head;
        }
        if let Some(h) = old_head {
            if let Some(head_node) = self.nodes[h].as_mut() {
                head_node.prev = Some(slot);
            }
        } else {
            self.tail = Some(slot);
        }
        self.head = Some(slot);
    }

    fn detach(&mut self, slot: usize) {
        let (prev, next) = match self.nodes[slot].as_ref() {
            Some(node) => (node.prev, node.next),
            None => return,
        };

        if let Some(p) = prev {
            if let Some(prev_node) = self.nodes[p].as_mut() {
                prev_node.next = next;
            }
        } else {
            self.head = next;
        }

        if let Some(n) = next {
            if let Some(next_node) = self.nodes[n].as_mut() {
                next_node.prev = prev;
            }
        } else {
            self.tail = prev;
        }

        if let Some(node) = self.nodes[slot].as_mut() {
            node.prev = None;
            node.next = None;
        }
    }

    fn move_to_front(&mut self, slot: usize) {
        if self.head == Some(slot) {
            return;
        }
        self.detach(slot);
        self.attach_front(slot);
    }

    fn evict_excess(&mut self) {
        while self.current_bytes > self.max_bytes || self.index.len() > self.max_entries {
            let Some(lru_slot) = self.tail else {
                break;
            };
            self.detach(lru_slot);
            if let Some(node) = self.nodes[lru_slot].take() {
                self.index.remove(&node.key);
                self.current_bytes = self.current_bytes.saturating_sub(node.bytes);
            }
            self.free_slots.push(lru_slot);
        }
    }
}
