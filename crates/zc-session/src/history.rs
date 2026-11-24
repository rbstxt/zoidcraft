use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default maximum number of entries retained in each history stack.
pub const DEFAULT_MAX_ENTRIES: usize = 200;

/// Default maximum total memory budget (64 MiB) across undo + redo history.
pub const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Trait for history snapshots that can estimate their memory footprint in bytes.
pub trait ByteSized {
    /// Approximate memory footprint of this snapshot in bytes.
    fn estimated_bytes(&self) -> usize;
}

impl ByteSized for String {
    fn estimated_bytes(&self) -> usize {
        self.capacity().max(self.len())
    }
}

impl ByteSized for Vec<u8> {
    fn estimated_bytes(&self) -> usize {
        self.capacity().max(self.len())
    }
}

impl ByteSized for serde_json::Value {
    fn estimated_bytes(&self) -> usize {
        serde_json::to_vec(self).map_or(0, |buf| buf.len())
    }
}

/// Errors produced by [`UndoStack`] transaction operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum HistoryError {
    #[error("a history transaction is already active")]
    TransactionAlreadyActive,
    #[error("no history transaction is currently active")]
    NoActiveTransaction,
}

/// One stored snapshot entry in the undo or redo stack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEntry<T> {
    pub label: String,
    pub snapshot: T,
    pub bytes: usize,
    pub revision: u64,
}

#[derive(Debug, Clone)]
struct ActiveTransaction<T> {
    label: String,
    baseline: T,
    baseline_bytes: usize,
    baseline_revision: u64,
    edit_count: usize,
}

/// Bounded undo/redo stack enforcing both entry-count and byte-budget limits,
/// with transaction grouping for live scrub/drag interactions.
#[derive(Debug, Clone)]
pub struct UndoStack<T> {
    undo_stack: VecDeque<HistoryEntry<T>>,
    redo_stack: Vec<HistoryEntry<T>>,
    undo_bytes: usize,
    redo_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
    current_revision: u64,
    saved_revision: Option<u64>,
    next_revision: u64,
    active_tx: Option<ActiveTransaction<T>>,
}

impl<T> Default for UndoStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> UndoStack<T> {
    /// Creates an empty undo stack using [`DEFAULT_MAX_ENTRIES`] and [`DEFAULT_MAX_BYTES`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_BYTES)
    }

    /// Creates an empty undo stack with explicit `max_entries` and `max_bytes` limits.
    #[must_use]
    pub fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: Vec::new(),
            undo_bytes: 0,
            redo_bytes: 0,
            max_entries,
            max_bytes,
            current_revision: 0,
            saved_revision: Some(0),
            next_revision: 1,
            active_tx: None,
        }
    }

    /// Configured maximum number of entries in the undo (or redo) stack.
    #[must_use]
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Configured maximum combined byte budget for undo + redo stacks.
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Updates limits and immediately evicts oldest entries if needed.
    pub fn set_limits(&mut self, max_entries: usize, max_bytes: usize) {
        self.max_entries = max_entries;
        self.max_bytes = max_bytes;
        self.enforce_limits(false);
    }

    /// Number of entries currently on the undo stack.
    #[must_use]
    pub fn undo_len(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of entries currently on the redo stack.
    #[must_use]
    pub fn redo_len(&self) -> usize {
        self.redo_stack.len()
    }

    /// Total tracked bytes on the undo stack.
    #[must_use]
    pub fn undo_bytes(&self) -> usize {
        self.undo_bytes
    }

    /// Total tracked bytes on the redo stack.
    #[must_use]
    pub fn redo_bytes(&self) -> usize {
        self.redo_bytes
    }

    /// Combined tracked bytes across both undo and redo stacks.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.undo_bytes.saturating_add(self.redo_bytes)
    }

    /// Returns `true` if at least one undo step is available and no transaction is active.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty() && self.active_tx.is_none()
    }

    /// Returns `true` if at least one redo step is available and no transaction is active.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty() && self.active_tx.is_none()
    }

    /// Label of the next operation that `undo` would revert.
    #[must_use]
    pub fn peek_undo_label(&self) -> Option<&str> {
        self.undo_stack.back().map(|entry| entry.label.as_str())
    }

    /// Label of the next operation that `redo` would reapply.
    #[must_use]
    pub fn peek_redo_label(&self) -> Option<&str> {
        self.redo_stack.last().map(|entry| entry.label.as_str())
    }

    /// Returns `true` when the current revision differs from the last saved revision.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.saved_revision != Some(self.current_revision)
    }

    /// Monotonic revision identifier of the current state.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.current_revision
    }

    /// Marks the current revision as saved (clears [`Self::is_dirty`]).
    pub fn mark_saved(&mut self) {
        self.saved_revision = Some(self.current_revision);
    }

    /// Returns `true` while a live transaction is in progress.
    #[must_use]
    pub fn is_in_transaction(&self) -> bool {
        self.active_tx.is_some()
    }

    /// Returns the label of the active transaction, if any.
    #[must_use]
    pub fn active_transaction_label(&self) -> Option<&str> {
        self.active_tx.as_ref().map(|tx| tx.label.as_str())
    }

    /// Iterates over all snapshots currently retained in undo history, redo history, and any active transaction baseline.
    pub fn snapshots(&self) -> impl Iterator<Item = &T> {
        self.undo_stack
            .iter()
            .map(|e| &e.snapshot)
            .chain(self.redo_stack.iter().map(|e| &e.snapshot))
            .chain(self.active_tx.iter().map(|tx| &tx.baseline))
    }

    /// Clears all undo/redo entries and resets dirty tracking to a clean state.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.undo_bytes = 0;
        self.redo_bytes = 0;
        self.active_tx = None;
        self.current_revision = 0;
        self.saved_revision = Some(0);
        self.next_revision = 1;
    }

    /// Records a single undo step with an explicit byte size estimate.
    ///
    /// If a transaction is currently active, this folds into the active transaction
    /// without pushing an intermediate undo entry.
    pub fn push_with_bytes(&mut self, label: impl Into<String>, before_snapshot: T, bytes: usize) {
        if self.active_tx.is_some() {
            self.record_transaction_edit();
            return;
        }

        self.clear_redo();
        let before_revision = self.current_revision;
        self.current_revision = self.alloc_revision();

        if self.max_entries == 0 || bytes > self.max_bytes {
            self.clear_undo();
            return;
        }

        self.undo_bytes = self.undo_bytes.saturating_add(bytes);
        self.undo_stack.push_back(HistoryEntry {
            label: label.into(),
            snapshot: before_snapshot,
            bytes,
            revision: before_revision,
        });
        self.enforce_limits(false);
    }

    /// Records a single undo step using [`ByteSized::estimated_bytes`].
    pub fn push(&mut self, label: impl Into<String>, before_snapshot: T)
    where
        T: ByteSized,
    {
        let bytes = before_snapshot.estimated_bytes();
        self.push_with_bytes(label, before_snapshot, bytes);
    }

    /// Begins a grouped transaction (for example, live drag-scrubbing or clip dragging)
    /// capturing the pre-gesture `baseline` snapshot and its estimated size in bytes.
    pub fn begin_transaction_with_bytes(
        &mut self,
        label: impl Into<String>,
        baseline: T,
        baseline_bytes: usize,
    ) -> Result<(), HistoryError> {
        if self.active_tx.is_some() {
            return Err(HistoryError::TransactionAlreadyActive);
        }

        self.active_tx = Some(ActiveTransaction {
            label: label.into(),
            baseline,
            baseline_bytes,
            baseline_revision: self.current_revision,
            edit_count: 0,
        });
        Ok(())
    }

    /// Begins a grouped transaction using [`ByteSized::estimated_bytes`].
    pub fn begin_transaction(
        &mut self,
        label: impl Into<String>,
        baseline: T,
    ) -> Result<(), HistoryError>
    where
        T: ByteSized,
    {
        let bytes = baseline.estimated_bytes();
        self.begin_transaction_with_bytes(label, baseline, bytes)
    }

    /// Marks that a live edit occurred inside the active transaction without pushing an undo step.
    pub fn record_transaction_edit(&mut self) {
        if let Some(tx) = self.active_tx.as_mut() {
            tx.edit_count = tx.edit_count.saturating_add(1);
            let rev = self.next_revision;
            self.next_revision = self.next_revision.saturating_add(1);
            self.current_revision = rev;
        }
    }

    /// Commits the active transaction as **one** undo step if `changed` is `true`.
    ///
    /// When `changed` is `false` (for example, the user scrubbed back to the initial value),
    /// the transaction is discarded without modifying history and `Ok(false)` is returned.
    pub fn commit_transaction(&mut self, changed: bool) -> Result<bool, HistoryError> {
        let tx = self
            .active_tx
            .take()
            .ok_or(HistoryError::NoActiveTransaction)?;

        if !changed {
            self.current_revision = tx.baseline_revision;
            return Ok(false);
        }

        self.clear_redo();
        if tx.edit_count == 0 {
            self.current_revision = self.alloc_revision();
        }

        if self.max_entries > 0 && tx.baseline_bytes <= self.max_bytes {
            self.undo_bytes = self.undo_bytes.saturating_add(tx.baseline_bytes);
            self.undo_stack.push_back(HistoryEntry {
                label: tx.label,
                snapshot: tx.baseline,
                bytes: tx.baseline_bytes,
                revision: tx.baseline_revision,
            });
            self.enforce_limits(false);
        } else {
            self.clear_undo();
        }

        Ok(true)
    }

    /// Cancels the active transaction, leaving history untouched and returning the baseline
    /// snapshot so the caller can restore pre-drag state.
    pub fn cancel_transaction(&mut self) -> Result<T, HistoryError> {
        let tx = self
            .active_tx
            .take()
            .ok_or(HistoryError::NoActiveTransaction)?;
        self.current_revision = tx.baseline_revision;
        Ok(tx.baseline)
    }

    /// Undoes the most recent operation, pushing `current_snapshot` onto the redo stack
    /// and returning `(label, previous_snapshot)`.
    pub fn undo_with_bytes(
        &mut self,
        current_snapshot: T,
        current_bytes: usize,
    ) -> Option<(String, T)> {
        if self.active_tx.is_some() {
            return None;
        }

        let entry = self.undo_stack.pop_back()?;
        self.undo_bytes = self.undo_bytes.saturating_sub(entry.bytes);

        let redo_revision = self.current_revision;
        self.current_revision = entry.revision;

        if self.max_entries > 0 && current_bytes <= self.max_bytes {
            self.redo_bytes = self.redo_bytes.saturating_add(current_bytes);
            self.redo_stack.push(HistoryEntry {
                label: entry.label.clone(),
                snapshot: current_snapshot,
                bytes: current_bytes,
                revision: redo_revision,
            });
            self.enforce_limits(true);
        } else {
            self.clear_redo();
        }

        Some((entry.label, entry.snapshot))
    }

    /// Undoes the most recent operation using [`ByteSized::estimated_bytes`].
    pub fn undo(&mut self, current_snapshot: T) -> Option<(String, T)>
    where
        T: ByteSized,
    {
        let bytes = current_snapshot.estimated_bytes();
        self.undo_with_bytes(current_snapshot, bytes)
    }

    /// Redoes the most recently undone operation, pushing `current_snapshot` back onto
    /// the undo stack and returning `(label, redone_snapshot)`.
    pub fn redo_with_bytes(
        &mut self,
        current_snapshot: T,
        current_bytes: usize,
    ) -> Option<(String, T)> {
        if self.active_tx.is_some() {
            return None;
        }

        let entry = self.redo_stack.pop()?;
        self.redo_bytes = self.redo_bytes.saturating_sub(entry.bytes);

        let undo_revision = self.current_revision;
        self.current_revision = entry.revision;

        if self.max_entries > 0 && current_bytes <= self.max_bytes {
            self.undo_bytes = self.undo_bytes.saturating_add(current_bytes);
            self.undo_stack.push_back(HistoryEntry {
                label: entry.label.clone(),
                snapshot: current_snapshot,
                bytes: current_bytes,
                revision: undo_revision,
            });
            self.enforce_limits(false);
        } else {
            self.clear_undo();
        }

        Some((entry.label, entry.snapshot))
    }

    /// Redoes the most recently undone operation using [`ByteSized::estimated_bytes`].
    pub fn redo(&mut self, current_snapshot: T) -> Option<(String, T)>
    where
        T: ByteSized,
    {
        let bytes = current_snapshot.estimated_bytes();
        self.redo_with_bytes(current_snapshot, bytes)
    }

    fn alloc_revision(&mut self) -> u64 {
        let rev = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        rev
    }

    fn clear_undo(&mut self) {
        self.undo_stack.clear();
        self.undo_bytes = 0;
    }

    fn clear_redo(&mut self) {
        self.redo_stack.clear();
        self.redo_bytes = 0;
    }

    fn enforce_limits(&mut self, prefer_keeping_redo: bool) {
        while self.undo_stack.len() > self.max_entries {
            if let Some(evicted) = self.undo_stack.pop_front() {
                self.undo_bytes = self.undo_bytes.saturating_sub(evicted.bytes);
            } else {
                break;
            }
        }
        while self.redo_stack.len() > self.max_entries {
            let evicted = self.redo_stack.remove(0);
            self.redo_bytes = self.redo_bytes.saturating_sub(evicted.bytes);
        }

        while self.total_bytes() > self.max_bytes {
            if prefer_keeping_redo {
                if let Some(evicted) = self.undo_stack.pop_front() {
                    self.undo_bytes = self.undo_bytes.saturating_sub(evicted.bytes);
                } else if !self.redo_stack.is_empty() {
                    let evicted = self.redo_stack.remove(0);
                    self.redo_bytes = self.redo_bytes.saturating_sub(evicted.bytes);
                } else {
                    break;
                }
            } else if !self.redo_stack.is_empty() {
                let evicted = self.redo_stack.remove(0);
                self.redo_bytes = self.redo_bytes.saturating_sub(evicted.bytes);
            } else if let Some(evicted) = self.undo_stack.pop_front() {
                self.undo_bytes = self.undo_bytes.saturating_sub(evicted.bytes);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_tracking_across_save_undo_redo_and_branch() {
        let mut stack = UndoStack::<String>::new();
        let mut current = String::from("v0");
        assert!(!stack.is_dirty());

        stack.push("edit 1", current.clone());
        current = String::from("v1");
        assert!(stack.is_dirty());

        stack.mark_saved();
        assert!(!stack.is_dirty());

        let (label, prev) = stack.undo(current).expect("undo available");
        assert_eq!(label, "edit 1");
        current = prev;
        assert_eq!(current, "v0");
        assert!(stack.is_dirty());

        let (label, next) = stack.redo(current).expect("redo available");
        assert_eq!(label, "edit 1");
        current = next;
        assert_eq!(current, "v1");
        assert!(!stack.is_dirty());

        // Undo back to v0, then branch with a new edit so saved v1 is unreachable.
        let (_, prev) = stack.undo(current).expect("undo available");
        current = prev;
        stack.push("edit 2", current);
        assert!(stack.is_dirty());
        assert!(!stack.can_redo());
    }

    #[test]
    fn memory_and_entry_bounds_never_exceeded() {
        let mut stack = UndoStack::<Vec<u8>>::with_limits(200, 64 * 1024);
        for i in 0..1_000_usize {
            let payload = vec![(i & 0xFF) as u8; 4096];
            stack.push(format!("edit-{i}"), payload);
            assert!(stack.undo_len() <= 200);
            assert!(stack.total_bytes() <= 64 * 1024);
        }
        assert_eq!(stack.undo_len(), 16);
        assert_eq!(stack.total_bytes(), 64 * 1024);
    }

    #[test]
    fn oversized_snapshot_clears_stack_to_preserve_contiguity() {
        let mut stack = UndoStack::<Vec<u8>>::with_limits(10, 16);
        stack.push("small-1", vec![1u8; 4]);
        stack.push("small-2", vec![2u8; 4]);
        assert_eq!(stack.undo_len(), 2);

        // Oversized push cannot be stored -> clears undo_stack so Undo never skips it.
        stack.push("too-big", vec![3u8; 32]);
        assert_eq!(stack.undo_len(), 0);
        assert!(!stack.can_undo());

        // Same for commit_transaction when baseline exceeds max_bytes.
        stack.push("small-3", vec![4u8; 4]);
        assert_eq!(stack.undo_len(), 1);
        stack
            .begin_transaction("big-tx", vec![5u8; 32])
            .expect("begin tx");
        assert!(stack.commit_transaction(true).expect("commit tx"));
        assert_eq!(stack.undo_len(), 0);

        // Oversized current snapshot on undo clears stale redo_stack.
        stack.push("e1", vec![1u8; 4]);
        stack.push("e2", vec![2u8; 4]);
        let _ = stack.undo(vec![3u8; 4]).expect("undo e2");
        assert_eq!(stack.redo_len(), 1);
        let _ = stack
            .undo(vec![9u8; 32])
            .expect("undo e1 with huge current");
        assert_eq!(stack.redo_len(), 0);

        // Oversized current snapshot on redo clears stale undo_stack.
        stack.push("r1", vec![1u8; 4]);
        stack.push("r2", vec![2u8; 4]);
        let _ = stack.undo(vec![3u8; 4]).expect("undo r2");
        assert_eq!(stack.undo_len(), 1);
        assert_eq!(stack.redo_len(), 1);
        let _ = stack
            .redo(vec![8u8; 32])
            .expect("redo r2 with huge current");
        assert_eq!(stack.undo_len(), 0);
    }
}
