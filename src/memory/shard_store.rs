//! Read a pinned Hub snapshot through logical Lance paths, merging unchanged flat files with
//! deterministic physical buckets. No writes or backend-specific state live in this decorator.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use object_store::path::Path as OPath;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore as OSObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OSResult,
};
use sha2::{Digest, Sha256};
use tokio::sync::OnceCell;

const SHARD_ROOT: &str = "__funes_shards__/v1";

/// Keep metadata outside Lance tables flat. Every captured Lance key has one deterministic
/// physical location, including files underneath prefixed tables and `_indices`.
pub(crate) fn physical_path(logical: &str) -> Option<String> {
    if logical.starts_with("__funes_shards__/") || !logical.split('/').any(|part| part.ends_with(".lance")) {
        return None;
    }
    let digest = hex::encode(Sha256::digest(logical.as_bytes()));
    Some(format!("{SHARD_ROOT}/{}/{logical}", &digest[..2]))
}

fn logical_path(physical: &OPath) -> Option<OPath> {
    let tail = physical.as_ref().strip_prefix(SHARD_ROOT)?.strip_prefix('/')?;
    let (bucket, logical) = tail.split_once('/')?;
    if bucket.len() != 2 || physical_path(logical).as_deref() != Some(physical.as_ref()) {
        return None;
    }
    Some(OPath::from(logical))
}

fn is_internal(path: &OPath) -> bool {
    path.as_ref() == "__funes_shards__" || path.as_ref().starts_with("__funes_shards__/")
}

fn matches_prefix(path: &OPath, prefix: Option<&OPath>) -> bool {
    prefix.is_none_or(|prefix| path.prefix_match(prefix).is_some())
}

fn is_version_hint(logical: &str) -> bool {
    logical.ends_with("/_versions/latest_version_hint.json") || logical == "_versions/latest_version_hint.json"
}

fn read_only() -> object_store::Error {
    object_store::Error::NotSupported {
        source: Box::new(std::io::Error::other("ShardStore is read-only")),
    }
}

/// The inner store must be pinned to an immutable revision. Inventory is memoized once for that
/// snapshot; auth and transient failures propagate and are not cached as an empty inventory.
#[derive(Clone, Debug)]
pub(crate) struct ShardStore {
    inner: Arc<dyn OSObjectStore>,
    inventory: Arc<OnceCell<BTreeMap<OPath, ObjectMeta>>>,
}

impl ShardStore {
    pub(crate) fn new(inner: Arc<dyn OSObjectStore>) -> Self {
        Self {
            inner,
            inventory: Arc::new(OnceCell::new()),
        }
    }

    async fn inventory(&self) -> OSResult<&BTreeMap<OPath, ObjectMeta>> {
        self.inventory
            .get_or_try_init(|| async {
                let prefix = OPath::from(SHARD_ROOT);
                let objects = match self.inner.list(Some(&prefix)).try_collect::<Vec<_>>().await {
                    Ok(objects) => objects,
                    Err(object_store::Error::NotFound { .. }) => Vec::new(),
                    Err(error) => return Err(error),
                };
                let mut inventory = BTreeMap::new();
                for mut object in objects {
                    if let Some(logical) = logical_path(&object.location) {
                        object.location = logical.clone();
                        inventory.insert(logical, object);
                    }
                }
                Ok(inventory)
            })
            .await
    }

    async fn logical_list(&self, prefix: Option<&OPath>) -> OSResult<Vec<ObjectMeta>> {
        if prefix.is_some_and(is_internal) {
            return Ok(Vec::new());
        }
        let mut objects = BTreeMap::new();
        let flat = match self.inner.list(prefix).try_collect::<Vec<_>>().await {
            Ok(objects) => objects,
            Err(object_store::Error::NotFound { .. }) => Vec::new(),
            Err(error) => return Err(error),
        };
        for object in flat {
            if !is_internal(&object.location) && matches_prefix(&object.location, prefix) {
                objects.insert(object.location.clone(), object);
            }
        }
        // A sharded replacement wins over a flat object with the same logical name, notably the
        // mutable version hint. Expose one logical object, not both physical copies.
        for (logical, object) in self.inventory().await? {
            if matches_prefix(logical, prefix) {
                objects.insert(logical.clone(), object.clone());
            }
        }
        Ok(objects.into_values().collect())
    }
}

impl std::fmt::Display for ShardStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ShardStore({})", self.inner)
    }
}

#[async_trait]
impl OSObjectStore for ShardStore {
    async fn get_opts(&self, location: &OPath, options: GetOptions) -> OSResult<GetResult> {
        let mut result = match physical_path(location.as_ref()) {
            Some(physical) => {
                let physical = OPath::from(physical);
                match self.inner.get_opts(&physical, options.clone()).await {
                    Ok(result) => result,
                    Err(object_store::Error::NotFound { .. }) => {
                        // The version hint is an optional derived acceleration artifact. If missing
                        // from sharded storage, we must NOT fall back to a stale flat hint (e.g. frozen
                        // at version 4 while sharded versions are at ~12k+), which would poison Lance into
                        // a sequential HEAD probe loop (4 -> 12k). Instead, returning NotFound forces Lance
                        // to cleanly fall back to authoritative inventory listing.
                        // All other files (manifests, data fragments, etc.) fall back to flat storage normally.
                        if is_version_hint(location.as_ref()) {
                            return Err(object_store::Error::NotFound {
                                path: location.to_string(),
                                source: "sharded version hint not found; skipping stale flat hint".into(),
                            });
                        }
                        self.inner.get_opts(location, options).await?
                    }
                    Err(error) => return Err(error),
                }
            }
            None => self.inner.get_opts(location, options).await?,
        };
        result.meta.location = location.clone();
        Ok(result)
    }

    fn list(&self, prefix: Option<&OPath>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let store = self.clone();
        let prefix = prefix.cloned();
        stream::once(async move { store.logical_list(prefix.as_ref()).await })
            .flat_map(|result| match result {
                Ok(objects) => stream::iter(objects.into_iter().map(Ok)).boxed(),
                Err(error) => stream::once(async move { Err(error) }).boxed(),
            })
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&OPath>) -> OSResult<ListResult> {
        let mut common_prefixes = BTreeSet::new();
        let mut objects = Vec::new();
        for object in self.logical_list(prefix).await? {
            let relative = match prefix {
                Some(prefix) => object
                    .location
                    .as_ref()
                    .strip_prefix(prefix.as_ref())
                    .unwrap_or_default()
                    .trim_start_matches('/'),
                None => object.location.as_ref(),
            };
            if let Some((directory, _)) = relative.split_once('/') {
                common_prefixes.insert(match prefix {
                    Some(prefix) => prefix.clone().join(directory),
                    None => OPath::from(directory),
                });
            } else {
                objects.push(object);
            }
        }
        Ok(ListResult {
            common_prefixes: common_prefixes.into_iter().collect(),
            objects,
        })
    }

    async fn put_opts(&self, _location: &OPath, _payload: PutPayload, _options: PutOptions) -> OSResult<PutResult> {
        Err(read_only())
    }

    async fn put_multipart_opts(
        &self,
        _location: &OPath,
        _options: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        Err(read_only())
    }

    fn delete_stream(&self, locations: BoxStream<'static, OSResult<OPath>>) -> BoxStream<'static, OSResult<OPath>> {
        locations
            .map(|location| location.and_then(|_| Err(read_only())))
            .boxed()
    }

    async fn copy_opts(&self, _from: &OPath, _to: &OPath, _options: CopyOptions) -> OSResult<()> {
        Err(read_only())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use object_store::memory::InMemory;
    use object_store::{GetRange, ObjectStoreExt};
    use std::sync::atomic::{AtomicUsize, Ordering};

    async fn put(store: &InMemory, path: &str, body: &'static [u8]) {
        store
            .put(&OPath::from(path), Bytes::from_static(body).into())
            .await
            .unwrap();
    }

    #[test]
    fn physical_keys_are_deterministic_and_do_not_shard_repo_metadata() {
        for logical in [
            "chunks.lance/_versions/10001.manifest",
            "nested/memory/chunks.lance/_indices/id/index.idx",
        ] {
            let physical = physical_path(logical).unwrap();
            assert_eq!(logical_path(&OPath::from(physical.clone())), Some(OPath::from(logical)));
            assert_eq!(physical_path(logical), Some(physical));
        }
        assert!(physical_path("README.md").is_none());
        assert!(physical_path("index_state.json").is_none());
        assert!(physical_path("__funes_shards__/v1/ff/chunks.lance/data/file.lance").is_none());
        assert!(logical_path(&OPath::from("__funes_shards__/v1/zz/chunks.lance/data/file.lance")).is_none());
    }

    #[tokio::test]
    async fn flat_and_sharded_reads_preserve_logical_metadata_ranges_and_head() {
        let backend = Arc::new(InMemory::new());
        put(&backend, "chunks.lance/data/old.lance", b"original").await;
        let logical = "chunks.lance/data/new.lance";
        put(&backend, &physical_path(logical).unwrap(), b"0123456789").await;
        let store = ShardStore::new(backend);
        assert_eq!(
            store
                .get(&OPath::from("chunks.lance/data/old.lance"))
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            "original"
        );
        let path = OPath::from(logical);
        let result = store
            .get_opts(
                &path,
                GetOptions {
                    range: Some(GetRange::Bounded(2..5)),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(result.meta.location, path);
        assert_eq!(result.range, 2..5);
        assert_eq!(result.bytes().await.unwrap(), "234");
        assert_eq!(store.head(&path).await.unwrap().location, path);
        assert_eq!(
            store.get_ranges(&path, &[0..2, 7..10]).await.unwrap(),
            vec![Bytes::from_static(b"01"), Bytes::from_static(b"789")]
        );
    }

    #[tokio::test]
    async fn logical_listing_is_sorted_deduplicated_component_scoped_and_hides_shards() {
        let backend = Arc::new(InMemory::new());
        put(&backend, "README.md", b"metadata").await;
        put(&backend, "chunks.lance/data/a.lance", b"old").await;
        put(&backend, "chunks.lance/database/not-a-child", b"unrelated").await;
        for logical in [
            "chunks.lance/data/a.lance",
            "chunks.lance/data/b.lance",
            "chunks.lance/_versions/10001.manifest",
        ] {
            put(&backend, &physical_path(logical).unwrap(), b"new").await;
        }
        let store = ShardStore::new(backend);
        let all = store.list(None).try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(all.len(), 5);
        assert!(all.iter().all(|meta| !is_internal(&meta.location)));
        assert!(all.windows(2).all(|pair| pair[0].location < pair[1].location));
        let data = store
            .list(Some(&OPath::from("chunks.lance/data")))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(
            store
                .get(&OPath::from("chunks.lance/data/a.lance"))
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            "new"
        );
        let top = store.list_with_delimiter(None).await.unwrap();
        assert_eq!(top.common_prefixes, vec![OPath::from("chunks.lance")]);
        assert_eq!(top.objects.len(), 1);
        let table = store
            .list_with_delimiter(Some(&OPath::from("chunks.lance")))
            .await
            .unwrap();
        assert_eq!(
            table.common_prefixes,
            vec![
                OPath::from("chunks.lance/_versions"),
                OPath::from("chunks.lance/data"),
                OPath::from("chunks.lance/database")
            ]
        );
        assert!(store
            .list(Some(&OPath::from(SHARD_ROOT)))
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn full_flat_version_directory_can_expose_another_sharded_manifest() {
        let backend = Arc::new(InMemory::new());
        for version in 1..=10_000 {
            put(&backend, &format!("chunks.lance/_versions/{version}.manifest"), b"flat").await;
        }
        let new = "chunks.lance/_versions/10001.manifest";
        put(&backend, &physical_path(new).unwrap(), b"new").await;
        let store = ShardStore::new(backend.clone());
        let manifests = store
            .list(Some(&OPath::from("chunks.lance/_versions")))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(manifests.len(), 10_001);
        assert_eq!(
            backend
                .list(Some(&OPath::from("chunks.lance/_versions")))
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .len(),
            10_000
        );
        assert_eq!(
            store.get(&OPath::from(new)).await.unwrap().bytes().await.unwrap(),
            "new"
        );
    }

    #[tokio::test]
    async fn sharded_hint_replaces_flat_hint_and_snapshot_inventory_is_cached() {
        let backend = Arc::new(InMemory::new());
        let hint = "chunks.lance/_versions/latest_version_hint.json";
        put(&backend, hint, b"old").await;
        put(&backend, &physical_path(hint).unwrap(), b"new").await;
        let store = ShardStore::new(backend.clone());
        assert_eq!(
            store.get(&OPath::from(hint)).await.unwrap().bytes().await.unwrap(),
            "new"
        );
        // Direct get does not populate the inventory cache.
        assert!(store.inventory.get().is_none());
        assert_eq!(store.inventory().await.unwrap().len(), 1);
        assert_eq!(store.inventory.get().unwrap().len(), 1);
        // New objects belong to a new snapshot/store, not this immutable inventory.
        let later = "chunks.lance/data/later.lance";
        put(&backend, &physical_path(later).unwrap(), b"later").await;
        assert_eq!(store.inventory().await.unwrap().len(), 1);
        let reopened = ShardStore::new(backend);
        assert_eq!(
            reopened.get(&OPath::from(later)).await.unwrap().bytes().await.unwrap(),
            "later"
        );
    }

    #[tokio::test]
    async fn all_mutations_are_rejected_and_flat_bytes_are_unchanged() {
        let backend = Arc::new(InMemory::new());
        let path = OPath::from("chunks.lance/data/old.lance");
        put(&backend, path.as_ref(), b"original").await;
        let store = ShardStore::new(backend.clone());
        assert!(store.put(&path, Bytes::from_static(b"changed").into()).await.is_err());
        assert!(store.put_multipart(&path).await.is_err());
        assert!(store.copy(&path, &OPath::from("copy")).await.is_err());
        assert!(store.delete(&path).await.is_err());
        assert_eq!(backend.get(&path).await.unwrap().bytes().await.unwrap(), "original");
    }

    #[derive(Debug)]
    struct FailingInventory {
        inner: Arc<InMemory>,
        failure_prefix: OPath,
        not_found: bool,
    }

    impl std::fmt::Display for FailingInventory {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FailingInventory")
        }
    }

    #[async_trait]
    impl OSObjectStore for FailingInventory {
        async fn get_opts(&self, location: &OPath, options: GetOptions) -> OSResult<GetResult> {
            self.inner.get_opts(location, options).await
        }

        fn list(&self, prefix: Option<&OPath>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            if prefix == Some(&self.failure_prefix) {
                let error = if self.not_found {
                    object_store::Error::NotFound {
                        path: SHARD_ROOT.to_string(),
                        source: Box::new(std::io::Error::other("missing shard directory")),
                    }
                } else {
                    object_store::Error::Generic {
                        store: "inventory-test",
                        source: Box::new(std::io::Error::other("temporary or authorization failure")),
                    }
                };
                stream::once(async move { Err(error) }).boxed()
            } else {
                self.inner.list(prefix)
            }
        }

        async fn list_with_delimiter(&self, prefix: Option<&OPath>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn put_opts(&self, _: &OPath, _: PutPayload, _: PutOptions) -> OSResult<PutResult> {
            Err(read_only())
        }

        async fn put_multipart_opts(&self, _: &OPath, _: PutMultipartOptions) -> OSResult<Box<dyn MultipartUpload>> {
            Err(read_only())
        }

        fn delete_stream(&self, locations: BoxStream<'static, OSResult<OPath>>) -> BoxStream<'static, OSResult<OPath>> {
            locations
                .map(|location| location.and_then(|_| Err(read_only())))
                .boxed()
        }

        async fn copy_opts(&self, _: &OPath, _: &OPath, _: CopyOptions) -> OSResult<()> {
            Err(read_only())
        }
    }

    #[tokio::test]
    async fn absent_inventory_is_empty_but_auth_or_transient_errors_propagate_without_caching() {
        let inner = Arc::new(InMemory::new());
        let path = OPath::from("chunks.lance/data/old.lance");
        put(&inner, path.as_ref(), b"original").await;
        let missing = ShardStore::new(Arc::new(FailingInventory {
            inner: inner.clone(),
            failure_prefix: OPath::from(SHARD_ROOT),
            not_found: true,
        }));
        assert_eq!(missing.get(&path).await.unwrap().bytes().await.unwrap(), "original");
        let failed = ShardStore::new(Arc::new(FailingInventory {
            inner,
            failure_prefix: OPath::from(SHARD_ROOT),
            not_found: false,
        }));
        // Direct get only probes physical and falls back to flat; it does not list SHARD_ROOT,
        // so shard listing failure does not break direct get of an existing flat object.
        assert_eq!(failed.get(&path).await.unwrap().bytes().await.unwrap(), "original");
        assert!(failed.inventory.get().is_none());
        // Operations that do inspect shard inventory still propagate transient errors without caching.
        assert!(failed.inventory().await.is_err());
        assert!(failed.inventory.get().is_none());
        assert!(failed.list(None).try_collect::<Vec<_>>().await.is_err());
        assert!(failed.inventory.get().is_none());
    }

    #[tokio::test]
    async fn shard_only_prefix_is_listed_when_the_flat_directory_does_not_exist() {
        let inner = Arc::new(InMemory::new());
        let prefix = OPath::from("chunks.lance/_indices/new-index");
        let logical = "chunks.lance/_indices/new-index/index.idx";
        put(&inner, &physical_path(logical).unwrap(), b"index").await;
        let store = ShardStore::new(Arc::new(FailingInventory {
            inner,
            failure_prefix: prefix.clone(),
            not_found: true,
        }));
        let listed = store.list(Some(&prefix)).try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].location, OPath::from(logical));
        assert_eq!(store.list_with_delimiter(Some(&prefix)).await.unwrap().objects.len(), 1);
    }

    #[derive(Debug, Default)]
    struct CallCounts {
        list: AtomicUsize,
        list_with_delimiter: AtomicUsize,
        get_opts: AtomicUsize,
    }

    #[derive(Debug)]
    struct MockCountingAndFaultStore {
        inner: Arc<InMemory>,
        counts: Arc<CallCounts>,
        fault_path: Option<OPath>,
        fault_error_msg: Option<&'static str>,
    }

    impl MockCountingAndFaultStore {
        fn new(inner: Arc<InMemory>, counts: Arc<CallCounts>) -> Self {
            Self {
                inner,
                counts,
                fault_path: None,
                fault_error_msg: None,
            }
        }

        fn with_fault(inner: Arc<InMemory>, counts: Arc<CallCounts>, fault_path: OPath, msg: &'static str) -> Self {
            Self {
                inner,
                counts,
                fault_path: Some(fault_path),
                fault_error_msg: Some(msg),
            }
        }
    }

    impl std::fmt::Display for MockCountingAndFaultStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockCountingAndFaultStore")
        }
    }

    #[async_trait]
    impl OSObjectStore for MockCountingAndFaultStore {
        async fn get_opts(&self, location: &OPath, options: GetOptions) -> OSResult<GetResult> {
            self.counts.get_opts.fetch_add(1, Ordering::SeqCst);
            if let (Some(fault_path), Some(msg)) = (&self.fault_path, self.fault_error_msg) {
                if location == fault_path {
                    return Err(object_store::Error::Generic {
                        store: "mock-backend",
                        source: Box::new(std::io::Error::other(msg)),
                    });
                }
            }
            self.inner.get_opts(location, options).await
        }

        fn list(&self, prefix: Option<&OPath>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.counts.list.fetch_add(1, Ordering::SeqCst);
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(&self, prefix: Option<&OPath>) -> OSResult<ListResult> {
            self.counts.list_with_delimiter.fetch_add(1, Ordering::SeqCst);
            self.inner.list_with_delimiter(prefix).await
        }

        async fn put_opts(&self, _: &OPath, _: PutPayload, _: PutOptions) -> OSResult<PutResult> {
            Err(read_only())
        }

        async fn put_multipart_opts(&self, _: &OPath, _: PutMultipartOptions) -> OSResult<Box<dyn MultipartUpload>> {
            Err(read_only())
        }

        fn delete_stream(&self, locations: BoxStream<'static, OSResult<OPath>>) -> BoxStream<'static, OSResult<OPath>> {
            locations
                .map(|location| location.and_then(|_| Err(read_only())))
                .boxed()
        }

        async fn copy_opts(&self, _: &OPath, _: &OPath, _: CopyOptions) -> OSResult<()> {
            Err(read_only())
        }
    }

    #[tokio::test]
    async fn get_head_range_never_call_list_and_sharded_wins_over_flat() {
        let inner = Arc::new(InMemory::new());
        let counts = Arc::new(CallCounts::default());

        let sharded_logical = "chunks.lance/data/sharded_only.lance";
        put(&inner, &physical_path(sharded_logical).unwrap(), b"sharded_content").await;

        let flat_logical = "chunks.lance/data/flat_only.lance";
        put(&inner, flat_logical, b"flat_content").await;

        let mixed_logical = "chunks.lance/_versions/latest_version_hint.json";
        put(&inner, mixed_logical, b"flat_hint").await;
        put(&inner, &physical_path(mixed_logical).unwrap(), b"sharded_hint").await;

        let metadata_logical = "README.md";
        put(&inner, metadata_logical, b"readme_content").await;

        let backend = Arc::new(MockCountingAndFaultStore::new(inner, counts.clone()));
        let store = ShardStore::new(backend);

        let res = store.get(&OPath::from(sharded_logical)).await.unwrap();
        assert_eq!(res.bytes().await.unwrap(), "sharded_content");
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);

        let res = store.get(&OPath::from(flat_logical)).await.unwrap();
        assert_eq!(res.bytes().await.unwrap(), "flat_content");
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);

        let res = store.get(&OPath::from(mixed_logical)).await.unwrap();
        assert_eq!(res.bytes().await.unwrap(), "sharded_hint");
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);

        let res = store.get(&OPath::from(metadata_logical)).await.unwrap();
        assert_eq!(res.bytes().await.unwrap(), "readme_content");
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);

        let head_sharded = store.head(&OPath::from(sharded_logical)).await.unwrap();
        assert_eq!(head_sharded.location, OPath::from(sharded_logical));
        let head_flat = store.head(&OPath::from(flat_logical)).await.unwrap();
        assert_eq!(head_flat.location, OPath::from(flat_logical));
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);

        let range_res = store
            .get_opts(
                &OPath::from(sharded_logical),
                GetOptions {
                    range: Some(GetRange::Bounded(0..7)),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(range_res.meta.location, OPath::from(sharded_logical));
        assert_eq!(range_res.bytes().await.unwrap(), "sharded");
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);

        let ranges = store
            .get_ranges(&OPath::from(flat_logical), &[0..4, 5..12])
            .await
            .unwrap();
        assert_eq!(
            ranges,
            vec![Bytes::from_static(b"flat"), Bytes::from_static(b"content")]
        );
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);

        assert!(store.inventory.get().is_none());

        let _ = store.list(None).try_collect::<Vec<_>>().await.unwrap();
        assert!(counts.list.load(Ordering::SeqCst) > 0);
        assert!(store.inventory.get().is_some());
    }

    #[tokio::test]
    async fn auth_or_transient_error_on_physical_does_not_fallback_to_flat() {
        let inner = Arc::new(InMemory::new());
        let counts = Arc::new(CallCounts::default());

        let logical = "chunks.lance/data/secret.lance";
        put(&inner, logical, b"flat_fallback_forbidden").await;

        let physical = physical_path(logical).unwrap();
        let backend = Arc::new(MockCountingAndFaultStore::with_fault(
            inner,
            counts.clone(),
            OPath::from(physical),
            "auth expired or transient network split",
        ));
        let store = ShardStore::new(backend);

        let err = store.get(&OPath::from(logical)).await.unwrap_err();
        match err {
            object_store::Error::Generic { source, .. } => {
                assert!(source.to_string().contains("auth expired or transient network split"));
            }
            other => panic!("expected Generic auth error, got: {other:?}"),
        }
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);

        let inner2 = Arc::new(InMemory::new());
        let counts2 = Arc::new(CallCounts::default());
        let flat_transient_logical = "chunks.lance/data/transient_flat.lance";
        let backend2 = Arc::new(MockCountingAndFaultStore::with_fault(
            inner2,
            counts2.clone(),
            OPath::from(flat_transient_logical),
            "flat storage connection reset",
        ));
        let store2 = ShardStore::new(backend2);

        let err2 = store2.get(&OPath::from(flat_transient_logical)).await.unwrap_err();
        match err2 {
            object_store::Error::Generic { source, .. } => {
                assert!(source.to_string().contains("flat storage connection reset"));
            }
            other => panic!("expected Generic flat error, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_objects_on_both_sides_return_not_found() {
        let inner = Arc::new(InMemory::new());
        let counts = Arc::new(CallCounts::default());
        let backend = Arc::new(MockCountingAndFaultStore::new(inner, counts.clone()));
        let store = ShardStore::new(backend);

        let missing_lance = OPath::from("chunks.lance/data/missing.lance");
        let err = store.get(&missing_lance).await.unwrap_err();
        assert!(matches!(err, object_store::Error::NotFound { .. }));

        let missing_other = OPath::from("missing.txt");
        let err_other = store.get(&missing_other).await.unwrap_err();
        assert!(matches!(err_other, object_store::Error::NotFound { .. }));

        assert_eq!(counts.list.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn stale_flat_hint_is_skipped_on_sharded_404_and_fresh_sharded_hint_wins() {
        let inner = Arc::new(InMemory::new());
        let counts = Arc::new(CallCounts::default());

        let hint_logical = "chunks.lance/_versions/latest_version_hint.json";
        // Flat store has stale hint from pre-sharded commits (e.g. version 4).
        put(&inner, hint_logical, br#"{"version":4}"#).await;

        let flat_manifest = "chunks.lance/_versions/4.manifest";
        put(&inner, flat_manifest, b"manifest_4").await;

        let sharded_manifest = "chunks.lance/_versions/12746.manifest";
        put(&inner, &physical_path(sharded_manifest).unwrap(), b"manifest_12746").await;

        let backend = Arc::new(MockCountingAndFaultStore::new(inner.clone(), counts.clone()));
        let store = ShardStore::new(backend);

        // When sharded hint is 404: returns NotFound immediately without falling back to flat hint.
        // This prevents Lance from attempting an unbounded sequential probe loop (4 -> 12746).
        let err = store.get(&OPath::from(hint_logical)).await.unwrap_err();
        assert!(matches!(err, object_store::Error::NotFound { .. }));
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);
        assert_eq!(counts.get_opts.load(Ordering::SeqCst), 1);

        let err_head = store.head(&OPath::from(hint_logical)).await.unwrap_err();
        assert!(matches!(err_head, object_store::Error::NotFound { .. }));
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);
        assert_eq!(counts.get_opts.load(Ordering::SeqCst), 2);

        // Meanwhile, other files (e.g. flat manifests) DO fall back to flat storage cleanly.
        let manifest_res = store.get(&OPath::from(flat_manifest)).await.unwrap();
        assert_eq!(manifest_res.bytes().await.unwrap(), "manifest_4");

        // Sharded manifests are read directly from physical shards without listing.
        let sharded_res = store.get(&OPath::from(sharded_manifest)).await.unwrap();
        assert_eq!(sharded_res.bytes().await.unwrap(), "manifest_12746");
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);

        // When fresh sharded hint is uploaded (e.g. version 12746):
        let physical = physical_path(hint_logical).unwrap();
        put(&inner, &physical, br#"{"version":12746}"#).await;

        counts.get_opts.store(0, Ordering::SeqCst);
        let res2 = store.get(&OPath::from(hint_logical)).await.unwrap();
        assert_eq!(res2.bytes().await.unwrap(), br#"{"version":12746}"#.as_slice());
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);
        assert_eq!(counts.get_opts.load(Ordering::SeqCst), 1);
    }

    use lance_io::object_store::WrappingObjectStore;

    #[derive(Debug)]
    struct ShardStoreWrapping(Arc<dyn OSObjectStore>);

    impl WrappingObjectStore for ShardStoreWrapping {
        fn wrap(&self, _prefix: &str, _original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
            Arc::new(ShardStore::new(self.0.clone()))
        }
    }

    #[derive(Debug)]
    struct PlainWrapping(Arc<dyn OSObjectStore>);

    impl WrappingObjectStore for PlainWrapping {
        fn wrap(&self, _prefix: &str, _original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
            self.0.clone()
        }
    }

    #[tokio::test]
    async fn real_lance_missing_stale_and_fresh_hint_regression() {
        use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use lance::dataset::builder::DatasetBuilder;
        use lance::dataset::Dataset;
        use lance::dataset::WriteParams;
        use lance_io::object_store::ObjectStoreParams;

        let inner = Arc::new(InMemory::new());
        // Exercise the same provider flags as production. Lance's memory:// provider ignores
        // list_is_lexically_ordered, so it cannot test the hint path. Both wrappers below
        // replace all backend I/O with InMemory; no Hub requests or credentials are needed.
        let uri = "hf://datasets/funes-tests/lance-hint/chunks.lance";

        let batch = |texts: Vec<&str>| {
            let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
            let rows =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from_iter_values(texts))]).unwrap();
            RecordBatchIterator::new([Ok(rows)], schema)
        };

        // Write version 1 with plain in-memory wrapper
        let mut ds = Dataset::write(
            batch(vec!["row_v1"]),
            uri,
            Some(WriteParams {
                // Production retains the legacy manifest paths; do not create V2 fixture names.
                enable_v2_manifest_paths: false,
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(Arc::new(PlainWrapping(inner.clone()))),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(ds.version().version, 1);

        // Append version 2
        ds.append(batch(vec!["row_v2"]), None).await.unwrap();

        // Move version 2 manifest to physical shard path, simulating historical flat v1 + sharded v2
        let objects = inner.list(None).try_collect::<Vec<_>>().await.unwrap();
        let v2_manifest_logical = objects
            .iter()
            .find(|object| object.location.as_ref().ends_with("/_versions/2.manifest"))
            .expect("legacy version 2 manifest must be present")
            .location
            .to_string();
        let table_prefix = v2_manifest_logical.strip_suffix("/_versions/2.manifest").unwrap();
        let v2_manifest_bytes = inner
            .get(&OPath::from(v2_manifest_logical.clone()))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        inner.delete(&OPath::from(v2_manifest_logical.clone())).await.unwrap();
        let v2_manifest_physical = physical_path(&v2_manifest_logical).unwrap();
        inner
            .put(&OPath::from(v2_manifest_physical), v2_manifest_bytes.into())
            .await
            .unwrap();

        let hint_logical = format!("{table_prefix}/_versions/latest_version_hint.json");
        let hint_physical = physical_path(&hint_logical).unwrap();

        // 1. Stale flat hint (v1) in place, while sharded hint is missing.
        // ShardStore skips the stale flat hint on 404, causing Lance to cleanly fall back
        // to authoritative listing and successfully open the true latest version (v2).
        inner
            .put(
                &OPath::from(hint_logical.clone()),
                Bytes::from_static(br#"{"version":1}"#).into(),
            )
            .await
            .unwrap();

        let counts = Arc::new(CallCounts::default());
        let counting_store = Arc::new(MockCountingAndFaultStore::new(inner.clone(), counts.clone()));

        let ds_stale_fallback = DatasetBuilder::from_uri(uri)
            .with_store_params(ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(ShardStoreWrapping(counting_store.clone()))),
                list_is_lexically_ordered: Some(false),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        assert_eq!(ds_stale_fallback.version().version, 2);
        assert_eq!(ds_stale_fallback.count_rows(None).await.unwrap(), 2);
        // Authoritative listing was triggered because stale flat hint was skipped
        assert!(counts.list.load(Ordering::SeqCst) > 0);

        // 2. Missing hint everywhere (neither flat nor sharded exists).
        // Lance falls back to authoritative listing and still opens latest v2.
        inner.delete(&OPath::from(hint_logical.clone())).await.unwrap();
        counts.list.store(0, Ordering::SeqCst);

        let ds_missing_hint = DatasetBuilder::from_uri(uri)
            .with_store_params(ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(ShardStoreWrapping(counting_store.clone()))),
                list_is_lexically_ordered: Some(false),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        assert_eq!(ds_missing_hint.version().version, 2);
        assert_eq!(ds_missing_hint.count_rows(None).await.unwrap(), 2);
        assert!(counts.list.load(Ordering::SeqCst) > 0);

        // 3. Fresh sharded hint (v2) written.
        // Lance uses the sharded hint directly, finding latest version 2 without listing!
        inner
            .put(
                &OPath::from(hint_physical),
                Bytes::from_static(br#"{"version":2}"#).into(),
            )
            .await
            .unwrap();
        counts.list.store(0, Ordering::SeqCst);

        let ds_fresh_hint = DatasetBuilder::from_uri(uri)
            .with_store_params(ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(ShardStoreWrapping(counting_store.clone()))),
                list_is_lexically_ordered: Some(false),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        assert_eq!(ds_fresh_hint.version().version, 2);
        assert_eq!(ds_fresh_hint.count_rows(None).await.unwrap(), 2);
        // With fresh sharded hint present, Lance resolves version without calling list()
        assert_eq!(counts.list.load(Ordering::SeqCst), 0);
    }
}
