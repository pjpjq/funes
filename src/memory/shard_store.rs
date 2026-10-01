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

    async fn resolve(&self, logical: &OPath) -> OSResult<OPath> {
        let Some(physical) = physical_path(logical.as_ref()) else {
            return Ok(logical.clone());
        };
        if self.inventory().await?.contains_key(logical) {
            Ok(OPath::from(physical))
        } else {
            Ok(logical.clone())
        }
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
        let physical = self.resolve(location).await?;
        let mut result = self.inner.get_opts(&physical, options).await?;
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
        assert!(failed.get(&path).await.is_err());
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
}
