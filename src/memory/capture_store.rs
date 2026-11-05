//! A write-capturing object-store decorator.
//!
//! [`CaptureStore`] wraps an inner [`ObjectStore`](object_store::ObjectStore): it records every
//! write in temporary files and delete in memory instead of forwarding it, and delegates reads to the inner store,
//! except for a path it has already captured, which it serves back (read-your-writes). A caller can
//! run a sequence of writes against it and then recover exactly the files that would have been
//! written or deleted, keyed by path, with nothing reaching the backend.
//!
//! ```text
//!   put    → captured on local disk  (never reaches the inner store)
//!   delete → captured in memory      (never reaches the inner store)
//!   get    → captured if present, else delegated to the inner store
//! ```
//!
//! It is generic: the inner store is any `ObjectStore`, so the tests exercise it over an in-memory
//! backend, and what to do with the captured files is left to the caller.
//!
//! **A decorator, because Rust has no inheritance.** You can't subclass a store and override
//! `put`; the idiomatic stand-in is to hold the `inner` store, forward every method unchanged, and
//! intercept the ones that matter. The cost is the hand-written delegation of each `ObjectStore`
//! method below — Rust has no auto-delegation.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
#[cfg(test)]
use bytes::Bytes;
use chrono::Utc;
use futures::stream::{self, BoxStream, StreamExt};
use object_store::path::Path as OPath;
use object_store::{
    CopyOptions, Error as OSError, Extensions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OSResult,
    UploadPart,
};

/// Capture of object-store changes: path → a temporary file, or `None` for a delete.
/// Shared between a [`CaptureStore`] and whatever holds it, so the changes can be read back after
/// the fact.
pub(crate) type Captured = Arc<Mutex<BTreeMap<OPath, Option<CapturedFile>>>>;

/// Closed temporary files keep both RAM and open descriptors independent of captured byte volume.
#[derive(Clone, Debug)]
pub(crate) struct CapturedFile {
    path: Arc<tempfile::TempPath>,
    len: u64,
}

impl CapturedFile {
    pub(crate) fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(bytes)?;
        Self::finish(file)
    }

    fn finish(mut file: tempfile::NamedTempFile) -> std::io::Result<Self> {
        file.flush()?;
        let len = file.as_file().metadata()?.len();
        Ok(Self {
            path: Arc::new(file.into_temp_path()),
            len,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        self.path.as_ref().as_ref()
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    #[cfg(test)]
    pub(crate) fn bytes(&self) -> Bytes {
        Bytes::from(std::fs::read(self.path()).unwrap())
    }
}

fn capture_io(error: std::io::Error) -> OSError {
    OSError::Generic {
        store: "CaptureStore",
        source: Box::new(error),
    }
}

/// Reads delegate to `inner` unless the path was already captured (read-your-writes, so a caller
/// can read back what it just wrote). Writes and deletes are captured and never forwarded.
#[derive(Debug)]
pub(crate) struct CaptureStore {
    inner: Arc<dyn OSObjectStore>,
    captured: Captured,
}

impl CaptureStore {
    /// Decorate `inner`, recording writes into the shared `captured` map.
    pub(crate) fn new(inner: Arc<dyn OSObjectStore>, captured: Captured) -> Self {
        Self { inner, captured }
    }
}

impl std::fmt::Display for CaptureStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CaptureStore({})", self.inner)
    }
}

#[async_trait]
impl OSObjectStore for CaptureStore {
    async fn put_opts(&self, location: &OPath, payload: PutPayload, _opts: PutOptions) -> OSResult<PutResult> {
        let mut file = tempfile::NamedTempFile::new().map_err(capture_io)?;
        for b in payload {
            file.write_all(&b).map_err(capture_io)?;
        }
        let body = CapturedFile::finish(file).map_err(capture_io)?;
        self.captured.lock().unwrap().insert(location.clone(), Some(body));
        Ok(PutResult {
            e_tag: None,
            version: None,
            extensions: Extensions::default(),
        })
    }

    async fn put_multipart_opts(
        &self,
        location: &OPath,
        _opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        Ok(Box::new(CaptureMultipart {
            location: location.clone(),
            file: Some(tempfile::NamedTempFile::new().map_err(capture_io)?),
            failed: false,
            captured: self.captured.clone(),
        }))
    }

    async fn get_opts(&self, location: &OPath, options: GetOptions) -> OSResult<GetResult> {
        let hit = self.captured.lock().unwrap().get(location).cloned();
        match hit {
            Some(None) => Err(deleted(location)),
            Some(Some(full)) => super::fetch_store::serve_from_file(full.path(), location, &options)
                .map_err(|_| capture_io(std::io::Error::other("reading captured file failed"))),
            None => self.inner.get_opts(location, options).await,
        }
    }

    fn list(&self, prefix: Option<&OPath>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let captured = self.captured.lock().unwrap().clone();
        let prefix = prefix.cloned();
        let extra: Vec<OSResult<ObjectMeta>> = captured
            .iter()
            .filter(|(p, _)| prefix.as_ref().is_none_or(|pre| p.as_ref().starts_with(pre.as_ref())))
            .filter_map(|(p, b)| b.as_ref().map(|b| Ok(meta(p.clone(), b.len()))))
            .collect();
        let inner = self
            .inner
            .list(prefix.as_ref())
            .filter(move |m| std::future::ready(m.as_ref().map_or(true, |m| !captured.contains_key(&m.location))));
        inner.chain(stream::iter(extra)).boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&OPath>) -> OSResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    /// A path written through this store is dropped rather than recorded: Lance writes each file
    /// once, so it did not exist before.
    fn delete_stream(&self, locations: BoxStream<'static, OSResult<OPath>>) -> BoxStream<'static, OSResult<OPath>> {
        let captured = self.captured.clone();
        locations
            .map(move |loc| {
                if let Ok(p) = &loc {
                    let mut captured = captured.lock().unwrap();
                    if let Some(Some(_)) = captured.get(p) {
                        captured.remove(p);
                    } else {
                        captured.insert(p.clone(), None);
                    }
                }
                loc
            })
            .boxed()
    }

    async fn copy_opts(&self, from: &OPath, to: &OPath, _opts: CopyOptions) -> OSResult<()> {
        // The decorator never writes to the underlying store — a copy lands in the capture. The
        // source comes from the capture if present, else a read of the underlying store.
        let hit = self.captured.lock().unwrap().get(from).cloned();
        let body = match hit {
            Some(Some(b)) => b,
            Some(None) => return Err(deleted(from)),
            None => {
                let mut file = tempfile::NamedTempFile::new().map_err(capture_io)?;
                let mut source = self.inner.get_opts(from, GetOptions::default()).await?.into_stream();
                while let Some(part) = source.next().await {
                    file.write_all(&part?).map_err(capture_io)?;
                }
                CapturedFile::finish(file).map_err(capture_io)?
            }
        };
        self.captured.lock().unwrap().insert(to.clone(), Some(body));
        Ok(())
    }
}

fn deleted(location: &OPath) -> OSError {
    OSError::NotFound {
        path: location.to_string(),
        source: "deleted".into(),
    }
}

fn meta(location: OPath, size: u64) -> ObjectMeta {
    ObjectMeta {
        location,
        last_modified: Utc::now(),
        size,
        e_tag: None,
        version: None,
    }
}

/// A captured multipart upload: parts are spooled and stored under `location` on `complete`.
struct CaptureMultipart {
    location: OPath,
    file: Option<tempfile::NamedTempFile>,
    failed: bool,
    captured: Captured,
}

#[async_trait]
impl MultipartUpload for CaptureMultipart {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        // Preserve part call order even when the returned futures are awaited concurrently.
        let result = (|| {
            let file = self
                .file
                .as_mut()
                .ok_or_else(|| std::io::Error::other("multipart upload finished"))?;
            if self.failed {
                return Err(std::io::Error::other("multipart upload failed"));
            }
            for b in data {
                file.write_all(&b)?;
            }
            Ok(())
        })();
        self.failed |= result.is_err();
        Box::pin(async move { result.map_err(capture_io) })
    }

    async fn complete(&mut self) -> OSResult<PutResult> {
        if self.failed {
            drop(self.file.take());
            return Err(capture_io(std::io::Error::other("multipart upload failed")));
        }
        let file = self
            .file
            .take()
            .ok_or_else(|| capture_io(std::io::Error::other("multipart upload finished")))?;
        let body = CapturedFile::finish(file).map_err(capture_io)?;
        self.captured.lock().unwrap().insert(self.location.clone(), Some(body));
        Ok(PutResult {
            e_tag: None,
            version: None,
            extensions: Extensions::default(),
        })
    }

    async fn abort(&mut self) -> OSResult<()> {
        drop(self.file.take());
        Ok(())
    }
}

impl std::fmt::Debug for CaptureMultipart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CaptureMultipart({})", self.location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn capture_over_memory() -> (CaptureStore, Arc<InMemory>) {
        let inner = Arc::new(InMemory::new());
        let store = CaptureStore::new(inner.clone(), Captured::default());
        (store, inner)
    }

    /// The load-bearing invariant: a write is captured and never reaches the underlying store —
    /// that is what stops the backend from issuing a per-file commit.
    #[tokio::test]
    async fn put_is_captured_not_forwarded() {
        let (store, inner) = capture_over_memory();
        let p = OPath::from("data/x.lance");
        store
            .put_opts(&p, PutPayload::from("frag"), PutOptions::default())
            .await
            .unwrap();
        assert!(
            store.captured.lock().unwrap().contains_key(&p),
            "write must be captured"
        );
        assert!(
            inner.get_opts(&p, GetOptions::default()).await.is_err(),
            "write must NOT reach the underlying store"
        );
    }

    /// Reads serve the capture first (so a caller can read back what it just wrote), then fall
    /// through to the underlying store.
    #[tokio::test]
    async fn reads_capture_first_then_delegates() {
        let (store, inner) = capture_over_memory();
        let m = OPath::from("_versions/2.manifest");
        store
            .put_opts(&m, PutPayload::from("new"), PutOptions::default())
            .await
            .unwrap();
        let got = store
            .get_opts(&m, GetOptions::default())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(got, Bytes::from_static(b"new"), "read-your-writes");

        let base = OPath::from("data/base.lance");
        inner
            .put_opts(&base, PutPayload::from("base"), PutOptions::default())
            .await
            .unwrap();
        let got = store
            .get_opts(&base, GetOptions::default())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(got, Bytes::from_static(b"base"), "delegates to the underlying store");
    }

    /// `copy` must land in the capture, never in the underlying store (a fallback that delegated to
    /// `inner` would trigger a stray backend write).
    #[tokio::test]
    async fn copy_lands_in_capture_never_underlying() {
        let (store, inner) = capture_over_memory();
        let from = OPath::from("data/a.lance");
        let to = OPath::from("data/b.lance");
        store
            .put_opts(&from, PutPayload::from("frag"), PutOptions::default())
            .await
            .unwrap();
        store.copy_opts(&from, &to, CopyOptions::default()).await.unwrap();
        assert!(store.captured.lock().unwrap().contains_key(&to));
        assert!(
            inner.get_opts(&to, GetOptions::default()).await.is_err(),
            "copy must not write to the underlying store"
        );
    }

    /// A delete of an underlying file is captured, never forwarded, and hides the file from reads.
    /// A delete of a file written through the store drops the write.
    #[tokio::test]
    async fn delete_is_captured_not_forwarded() {
        let (store, inner) = capture_over_memory();
        let old = OPath::from("_versions/1.manifest");
        inner
            .put_opts(&old, PutPayload::from("old"), PutOptions::default())
            .await
            .unwrap();
        let new = OPath::from("data/new.lance");
        store
            .put_opts(&new, PutPayload::from("new"), PutOptions::default())
            .await
            .unwrap();

        let paths = stream::iter([Ok(old.clone()), Ok(new.clone())]).boxed();
        let deleted: Vec<_> = store.delete_stream(paths).collect().await;
        assert!(deleted.iter().all(Result::is_ok));

        {
            let captured = store.captured.lock().unwrap();
            assert_eq!(captured.len(), 1, "the captured write is dropped");
            assert!(
                captured.get(&old).is_some_and(Option::is_none),
                "the underlying delete is recorded"
            );
        }
        assert!(
            inner.get_opts(&old, GetOptions::default()).await.is_ok(),
            "delete must NOT reach the underlying store"
        );
        assert!(store.get_opts(&old, GetOptions::default()).await.is_err());
        let listed: Vec<_> = store.list(None).collect().await;
        assert!(listed.is_empty(), "a deleted file is not listed: {listed:?}");
        let copy = OPath::from("_versions/2.manifest");
        assert!(store.copy_opts(&old, &copy, CopyOptions::default()).await.is_err());
        assert!(
            !store.captured.lock().unwrap().contains_key(&copy),
            "a deleted file is not copied"
        );
    }

    /// `list` shows both the underlying files and the captured ones, so a caller sees the version
    /// it just wrote alongside the existing ones.
    #[tokio::test]
    async fn list_merges_underlying_and_captured() {
        let (store, inner) = capture_over_memory();
        inner
            .put_opts(
                &OPath::from("data/base.lance"),
                PutPayload::from("b"),
                PutOptions::default(),
            )
            .await
            .unwrap();
        store
            .put_opts(
                &OPath::from("_versions/2.manifest"),
                PutPayload::from("m"),
                PutOptions::default(),
            )
            .await
            .unwrap();
        let names: std::collections::HashSet<String> = store
            .list(None)
            .map(|r| r.unwrap().location.to_string())
            .collect()
            .await;
        assert!(names.contains("data/base.lance"), "underlying file listed");
        assert!(names.contains("_versions/2.manifest"), "captured file listed");
    }

    #[tokio::test]
    async fn multipart_spools_in_call_order_and_preserves_ranges() {
        let (store, inner) = capture_over_memory();
        let name = OPath::from("data/multipart.lance");
        let mut upload = store
            .put_multipart_opts(&name, PutMultipartOptions::default())
            .await
            .unwrap();
        let first = upload.put_part(PutPayload::from_static(b"first"));
        let second = upload.put_part(PutPayload::from_static(b"second"));
        second.await.unwrap();
        first.await.unwrap();
        assert!(!store.captured.lock().unwrap().contains_key(&name));
        upload.complete().await.unwrap();
        assert!(upload.put_part(PutPayload::from_static(b"late")).await.is_err());
        assert_eq!(
            store
                .get_opts(&name, GetOptions::default())
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            "firstsecond"
        );
        let range = GetOptions {
            range: Some(object_store::GetRange::Bounded(3..8)),
            ..Default::default()
        };
        assert_eq!(
            store.get_opts(&name, range).await.unwrap().bytes().await.unwrap(),
            "stsec"
        );
        let head = store
            .get_opts(
                &name,
                GetOptions {
                    head: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(head.meta.size, 11);
        assert!(head.bytes().await.unwrap().is_empty());
        assert!(inner.get_opts(&name, GetOptions::default()).await.is_err());
        let path = store
            .captured
            .lock()
            .unwrap()
            .get(&name)
            .unwrap()
            .as_ref()
            .unwrap()
            .path()
            .to_path_buf();
        drop(upload);
        drop(store);
        assert!(!path.exists(), "dropping the last capture removes its scratch file");
    }

    #[tokio::test]
    async fn multipart_abort_and_drop_remove_uncommitted_files() {
        let captured = Captured::default();
        for abort in [false, true] {
            let file = tempfile::NamedTempFile::new().unwrap();
            let path = file.path().to_path_buf();
            let mut upload = CaptureMultipart {
                location: OPath::from("data/aborted.lance"),
                file: Some(file),
                failed: false,
                captured: captured.clone(),
            };
            upload.put_part(PutPayload::from_static(b"pending")).await.unwrap();
            if abort {
                upload.abort().await.unwrap();
                assert!(!path.exists());
                assert!(upload.complete().await.is_err());
            }
            drop(upload);
            assert!(!path.exists());
        }
        assert!(captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn file_capture_copies_inner_and_keeps_open_reads_valid_after_delete() {
        let (store, inner) = capture_over_memory();
        let from = OPath::from("base");
        let to = OPath::from("copy");
        inner
            .put_opts(&from, PutPayload::from_static(b"copied"), PutOptions::default())
            .await
            .unwrap();
        store.copy_opts(&from, &to, CopyOptions::default()).await.unwrap();
        let read = store.get_opts(&to, GetOptions::default()).await.unwrap();
        let deleted: Vec<_> = store.delete_stream(stream::iter([Ok(to)]).boxed()).collect().await;
        assert!(deleted.iter().all(Result::is_ok));
        assert_eq!(read.bytes().await.unwrap(), "copied");
        let empty = OPath::from("empty");
        store
            .put_opts(&empty, PutPayload::from_static(b""), PutOptions::default())
            .await
            .unwrap();
        assert!(store
            .get_opts(&empty, GetOptions::default())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn failed_spool_write_preserves_previous_capture_and_cleans_partial_file() {
        let captured = Captured::default();
        let location = OPath::from("data/failed.lance");
        captured
            .lock()
            .unwrap()
            .insert(location.clone(), Some(CapturedFile::from_bytes(b"previous").unwrap()));
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        // A read-only descriptor injects a real write error without filling the host disk.
        *file.as_file_mut() = std::fs::File::open(&path).unwrap();
        let mut upload = CaptureMultipart {
            location: location.clone(),
            file: Some(file),
            failed: false,
            captured: captured.clone(),
        };
        assert!(upload.put_part(PutPayload::from_static(b"partial")).await.is_err());
        assert!(upload.complete().await.is_err());
        assert!(!path.exists());
        assert_eq!(
            captured.lock().unwrap()[&location].as_ref().unwrap().bytes(),
            "previous"
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    #[ignore = "writes 2 GiB of scratch data; run separately on the build host"]
    async fn disk_spooling_keeps_anonymous_memory_bounded() {
        let (store, inner) = capture_over_memory();
        let chunk = Bytes::from(vec![0x5a; 8 * 1024 * 1024]);
        eprintln!("CAPTURE_PRESSURE_START");
        for file in 0..4 {
            let name = OPath::from(format!("data/pressure-{file}.lance"));
            let mut upload = store
                .put_multipart_opts(&name, PutMultipartOptions::default())
                .await
                .unwrap();
            for _ in 0..64 {
                upload.put_part(PutPayload::from(chunk.clone())).await.unwrap();
            }
            upload.complete().await.unwrap();
            let range = GetOptions {
                range: Some(object_store::GetRange::Bounded(
                    512 * 1024 * 1024 - 8..512 * 1024 * 1024,
                )),
                ..Default::default()
            };
            assert_eq!(
                store
                    .get_opts(&name, range)
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap()
                    .as_ref(),
                &[0x5a; 8]
            );
        }
        let files: Vec<_> = store.list(None).collect().await;
        assert_eq!(
            files.iter().map(|item| item.as_ref().unwrap().size).sum::<u64>(),
            2 * 1024 * 1024 * 1024
        );
        assert!(inner.list(None).collect::<Vec<_>>().await.is_empty());
        // The trusted parent samples RssAnon; the isolated test cannot see host /proc.
        eprintln!("CAPTURE_PRESSURE_LOADED captured_bytes=2147483648");
        std::thread::sleep(std::time::Duration::from_secs(3));
    }
}
