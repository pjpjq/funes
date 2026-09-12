from __future__ import annotations
import json, os, time
from urllib import request, error
from .config import Config

class SyncClient:
    def __init__(self, config: Config|None=None): self.config=config or Config.load()
    def ingest(self, records:list[dict]):
        if not records: return {"accepted":0}
        url=self.config.remote_url.rstrip("/")+"/ingest"
        # The native Space bridge and the migration compatibility service both
        # accept this canonical envelope; durable=true is mandatory in either
        # implementation.
        body=json.dumps({"device_id":self.config.device_id,"documents":records}, ensure_ascii=False).encode()
        headers={"Content-Type":"application/json","User-Agent":"funes-sync/1"}
        token=os.environ.get("FUNES_API_TOKEN")
        if not token:
            raise RuntimeError("FUNES_API_TOKEN is not configured")
        hub_token = os.environ.get("FUNES_HF_TOKEN") or os.environ.get("HF_TOKEN")
        if hub_token:
            headers["Authorization"] = "Bearer " + hub_token
            headers["X-Funes-Authorization"] = "Bearer " + token
        else:
            headers["Authorization"] = "Bearer " + token
        last = None
        for attempt in range(4):
            req=request.Request(url,data=body,headers=headers,method="POST")
            try:
                with request.urlopen(req,timeout=30) as r:
                    raw=r.read()
                    result = json.loads(raw) if raw else {"accepted": len(records), "durable": True}
                    # The daemon may acknowledge a local queue row only after
                    # the remote confirms a durable commit.  A 2xx response
                    # without that contract is treated as retryable rather
                    # than silently losing the pending record.
                    if result.get("durable") is not True:
                        raise RuntimeError("remote did not confirm a durable commit")
                    accepted = int(result.get("accepted", result.get("created", 0) + result.get("updated", 0) + result.get("deduped", 0)))
                    if accepted < len(records):
                        raise RuntimeError(f"remote accepted {accepted}/{len(records)} records")
                    return result
            except error.HTTPError as exc:
                last = exc
                if exc.code not in (408, 425, 429) and exc.code < 500:
                    raise
            except (error.URLError, TimeoutError, OSError) as exc:
                last = exc
            if attempt < 3:
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"remote ingest failed after retries: {type(last).__name__}") from last

    def health(self) -> bool:
        try:
            with request.urlopen(self.config.remote_url.rstrip("/") + "/ready", timeout=8) as r:
                return 200 <= r.status < 300
        except (OSError, error.URLError):
            return False

    def sync_snapshot(self) -> dict:
        token = os.environ.get("FUNES_API_TOKEN")
        if not token:
            raise RuntimeError("FUNES_API_TOKEN is not configured")
        headers={"Content-Type":"application/json"}
        hub_token=os.environ.get("FUNES_HF_TOKEN") or os.environ.get("HF_TOKEN")
        if hub_token:
            headers["Authorization"]="Bearer "+hub_token
            headers["X-Funes-Authorization"]="Bearer "+token
        else:
            headers["Authorization"]="Bearer "+token
        req=request.Request(self.config.remote_url.rstrip("/")+"/sync",data=b"{}",headers=headers,method="POST")
        with request.urlopen(req,timeout=120) as r:
            raw=r.read(); result=json.loads(raw) if raw else {}
            if result.get("durable") is not True:
                raise RuntimeError("remote snapshot sync did not confirm a durable commit")
            return result
