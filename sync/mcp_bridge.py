from __future__ import annotations
import json,sys,os
from urllib import request
from .store import Store

def _remote_call(path, payload):
    base=os.environ.get("FUNES_REMOTE_URL", "").rstrip("/")
    token=os.environ.get("FUNES_API_TOKEN", "")
    if not base or not token:
        return None
    req=request.Request(base+path, data=json.dumps(payload,ensure_ascii=False).encode(), headers={"Content-Type":"application/json","Authorization":"Bearer "+token,"User-Agent":"funes-sync-mcp/1"}, method="POST")
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read() or b"{}")

def serve(store=None):
    store=store or Store()
    remote=bool(os.environ.get("FUNES_REMOTE_URL") and os.environ.get("FUNES_API_TOKEN"))
    for line in sys.stdin:
        try:
            msg=json.loads(line); method=msg.get("method"); ident=msg.get("id"); p=msg.get("params") or {}
            if method=="initialize": result={"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"funes-sync","version":"1"}}
            elif method=="notifications/initialized": continue
            elif method=="tools/list": result={"tools":[{"name":"recall","description":"Search the unified Codex, Pi and Claude memory; return original raw context","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},"source_agent":{"type":"string"},"project":{"type":"string"}} ,"required":["query"]}},{"name":"get","description":"Get one original memory record","inputSchema":{"type":"object","properties":{"record_id":{"type":"string"}},"required":["record_id"]}},{"name":"status","description":"Show unified memory sync status","inputSchema":{"type":"object","properties":{}}}]}
            elif method=="tools/call":
                name=p.get("name"); args=p.get("arguments") or {}
                if remote:
                    if name=="recall": val=_remote_call("/search", args)
                    elif name=="get": val=_remote_call("/get", {"id":args.get("record_id","")})
                    elif name=="status": val=_remote_call("/sync/status", {})
                    else: val={"error":"unknown_tool"}
                else:
                    val=store.search(args.get("query", "")) if name=="recall" else store.get(args.get("record_id", ""))
                result={"content":[{"type":"text","text":json.dumps(val,ensure_ascii=False)}]}
            else: result={}
            if ident is not None: print(json.dumps({"jsonrpc":"2.0","id":ident,"result":result}),flush=True)
        except Exception as e:
            if 'ident' in locals() and ident is not None: print(json.dumps({"jsonrpc":"2.0","id":ident,"error":{"code":-32000,"message":str(e)}}),flush=True)
