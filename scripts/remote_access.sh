#!/bin/bash
# Reach the Veystrix API on this Mac from your other devices over Tailscale.
#
#   scripts/remote_access.sh on [PORT]   share http://127.0.0.1:PORT (default 8080) inside your tailnet, over HTTPS
#   scripts/remote_access.sh off         stop sharing
#   scripts/remote_access.sh status      show the address and whether the engine answers through it
#
# How it works: Tailscale Serve on this Mac accepts HTTPS from devices signed in to YOUR tailnet and forwards it to
# the engine on the loopback address. The engine itself still listens on 127.0.0.1 only, so nothing on the local
# network or the internet can reach it. This script never uses Tailscale Funnel (which would publish to the internet).
# The setting is stored by Tailscale and survives restarts of the Mac, of Tailscale and of the app.
set -euo pipefail

TS="$(command -v tailscale || true)"
[[ -n "$TS" ]] || TS="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
[[ -x "$TS" ]] || { echo "Tailscale is not installed (https://tailscale.com/download)." >&2; exit 1; }

address() {   # this Mac's tailnet name, without the trailing dot
  "$TS" status --json | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))'
}

case "${1:-status}" in
  on)
    port="${2:-${VEYSTRIX_PORT:-8080}}"
    [[ "$port" =~ ^[0-9]+$ ]] || { echo "PORT must be a number" >&2; exit 2; }
    "$TS" serve --bg --https=443 "http://127.0.0.1:$port" >/dev/null
    echo "Sharing 127.0.0.1:$port inside your tailnet."
    echo "Base URL for clients:  https://$(address)/v1"
    echo "The engine must be running on this Mac (press Start in Veystrix), and the Mac must be awake."
    ;;
  off)
    "$TS" serve --https=443 off >/dev/null
    echo "Stopped sharing. The engine is reachable from this Mac only."
    ;;
  status)
    "$TS" serve status
    name="$(address)"     # the first request after "on" waits for Tailscale to issue the certificate (about 25 s)
    echo "Base URL for clients:  https://$name/v1"
    code="$(/usr/bin/curl -s -o /dev/null -m 40 -w '%{http_code}' "https://$name/health" || true)"
    case "$code" in
      200) echo "Engine: answering through Tailscale.";;
      502) echo "Engine: not running (start it in the Veystrix app). Sharing is set up.";;
      401|403) echo "Engine: running and asking for the API key.";;
      *) echo "Engine: no answer through Tailscale (HTTP ${code:-none}). Is sharing on? Run: $0 on";;
    esac
    ;;
  *)
    sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
