# Using Veystrix from another device over Tailscale

The engine listens on `127.0.0.1` only, and that does not change. To reach it from your other devices (a home PC, a phone, a server), Tailscale Serve on the Mac forwards HTTPS requests from devices in **your tailnet** to the engine. Nothing is opened on the local network or the internet.

## On the Mac that runs Veystrix (once)

```sh
scripts/remote_access.sh on        # share port 8080; pass another port if you changed it in Settings
scripts/remote_access.sh status    # shows the address and whether the engine answers
scripts/remote_access.sh off       # stop sharing
```

Tailscale stores the setting. It survives restarts of the Mac, of Tailscale and of the app.

The address has the form `https://<mac-name>.<tailnet>.ts.net`. `status` prints it. The certificate is issued by Tailscale for that name, so clients need no extra trust settings.

Three things must be true for a remote request to work:

1. The engine is running: press **Start** in the Veystrix app (or run `./serve.sh`). Sharing does not start a model.
2. The Mac is awake. With "Keep the Mac awake while the engine runs" on, it stays awake while the engine runs. On battery with the lid closed it sleeps.
3. The other device is signed in to the same Tailscale account and Tailscale is connected.

## On the other device

Any OpenAI- or Anthropic-compatible client works. Use the Tailscale address as the base URL:

| Setting | Value |
|---|---|
| Base URL (OpenAI style) | `https://<mac-name>.<tailnet>.ts.net/v1` |
| Base URL (Anthropic style) | `https://<mac-name>.<tailnet>.ts.net` |
| API key | anything, unless "Require an API key" is on in Settings; then the key shown there |
| Model | the id listed by `/v1/models` |

Check from the other device:

```sh
curl https://<mac-name>.<tailnet>.ts.net/health
curl https://<mac-name>.<tailnet>.ts.net/v1/models
curl https://<mac-name>.<tailnet>.ts.net/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer local' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"stream":false}'
```

A `502` from the address means sharing is on and the engine is not running.

## Who can reach it

Every device signed in to your tailnet can send requests, and the engine cannot tell them apart from requests made on the Mac itself. If the tailnet has devices you would not hand the model to, turn on **Settings > Local API key > Require an API key** and give the key only to the clients you want.

Do not use `tailscale funnel` for this. Funnel publishes the address to the whole internet.

## Limits

- One request runs at a time. A second client waits up to 20 s and then gets `503 server_busy`.
- Speed is the Mac's. The network adds little: answers are streamed as text.
- The engine is not started or stopped remotely. To do that, connect to the Mac (Screen Sharing or SSH over Tailscale) and use the app or `./serve.sh`.
