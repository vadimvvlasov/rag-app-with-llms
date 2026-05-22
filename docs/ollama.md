# Ollama configuration

Ollama runs as a systemd service. Runtime settings are managed via a drop-in override file.

## OLLAMA_NUM_PARALLEL

Controls how many requests Ollama processes simultaneously. Set on the server — not in `.env`.

**Check current value**
```bash
sudo systemctl show ollama --property=Environment | grep NUM_PARALLEL
# or inspect the file directly:
cat /etc/systemd/system/ollama.service.d/override.conf
```

**Change the value**
```bash
sudo nano /etc/systemd/system/ollama.service.d/override.conf
# Edit the number, save (Ctrl+O, Enter, Ctrl+X), then apply:
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

**Verify it applied**
```bash
sudo systemctl show ollama --property=Environment | grep NUM_PARALLEL
```

**Current setting:** `OLLAMA_NUM_PARALLEL=16`

## Notes

- Match `max_workers` in `ThreadPoolExecutor` to `OLLAMA_NUM_PARALLEL`
- The override file survives Ollama package updates (unlike editing the main service file)
