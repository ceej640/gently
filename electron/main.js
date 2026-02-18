/**
 * Gently Desktop — Electron Main Process
 *
 * Responsibilities:
 *   1. First-run wizard (collect API key + storage path)
 *   2. Spawn Python backend (dev: venv, prod: PyInstaller bundle)
 *   3. Load the web UI in a BrowserWindow once the server is ready
 *   4. Health polling + graceful shutdown
 */

const { app, BrowserWindow, ipcMain, dialog, net } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn, execSync } = require("child_process");
const http = require("http");

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
let mainWindow = null;
let wizardWindow = null;
let pythonProcess = null;
let serverPort = null;
let healthInterval = null;
let config = null;

const isDev = process.env.NODE_ENV === "development";

// ---------------------------------------------------------------------------
// Config helpers — stored in OS-appropriate userData directory
// ---------------------------------------------------------------------------
function configPath() {
  return path.join(app.getPath("userData"), "config.json");
}

function loadConfig() {
  try {
    const raw = fs.readFileSync(configPath(), "utf-8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function saveConfig(cfg) {
  const dir = path.dirname(configPath());
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(configPath(), JSON.stringify(cfg, null, 2));
}

// ---------------------------------------------------------------------------
// Port discovery — find a free TCP port
// ---------------------------------------------------------------------------
function findFreePort() {
  return new Promise((resolve, reject) => {
    const srv = require("net").createServer();
    srv.listen(0, "127.0.0.1", () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}

// ---------------------------------------------------------------------------
// Python process management
// ---------------------------------------------------------------------------
function pythonExePath() {
  if (isDev) {
    // Dev mode: use venv Python from repo root
    const repoRoot = path.resolve(__dirname, "..");
    if (process.platform === "win32") {
      return path.join(repoRoot, "venv", "Scripts", "python.exe");
    }
    return path.join(repoRoot, "venv", "bin", "python");
  }

  // Production: PyInstaller bundle in resources
  const exeName =
    process.platform === "win32" ? "electron_entry.exe" : "electron_entry";
  return path.join(process.resourcesPath, "python-dist", "electron_entry", exeName);
}

function pythonArgs(port, storagePath) {
  if (isDev) {
    const entryScript = path.resolve(__dirname, "..", "gently", "electron_entry.py");
    return [entryScript, "--port", String(port), "--storage-path", storagePath];
  }
  // PyInstaller bundle — the exe IS the entry point
  return ["--port", String(port), "--storage-path", storagePath];
}

function spawnPython(port, storagePath, apiKey) {
  const exe = pythonExePath();
  const args = pythonArgs(port, storagePath);

  const env = { ...process.env, ANTHROPIC_API_KEY: apiKey };

  // In production the exe is the PyInstaller bundle, not python
  const proc = isDev
    ? spawn(exe, args, { env, stdio: ["ignore", "pipe", "pipe"] })
    : spawn(exe, args, { env, stdio: ["ignore", "pipe", "pipe"] });

  return proc;
}

/**
 * Wait for the Python process to print "READY:{port}" on stdout.
 * Resolves with the port number, or rejects after a timeout.
 */
function waitForReady(proc, timeoutMs = 60000) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let output = "";

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error("Python backend did not become ready in time"));
      }
    }, timeoutMs);

    proc.stdout.on("data", (chunk) => {
      output += chunk.toString();
      const lines = output.split(/\r?\n/);
      for (const line of lines) {
        const match = line.match(/^READY:(\d+)$/);
        if (match && !settled) {
          settled = true;
          clearTimeout(timer);
          resolve(parseInt(match[1], 10));
          return;
        }
        // Surface errors from the Python process
        if (line.startsWith("ERROR:") && !settled) {
          settled = true;
          clearTimeout(timer);
          reject(new Error(line.slice(6)));
          return;
        }
      }
    });

    proc.on("error", (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(err);
      }
    });

    proc.on("exit", (code) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`Python process exited with code ${code}`));
      }
    });

    // Log stderr for debugging
    proc.stderr.on("data", (chunk) => {
      if (isDev) {
        process.stderr.write(chunk);
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Kill Python process tree (cross-platform)
// ---------------------------------------------------------------------------
function killPython() {
  if (!pythonProcess) return;

  const pid = pythonProcess.pid;
  if (!pid) return;

  try {
    if (process.platform === "win32") {
      execSync(`taskkill /F /T /PID ${pid}`, { stdio: "ignore" });
    } else {
      process.kill(-pid, "SIGTERM");
      // Escalate after 5 seconds
      setTimeout(() => {
        try {
          process.kill(-pid, "SIGKILL");
        } catch {
          // already dead
        }
      }, 5000);
    }
  } catch {
    // Process may already be dead
  }

  pythonProcess = null;
}

// ---------------------------------------------------------------------------
// Health polling
// ---------------------------------------------------------------------------
function startHealthPolling(port) {
  let consecutiveFailures = 0;

  healthInterval = setInterval(() => {
    const req = http.get(`http://127.0.0.1:${port}/health`, (res) => {
      if (res.statusCode === 200) {
        consecutiveFailures = 0;
      } else {
        consecutiveFailures++;
      }
    });

    req.on("error", () => {
      consecutiveFailures++;
      if (consecutiveFailures >= 3 && mainWindow && !mainWindow.isDestroyed()) {
        dialog.showErrorBox(
          "Backend Error",
          "The Gently backend has stopped responding. The application will close."
        );
        app.quit();
      }
    });

    req.setTimeout(5000, () => {
      req.destroy();
      consecutiveFailures++;
    });
  }, 5000);
}

function stopHealthPolling() {
  if (healthInterval) {
    clearInterval(healthInterval);
    healthInterval = null;
  }
}

// ---------------------------------------------------------------------------
// Window creation
// ---------------------------------------------------------------------------
function createWizardWindow() {
  wizardWindow = new BrowserWindow({
    width: 540,
    height: 520,
    resizable: false,
    frame: true,
    title: "Gently — Setup",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  wizardWindow.setMenuBarVisibility(false);
  wizardWindow.loadFile(path.join(__dirname, "wizard.html"));
}

function createSplashContent() {
  // Simple inline HTML shown while Python starts
  return `data:text/html,
    <html>
    <head><style>
      body {
        margin: 0; display: flex; align-items: center; justify-content: center;
        height: 100vh; background: #0f172a; color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        flex-direction: column;
      }
      .spinner {
        width: 40px; height: 40px; border: 3px solid #334155;
        border-top-color: #60a5fa; border-radius: 50%;
        animation: spin 0.8s linear infinite; margin-bottom: 20px;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
    </style></head>
    <body>
      <div class="spinner"></div>
      <div>Starting Gently backend&hellip;</div>
    </body>
    </html>`;
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    title: "Gently",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.setMenuBarVisibility(false);
  return mainWindow;
}

// ---------------------------------------------------------------------------
// Launch sequence
// ---------------------------------------------------------------------------
async function launchApp(cfg) {
  config = cfg;

  // Create main window with splash
  const win = createMainWindow();
  win.loadURL(createSplashContent());

  try {
    // Find a free port
    serverPort = await findFreePort();

    // Spawn Python backend
    pythonProcess = spawnPython(serverPort, cfg.storagePath, cfg.apiKey);

    // Detach on Unix so we can kill the process group
    if (process.platform !== "win32" && pythonProcess.pid) {
      try {
        process.kill(pythonProcess.pid, 0); // verify alive
      } catch {
        // ignore
      }
    }

    // Wait for READY signal
    const readyPort = await waitForReady(pythonProcess);
    serverPort = readyPort;

    // Load the web UI
    win.loadURL(`http://127.0.0.1:${serverPort}/`);

    // Start health monitoring
    startHealthPolling(serverPort);
  } catch (err) {
    dialog.showErrorBox(
      "Startup Error",
      `Failed to start the Gently backend:\n\n${err.message}`
    );
    killPython();
    app.quit();
  }
}

// ---------------------------------------------------------------------------
// IPC handlers (used by wizard)
// ---------------------------------------------------------------------------
ipcMain.handle("save-config", async (_event, cfg) => {
  saveConfig(cfg);
  return true;
});

ipcMain.handle("select-folder", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory", "createDirectory"],
    title: "Choose storage folder",
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle("get-default-storage-path", () => {
  return path.join(app.getPath("documents"), "Gently");
});

ipcMain.handle("validate-api-key", (_event, key) => {
  // Basic format check — real validation happens when the copilot starts
  return typeof key === "string" && key.startsWith("sk-ant-");
});

ipcMain.on("wizard-done", (_event, cfg) => {
  if (wizardWindow && !wizardWindow.isDestroyed()) {
    wizardWindow.close();
    wizardWindow = null;
  }
  launchApp(cfg);
});

ipcMain.on("wizard-skip", (_event, cfg) => {
  if (wizardWindow && !wizardWindow.isDestroyed()) {
    wizardWindow.close();
    wizardWindow = null;
  }
  // Launch the backend in UI-only mode (no API key → no copilot, but UI works)
  launchApp({ ...cfg, apiKey: "" });
});

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------
app.whenReady().then(() => {
  config = loadConfig();

  if (config && config.apiKey) {
    // Config exists with key — launch directly
    launchApp(config);
  } else if (config && !config.apiKey) {
    // Previously skipped — launch backend in UI-only mode
    launchApp(config);
  } else {
    // First run — show wizard
    createWizardWindow();
  }
});

app.on("window-all-closed", () => {
  app.quit();
});

app.on("before-quit", () => {
  stopHealthPolling();
  killPython();
});

app.on("will-quit", () => {
  stopHealthPolling();
  killPython();
});
