/**
 * Preload script — exposes a safe IPC bridge to the wizard renderer.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("gently", {
  saveConfig: (cfg) => ipcRenderer.invoke("save-config", cfg),
  selectFolder: () => ipcRenderer.invoke("select-folder"),
  getDefaultStoragePath: () => ipcRenderer.invoke("get-default-storage-path"),
  validateApiKey: (key) => ipcRenderer.invoke("validate-api-key", key),
  wizardDone: (cfg) => ipcRenderer.send("wizard-done", cfg),
  wizardSkip: (cfg) => ipcRenderer.send("wizard-skip", cfg),
});
