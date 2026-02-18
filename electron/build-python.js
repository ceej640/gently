/**
 * Build the Python backend with PyInstaller.
 *
 * Runs from the repo root so PyInstaller can resolve all paths in the spec.
 * Usage:  node electron/build-python.js   (or: npm run build:python)
 */

const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const repoRoot = path.resolve(__dirname, "..");
const specFile = path.join(repoRoot, "pyinstaller.spec");

if (!fs.existsSync(specFile)) {
  console.error("ERROR: pyinstaller.spec not found at", specFile);
  process.exit(1);
}

console.log("Building Python backend with PyInstaller...");
console.log("  Spec:", specFile);
console.log("  Working dir:", repoRoot);

try {
  execSync("pyinstaller --clean --noconfirm pyinstaller.spec", {
    cwd: repoRoot,
    stdio: "inherit",
  });
  console.log("\nPython build complete. Output: dist/electron_entry/");
} catch (err) {
  console.error("\nPyInstaller build failed.");
  process.exit(1);
}
