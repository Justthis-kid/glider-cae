// Electron main process: opens the built web app in a window.
//
// The page is served from a custom app:// scheme instead of file:// so that fetch(), Web Workers
// and WebAssembly streaming (all needed by Pyodide) behave exactly as they do on a web server.
// Run with --dev to load the Vite dev server instead (npm run electron:dev).
const { app, BrowserWindow, protocol, shell } = require('electron')
const fs = require('node:fs')
const path = require('node:path')
const { mimeFor, resolveRequest } = require('./serve.cjs')

const DEV = process.argv.includes('--dev')
const DEV_URL = 'http://127.0.0.1:5173'
const DIST = path.join(__dirname, '..', 'dist')
const APP_URL = 'app://glider/index.html'

protocol.registerSchemesAsPrivileged([
  { scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } },
])

// The Python runtime ships inside the app when `npm run fetch:pyodide` ran before the build.
function hasLocalRuntime() {
  const root = DEV ? path.join(__dirname, '..', 'public') : DIST
  return fs.existsSync(path.join(root, 'pyodide', 'pyodide.js'))
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1360,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    show: false,
    backgroundColor: '#a9c7de',
    title: 'Glider Flight Lab',
    autoHideMenuBar: true,
    icon: path.join(__dirname, '..', 'build', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
      additionalArguments: hasLocalRuntime() ? ['--glider-offline'] : [],
    },
  })
  win.once('ready-to-show', () => win.show())

  // links to other sites open in the default browser, never inside the app window
  const isInternal = (url) => url.startsWith('app://') || url.startsWith(DEV_URL)
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) shell.openExternal(url)
    return { action: 'deny' }
  })
  win.webContents.on('will-navigate', (event, url) => {
    if (!isInternal(url)) {
      event.preventDefault()
      if (/^https?:/.test(url)) shell.openExternal(url)
    }
  })

  win.loadURL(DEV ? DEV_URL : APP_URL)
  return win
}

if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', () => {
    const [win] = BrowserWindow.getAllWindows()
    if (win) {
      if (win.isMinimized()) win.restore()
      win.focus()
    }
  })

  app.whenReady().then(() => {
    protocol.handle('app', async (request) => {
      const file = resolveRequest(DIST, new URL(request.url).pathname)
      if (!file) return new Response('Forbidden', { status: 403 })
      try {
        const data = await fs.promises.readFile(file)
        return new Response(data, { headers: { 'Content-Type': mimeFor(file) } })
      } catch {
        return new Response('Not found', { status: 404 })
      }
    })
    createWindow()
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
  })
}
