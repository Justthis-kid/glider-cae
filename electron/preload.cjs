// Tells the web app it is running as the desktop app and whether the Python runtime is bundled.
const { contextBridge } = require('electron')

contextBridge.exposeInMainWorld('glider', {
  isDesktop: true,
  platform: process.platform,
  offlineRuntime: process.argv.includes('--glider-offline'),
})
