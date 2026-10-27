// Pure helpers for serving the built web app from the app:// scheme (unit-testable without Electron).
const path = require('node:path')

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm', // WebAssembly.instantiateStreaming insists on this exact type
  '.zip': 'application/zip',
  '.whl': 'application/zip',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
}

function mimeFor(file) {
  return MIME[path.extname(file).toLowerCase()] || 'application/octet-stream'
}

/**
 * Map a request path such as "/assets/app.js" to a file under `root`.
 * Returns null for anything that would leave `root` (path traversal) or is malformed.
 */
function resolveRequest(root, pathname) {
  let rel
  try {
    rel = decodeURIComponent(pathname || '/')
  } catch {
    return null
  }
  if (rel.includes('\0') || rel.includes('\\')) return null
  if (rel.endsWith('/')) rel += 'index.html'
  const clean = path.posix.normalize('/' + rel) // ".." can never climb above "/"
  const file = path.join(root, ...clean.split('/').filter(Boolean))
  return file.startsWith(root + path.sep) || file === root ? file : null
}

module.exports = { mimeFor, resolveRequest }
