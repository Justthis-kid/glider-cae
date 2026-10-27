// Downloads the Pyodide runtime (and the packages the app needs) into public/pyodide so the
// desktop app works offline. Run automatically by the dist:* scripts; safe to re-run, files that
// are already present are skipped.
import { access, mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..')
const { version } = JSON.parse(await readFile(path.join(root, 'pyodide.config.json'), 'utf8'))
const BASE = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`
const OUT = path.join(root, 'public', 'pyodide')

const CORE = ['pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json']
const PACKAGES = ['numpy', 'micropip'] // micropip lets the app install trimesh when online

async function exists(file) {
  try {
    await access(file)
    return true
  } catch {
    return false
  }
}

async function download(name) {
  const target = path.join(OUT, name)
  if (await exists(target)) return false
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await fetch(BASE + name)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      await writeFile(target, Buffer.from(await res.arrayBuffer()))
      console.log(`  downloaded ${name}`)
      return true
    } catch (err) {
      if (attempt === 3) throw new Error(`Could not download ${BASE + name}: ${err.message}`)
      await new Promise((r) => setTimeout(r, 1000 * attempt))
    }
  }
  return false
}

await mkdir(OUT, { recursive: true })
console.log(`Pyodide ${version} -> ${path.relative(root, OUT)}`)
for (const name of CORE) await download(name)

// the lock file lists every package wheel and its dependencies
const lock = JSON.parse(await readFile(path.join(OUT, 'pyodide-lock.json'), 'utf8'))
const wanted = new Set()
const visit = (name) => {
  const key = name.toLowerCase().replace(/_/g, '-')
  if (wanted.has(key)) return
  const pkg = lock.packages[key]
  if (!pkg) throw new Error(`Package "${name}" is not in the Pyodide ${version} lock file`)
  wanted.add(key)
  for (const dep of pkg.depends ?? []) visit(dep)
}
PACKAGES.forEach(visit)
for (const key of wanted) await download(lock.packages[key].file_name)
console.log(`Done: ${CORE.length} core files + ${wanted.size} packages (${[...wanted].join(', ')}).`)
