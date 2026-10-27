// Bundle every module of the Python package as raw text so the web worker can
// write them into Pyodide's virtual file system. The package in
// ../python/glider_cae stays the single source of truth: the same files run
// natively (pytest, CLI) and in the browser.
const modules = import.meta.glob('../python/glider_cae/*.py', {
  query: '?raw',
  import: 'default',
  eager: true,
})

export const pythonFiles = Object.fromEntries(
  Object.entries(modules).map(([path, source]) => [path.split('/').pop(), source]),
)
