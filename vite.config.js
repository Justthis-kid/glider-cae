import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves a project site from /<repo-name>/, so the deploy workflow passes
// VITE_BASE. Locally and in the desktop app it falls back to '/'.
// The dev server address is fixed because the desktop dev command waits for it.
export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  plugins: [react()],
  server: { host: '127.0.0.1', port: 5173, strictPort: true },
})
