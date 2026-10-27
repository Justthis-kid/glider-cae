import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './styles.css'

// No <StrictMode>: in dev it mounts effects twice, which would start (and
// immediately discard) a second Pyodide runtime.
createRoot(document.getElementById('root')).render(<App />)
