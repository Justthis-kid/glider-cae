import { bodyToThree } from './frames'

// The Python side sends the oriented model as one base64 blob: float32 vertices (body frame:
// x forward, y right, z down) followed by uint32 triangle indices. Returns typed arrays already
// mapped into the glider's local three.js axes.
export function decodePreview({ vertices, faces, data }) {
  const bin = atob(data)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)

  const body = new Float32Array(bytes.buffer, 0, vertices * 3)
  const indices = new Uint32Array(bytes.buffer, vertices * 12, faces * 3)
  const positions = new Float32Array(vertices * 3)
  for (let i = 0; i < vertices; i++) {
    const [x, y, z] = bodyToThree([body[3 * i], body[3 * i + 1], body[3 * i + 2]])
    positions[3 * i] = x
    positions[3 * i + 1] = y
    positions[3 * i + 2] = z
  }
  return { positions, indices: new Uint32Array(indices) }
}
