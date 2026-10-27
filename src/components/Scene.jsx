import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { bodyToThree, nedToThree, quatNedToThree } from '../lib/frames'

const SKY = '#a9c7de'
const HAZE = '#dce8f0'
const GROUND = '#8f9c7e'

/** Small deterministic PRNG so the pylon field is identical on every load. */
function lcg(seed) {
  let s = seed >>> 0
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296
}

/** Flat ground, a grid that follows the glider in 10 m steps, and tall white
 *  pylons that give a sense of speed and parallax from 100+ m up. */
function Terrain({ stateRef }) {
  const plane = useRef()
  const grid = useRef()
  const poles = useRef()
  const COUNT = 700

  useLayoutEffect(() => {
    const rnd = lcg(7)
    const matrix = new THREE.Matrix4()
    const rotation = new THREE.Quaternion()
    for (let i = 0; i < COUNT; i++) {
      const height = 6 + rnd() * 26
      const position = new THREE.Vector3((rnd() - 0.5) * 3000, height / 2, (rnd() - 0.5) * 3000)
      matrix.compose(position, rotation, new THREE.Vector3(1.4, height, 1.4))
      poles.current.setMatrixAt(i, matrix)
    }
    poles.current.instanceMatrix.needsUpdate = true
  }, [])

  useFrame(() => {
    const s = stateRef.current
    if (!s) return
    const [x, , z] = nedToThree(s.position_ned)
    plane.current.position.set(x, 0, z)
    grid.current.position.set(Math.round(x / 10) * 10, 0.03, Math.round(z / 10) * 10)
  })

  return (
    <>
      <mesh ref={plane} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[6000, 6000]} />
        <meshLambertMaterial color={GROUND} />
      </mesh>
      <gridHelper ref={grid} args={[3000, 300, '#76836a', '#86937a']} />
      <instancedMesh ref={poles} args={[undefined, undefined, COUNT]} frustumCulled={false}>
        <boxGeometry args={[1, 1, 1]} />
        <meshLambertMaterial color="#eef1ee" />
      </instancedMesh>
    </>
  )
}

/** Built-in glider (nose toward -Z) used until a model is loaded. */
function PlaceholderGlider() {
  return (
    <>
      <mesh position={[0, 0, -0.12]} rotation={[Math.PI / 2, 0, 0]}>
        <capsuleGeometry args={[0.045, 0.5, 6, 12]} />
        <meshStandardMaterial color="#14263a" roughness={0.5} />
      </mesh>
      <mesh position={[0, 0.01, 0.5]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.014, 0.02, 0.75, 8]} />
        <meshStandardMaterial color="#14263a" roughness={0.5} />
      </mesh>
      <mesh position={[0, 0.04, -0.08]}>
        <boxGeometry args={[2.0, 0.02, 0.17]} />
        <meshStandardMaterial color="#f4f6f7" roughness={0.6} />
      </mesh>
      {[-0.91, 0.91].map((x) => (
        <mesh key={x} position={[x, 0.041, -0.08]}>
          <boxGeometry args={[0.18, 0.021, 0.17]} />
          <meshStandardMaterial color="#e4572e" roughness={0.6} />
        </mesh>
      ))}
      <mesh position={[0, 0.01, 0.82]}>
        <boxGeometry args={[0.55, 0.012, 0.1]} />
        <meshStandardMaterial color="#f4f6f7" roughness={0.6} />
      </mesh>
      <mesh position={[0, 0.09, 0.84]}>
        <boxGeometry args={[0.012, 0.16, 0.12]} />
        <meshStandardMaterial color="#e4572e" roughness={0.6} />
      </mesh>
    </>
  )
}

/** The user's mesh, already oriented into body axes by the analysis. Shifted so the centre of
 *  gravity sits at the group origin, which is where the simulation puts the glider. */
function MeshModel({ preview, cgBody }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(preview.positions, 3))
    g.setIndex(new THREE.BufferAttribute(preview.indices, 1))
    g.computeVertexNormals()
    return g
  }, [preview])
  useEffect(() => () => geometry.dispose(), [geometry])

  const [cx, cy, cz] = bodyToThree(cgBody ?? [0, 0, 0])
  return (
    <mesh geometry={geometry} position={[-cx, -cy, -cz]}>
      <meshStandardMaterial color="#e9edf0" roughness={0.65} metalness={0.05}
                            side={THREE.DoubleSide} flatShading />
    </mesh>
  )
}

function GliderModel({ stateRef, preview, cgBody }) {
  const group = useRef()

  useFrame(() => {
    const s = stateRef.current
    if (!s || !group.current) return
    const [x, y, z] = nedToThree(s.position_ned)
    const [qx, qy, qz, qw] = quatNedToThree(s.quat)
    group.current.position.set(x, y, z)
    group.current.quaternion.set(qx, qy, qz, qw)
  })

  return (
    <group ref={group}>
      {preview ? <MeshModel preview={preview} cgBody={cgBody} /> : <PlaceholderGlider />}
    </group>
  )
}

/** Flight path: one vertex every 0.1 s of simulated time, oldest dropped first. */
function Trail({ stateRef, max = 900 }) {
  const count = useRef(0)
  const lastT = useRef(-1)
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(max * 3), 3))
    g.setDrawRange(0, 0)
    return g
  }, [max])
  const line = useMemo(() => {
    const l = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: '#e4572e' }))
    l.frustumCulled = false
    return l
  }, [geometry])

  useFrame(() => {
    const s = stateRef.current
    if (!s) return
    if (s.t < lastT.current) {            // simulation was reset
      count.current = 0
      lastT.current = -1
    }
    if (s.t - lastT.current < 0.1) return
    lastT.current = s.t

    const positions = geometry.attributes.position.array
    if (count.current >= max) {
      positions.copyWithin(0, 3)
      count.current = max - 1
    }
    positions.set(nedToThree(s.position_ned), count.current * 3)
    count.current += 1
    geometry.attributes.position.needsUpdate = true
    geometry.setDrawRange(0, count.current)
  })

  return <primitive object={line} />
}

/** Follows behind the glider's heading (yaw only) so roll and pitch stay visible. */
function ChaseCamera({ stateRef, span }) {
  const target = useMemo(() => new THREE.Vector3(), [])
  const look = useMemo(() => new THREE.Vector3(), [])
  const snapped = useRef(false)

  useFrame(({ camera }, delta) => {
    const s = stateRef.current
    if (!s) return
    const [x, y, z] = nedToThree(s.position_ned)
    const yaw = THREE.MathUtils.degToRad(s.euler_deg[2])
    // three-space forward is (sin yaw, 0, -cos yaw); sit behind and above it, further out for bigger models
    const dist = Math.min(Math.max(2.7 * span, 2), 40)
    target.set(x - Math.sin(yaw) * dist, y + 0.3 * dist, z + Math.cos(yaw) * dist)
    if (!snapped.current) {
      camera.position.copy(target)
      snapped.current = true
    }
    camera.position.lerp(target, 1 - Math.exp(-3 * delta))
    camera.lookAt(look.set(x, y + 0.04 * dist, z))
  })
  return null
}

export default function Scene({ stateRef, preview, cgBody, span = 2 }) {
  return (
    <Canvas
      className="scene"
      dpr={[1, 2]}
      camera={{ fov: 55, near: 0.1, far: 5000, position: [0, 122, 6] }}
    >
      <color attach="background" args={[SKY]} />
      <fog attach="fog" args={[HAZE, 250, 2200]} />
      <hemisphereLight args={['#ffffff', GROUND, 1.1]} />
      <directionalLight position={[60, 120, 40]} intensity={1.8} />
      <Terrain stateRef={stateRef} />
      <Trail stateRef={stateRef} />
      <GliderModel stateRef={stateRef} preview={preview} cgBody={cgBody} />
      <ChaseCamera stateRef={stateRef} span={span} />
    </Canvas>
  )
}
