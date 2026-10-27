// The physics uses aerospace frames (inertial NED, body FRD). Three.js is
// right-handed with +Y up and the camera looking down -Z. Mapping:
//   three = (East, Up, -North) = (E, -D, -N)         (a proper rotation, det = +1)
// A rotation quaternion q = (w, v) maps the same way: v' = M v, w unchanged.
// A body-frame point (forward, right, down) maps with the same rule: (y, -z, -x).

export const nedToThree = ([n, e, d]) => [e, -d, -n]

// Python quaternion [w, x, y, z] (body -> NED)  ->  three.js order [x, y, z, w]
export const quatNedToThree = ([w, x, y, z]) => [y, -z, -x, w]

// Body-frame point (x forward, y right, z down) -> the glider's local three.js coordinates:
// nose toward -Z, right wing toward +X, up toward +Y.
export const bodyToThree = ([x, y, z]) => [y, -z, -x]
