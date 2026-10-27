// electron-builder "afterPack" hook. Apple Silicon refuses to start an app with no signature at
// all, so unsigned builds are given an ad-hoc signature ("-"). This is not notarisation: people
// still approve the first launch (see README). A real Developer ID certificate replaces this.
const { execFileSync } = require('node:child_process')
const path = require('node:path')

exports.default = async function adhocSign(context) {
  if (context.electronPlatformName !== 'darwin') return
  const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`)
  execFileSync('codesign', ['--force', '--deep', '--sign', '-', app], { stdio: 'inherit' })
}
