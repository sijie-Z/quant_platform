// Dependency-free i18n completeness check.
//
// Verifies that:
//   1. every `$t('dotted.key')` used under src/ resolves in BOTH locales, and
//   2. the zh-CN and en-US bundles have identical key sets.
//
// Run with: node scripts/check-i18n.mjs
//
// It exists because the old language toggle flipped to `en-US` while only a
// zh-CN bundle existed, so every label rendered its raw key (`nav.terminal`).

import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import zhCN from '../src/i18n/zh-CN.js'
import enUS from '../src/i18n/en-US.js'

const here = dirname(fileURLToPath(import.meta.url))
const srcRoot = join(here, '..', 'src')

function flatten(bundle, prefix = '', out = {}) {
  for (const [key, value] of Object.entries(bundle)) {
    const path = prefix ? `${prefix}.${key}` : key
    if (value && typeof value === 'object') flatten(value, path, out)
    else out[path] = value
  }
  return out
}

function walk(dir, files = []) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (statSync(full).isDirectory()) walk(full, files)
    else if (name.endsWith('.vue') || name.endsWith('.js')) files.push(full)
  }
  return files
}

const flatZh = flatten(zhCN)
const flatEn = flatten(enUS)
const problems = []

const used = new Set()
// Match both quote styles; a future `$t("key")` call site must not slip past
// the check.
const keyRe = /\$t\(\s*(['"])([^'"]+)\1/g
for (const file of walk(srcRoot)) {
  const text = readFileSync(file, 'utf8')
  let match
  while ((match = keyRe.exec(text)) !== null) used.add(match[2])
}

for (const key of used) {
  if (!(key in flatZh)) problems.push(`used but missing from zh-CN: ${key}`)
  if (!(key in flatEn)) problems.push(`used but missing from en-US: ${key}`)
}

for (const key of Object.keys(flatZh)) {
  if (!(key in flatEn)) problems.push(`zh-CN key missing from en-US: ${key}`)
}
for (const key of Object.keys(flatEn)) {
  if (!(key in flatZh)) problems.push(`en-US key missing from zh-CN: ${key}`)
}

if (problems.length) {
  console.error(`i18n check FAILED (${problems.length} problem(s)):`)
  for (const problem of problems) console.error(`  - ${problem}`)
  process.exit(1)
}

console.log(`i18n check OK: ${used.size} used keys, ${Object.keys(flatZh).length} keys per locale, both bundles complete.`)
