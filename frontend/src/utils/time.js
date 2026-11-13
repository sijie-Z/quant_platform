// Backend timestamps are naive Asia/Shanghai strings (`datetime.now()
// .isoformat()`), e.g. `2026-10-21T09:30:00.123456`. `new Date(iso)` interprets
// them as the browser's local time, so a user outside China saw shifted times
// in run history and live-trading panels. Parse naive values as UTC+8 and keep
// explicitly zoned values as-is.

const ZONED_RE = /(?:[zZ]|[+-]\d{2}:?\d{2})$/
const NAIVE_RE = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?$/

export function parseBackendTime(value) {
  if (value == null || value === '') return null
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value
  }
  const text = String(value).trim()
  if (!text) return null

  // Already carries a zone (ISO `Z` or `±hh:mm`): let the engine parse it.
  if (ZONED_RE.test(text)) {
    const zoned = new Date(text)
    return Number.isNaN(zoned.getTime()) ? null : zoned
  }

  const match = NAIVE_RE.exec(text)
  if (match) {
    const fraction = (match[3] || '0').slice(0, 3).padEnd(3, '0')
    const date = new Date(`${match[1]}T${match[2]}.${fraction}+08:00`)
    return Number.isNaN(date.getTime()) ? null : date
  }

  // Date-only strings (`2026-10-21`) parse as UTC midnight; acceptable.
  const fallback = new Date(text)
  return Number.isNaN(fallback.getTime()) ? null : fallback
}

export function formatBackendTime(value, locale, options = {}) {
  const date = parseBackendTime(value)
  if (!date) return '-'
  const tag = locale === 'zh-CN' ? 'zh-CN' : 'en-US'
  return date.toLocaleString(tag, {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    ...options,
  })
}

export function formatBackendClock(value, locale) {
  const date = parseBackendTime(value)
  if (!date) return ''
  const tag = locale === 'zh-CN' ? 'zh-CN' : 'en-US'
  return date.toLocaleTimeString(tag, { hour12: false })
}

// Current calendar date in the exchange timezone (Asia/Shanghai), as
// YYYY-MM-DD. Comparing a quote's bar date against the *browser's* local date
// rejected valid same-day bars for users west of UTC+8.
export function shanghaiToday() {
  try {
    return new Intl.DateTimeFormat('en-CA', {
      timeZone: 'Asia/Shanghai',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).format(new Date())
  } catch (_) {
    const shifted = new Date(Date.now() + 8 * 3600 * 1000)
    const year = shifted.getUTCFullYear()
    const month = String(shifted.getUTCMonth() + 1).padStart(2, '0')
    const day = String(shifted.getUTCDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }
}
