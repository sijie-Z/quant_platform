// Canonical A-share symbol helpers — a client-side mirror of
// `shared/symbols.py::normalize_symbol` / `symbol_code` / `infer_exchange`.
//
// The backend stores portfolio holdings as bare six-digit codes while the OMS
// normalizes tickers to `CODE.EXCHANGE`. Importing `600519.SH` used to be
// rejected or silently dropped, and the Live table showed `600519` for the
// same instrument the OMS showed as `600519.SH`. Keeping one implementation
// of the normalization semantics on the client (and using it for every view)
// fixes both.
//
// Accepted inputs: `600519`, `600519.SH`, `sh600519`, `sh.600519`, `sh-600519`,
// lower/upper case. An explicitly declared exchange is preserved
// (`000001.SH` is the Shanghai index, not Ping An Bank `000001.SZ`).

const EXCHANGES = ['SH', 'SZ', 'BJ']

const SUFFIX_RE = /^(\d{6})[.\-_]?(SH|SZ|BJ)$/
const PREFIX_RE = /^(SH|SZ|BJ)[.\-_]?(\d{6})$/
const BARE_RE = /^(\d{6})$/
const CODE_RE = /(\d{6})/

export function symbolCode(symbol) {
  if (symbol == null) return null
  const match = CODE_RE.exec(String(symbol))
  return match ? match[1] : null
}

export function inferExchange(code) {
  if (!code) return 'SZ'
  if (code.startsWith('920') || code.startsWith('4') || code.startsWith('8')) return 'BJ'
  if (code.startsWith('5') || code.startsWith('6') || code.startsWith('9')) return 'SH'
  if (code.startsWith('0') || code.startsWith('2') || code.startsWith('3')) return 'SZ'
  if (code.startsWith('12') || code.startsWith('15') || code.startsWith('16') || code.startsWith('18')) return 'SZ'
  if (code.startsWith('1')) return 'SH'
  return 'SZ'
}

export function normalizeSymbol(symbol) {
  const text = String(symbol == null ? '' : symbol).trim().toUpperCase()
  let match = SUFFIX_RE.exec(text)
  if (match) return `${match[1]}.${match[2]}`
  match = PREFIX_RE.exec(text)
  if (match) return `${match[2]}.${match[1]}`
  match = BARE_RE.exec(text)
  if (match) return `${match[1]}.${inferExchange(match[1])}`
  // No six-digit code: return the trimmed text so callers can still log it.
  // Such a symbol never matches a real instrument.
  return text
}

export function isKnownExchange(exchange) {
  return EXCHANGES.includes(exchange)
}
