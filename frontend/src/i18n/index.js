import { reactive, computed, provide, inject } from 'vue'
import zhCN from './zh-CN.js'
import enUS from './en-US.js'

const LANG_KEY = 'quant_platform_lang'

const messages = { 'zh-CN': zhCN, 'en-US': enUS }
const DEFAULT_LOCALE = 'zh-CN'
export const AVAILABLE_LOCALES = Object.keys(messages)

function resolve(bundle, path) {
  const keys = path.split('.')
  let msg = bundle
  for (const key of keys) {
    if (msg && typeof msg === 'object') msg = msg[key]
    else return undefined
  }
  return typeof msg === 'string' ? msg : undefined
}

function initialLocale() {
  let stored = ''
  try {
    stored = localStorage.getItem(LANG_KEY) || ''
  } catch (_) {}
  return AVAILABLE_LOCALES.includes(stored) ? stored : DEFAULT_LOCALE
}

const state = reactive({
  locale: initialLocale(),
})

export function setLocale(locale) {
  // Never persist a locale without a message bundle: that is what rendered raw
  // keys (`nav.terminal`, `kpi.totalReturn`, ...) under the old en-US toggle.
  const next = AVAILABLE_LOCALES.includes(locale) ? locale : DEFAULT_LOCALE
  state.locale = next
  try {
    localStorage.setItem(LANG_KEY, next)
  } catch (_) {}
  if (typeof document !== 'undefined') document.documentElement.lang = next
}

export function getLocale() {
  return state.locale
}

// Lookup order: active locale -> zh-CN (the canonical, complete bundle) ->
// caller fallback -> the key itself. Raw keys can therefore only appear if a
// key is missing from every bundle and no fallback was supplied; the
// `scripts/check-i18n.mjs` check enforces that every key used in `src/` exists
// in both bundles.
export function t(path, fallback) {
  return (
    resolve(messages[state.locale], path) ??
    resolve(messages[DEFAULT_LOCALE], path) ??
    fallback ??
    path
  )
}

// Composable API
export function useI18n() {
  const locale = computed(() => state.locale)
  function $t(path, fallback) {
    return t(path, fallback)
  }
  function setLang(l) {
    setLocale(l)
  }
  return { locale, $t, setLang, t: $t }
}

// provide/inject mode
export const I18N_KEY = Symbol('i18n')

export function provideI18n() {
  provide(I18N_KEY, { locale: computed(() => state.locale), t, setLocale })
}

export function injectI18n() {
  return inject(I18N_KEY, { locale: computed(() => state.locale), t, setLocale })
}

// Initialize the document language.
setLocale(state.locale)

export default {
  install(app) {
    app.config.globalProperties.$t = t
    app.provide(I18N_KEY, { locale: computed(() => state.locale), t: (path, fb) => t(path, fb), setLocale })
  },
}
