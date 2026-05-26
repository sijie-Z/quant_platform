import { reactive, computed, provide, inject } from 'vue'
import zhCN from './zh-CN.js'

const LANG_KEY = 'quant_platform_lang'

const messages = { 'zh-CN': zhCN }

const state = reactive({
  locale: localStorage.getItem(LANG_KEY) || 'zh-CN',
})

export function setLocale(locale) {
  state.locale = locale
  localStorage.setItem(LANG_KEY, locale)
  document.documentElement.lang = locale
}

export function getLocale() {
  return state.locale
}

export function t(path, fallback) {
  const keys = path.split('.')
  let msg = messages[state.locale]
  for (const key of keys) {
    if (msg && typeof msg === 'object') msg = msg[key]
    else return fallback || path
  }
  return typeof msg === 'string' ? msg : (fallback || path)
}

// 组合式 API
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

// provide/inject 模式
export const I18N_KEY = Symbol('i18n')

export function provideI18n() {
  provide(I18N_KEY, { locale: computed(() => state.locale), t, setLocale })
}

export function injectI18n() {
  return inject(I18N_KEY, { locale: computed(() => state.locale), t, setLocale })
}

// 初始化语言
setLocale(state.locale)

export default { install(app) {
  app.config.globalProperties.$t = t
  app.provide(I18N_KEY, { locale: computed(() => state.locale), t: (path, fb) => t(path, fb), setLocale })
}}
