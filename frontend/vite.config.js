import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // The status stream connects to /api/ws; without ws:true the dev
        // server rejected the upgrade and the dashboard silently fell back
        // to polling.
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    rollupOptions: {
      output: {
        manualChunks: {
          echarts: ['echarts'],
          vue: ['vue'],
        },
      },
    },
  },
})
