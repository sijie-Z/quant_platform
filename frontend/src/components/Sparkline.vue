<template>
  <canvas ref="canvasRef" :width="width" :height="height" class="sparkline"></canvas>
</template>

<script setup>
import { ref, watch, onMounted, nextTick } from 'vue'

const props = defineProps({
  data: { type: Array, default: () => [] },
  color: { type: String, default: '#4da6ff' },
  width: { type: Number, default: 80 },
  height: { type: Number, default: 28 },
  lineWidth: { type: Number, default: 1.5 },
  filled: { type: Boolean, default: true },
})

const canvasRef = ref(null)

function draw() {
  const canvas = canvasRef.value
  if (!canvas || !props.data.length) return

  const ctx = canvas.getContext('2d')
  const dpr = window.devicePixelRatio || 1
  const w = props.width
  const h = props.height

  canvas.width = w * dpr
  canvas.height = h * dpr
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, w, h)

  const values = props.data
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = (max - min) || 1
  const padding = 2

  const points = values.map((v, i) => ({
    x: padding + (i / (values.length - 1)) * (w - padding * 2),
    y: padding + (1 - (v - min) / range) * (h - padding * 2),
  }))

  // Fill area
  if (props.filled) {
    ctx.beginPath()
    ctx.moveTo(points[0].x, h)
    for (const p of points) ctx.lineTo(p.x, p.y)
    ctx.lineTo(points[points.length - 1].x, h)
    ctx.closePath()

    const grad = ctx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, props.color + '30')
    grad.addColorStop(1, props.color + '05')
    ctx.fillStyle = grad
    ctx.fill()
  }

  // Line
  ctx.beginPath()
  ctx.moveTo(points[0].x, points[0].y)
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]
    const curr = points[i]
    const cpx = (prev.x + curr.x) / 2
    ctx.quadraticCurveTo(prev.x + (curr.x - prev.x) * 0.5, prev.y, cpx, (prev.y + curr.y) / 2)
    ctx.quadraticCurveTo(cpx, curr.y, curr.x, curr.y)
  }
  ctx.strokeStyle = props.color
  ctx.lineWidth = props.lineWidth
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.stroke()

  // End dot
  const last = points[points.length - 1]
  ctx.beginPath()
  ctx.arc(last.x, last.y, 2, 0, Math.PI * 2)
  ctx.fillStyle = props.color
  ctx.fill()
}

watch(() => props.data, () => nextTick(draw), { deep: true })
watch(() => props.color, () => nextTick(draw))

onMounted(() => { if (props.data.length) draw() })
</script>

<style scoped>
.sparkline {
  display: block;
  opacity: 0.8;
}
</style>
