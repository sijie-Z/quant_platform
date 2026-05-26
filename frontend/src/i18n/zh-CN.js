export default {
  // 导航
  nav: { terminal: '终端', trading: '交易', live: '实时', oms: '订单', compare: '对比', sweep: '扫描', factors: '因子', history: '历史', settings: '设置', monitor: '监控' },
  // 状态栏
  statusBar: { connected: '已连接', disconnected: '未连接', lastRun: '上次运行', version: '版本', run: '运行', save: '保存', export: '导出' },
  // KPI
  kpi: {
    totalReturn: '总收益', annualizedReturn: '年化收益', volatility: '波动率', sharpe: '夏普比',
    maxDrawdown: '最大回撤', calmar: '卡玛比', winRate: '胜率', profitLossRatio: '盈亏比',
    turnover: '换手率', alpha: 'Alpha', beta: 'Beta', informationRatio: '信息比'
  },
  // 终端仪表盘
  terminal: {
    pipeline: '流水线', runPipeline: '运行流水线', loading: '加载中...', run: '运行', result: '结果',
    equityCurve: '净值曲线', drawdown: '回撤', benchmark: '基准', portfolio: '组合',
    factorIc: '因子 IC 热力图', riskGauges: '风险仪表盘', portfolioExposure: '组合暴露',
    returnDistribution: '收益分布', excessReturn: '超额收益', topHoldings: '前十大持仓',
    factorScatter: '因子散点图', pnlAttribution: '盈亏归因', turnoverAnalysis: '换手分析',
    drawdownPeriods: '回撤周期', factorCorrelation: '因子相关性', icDecay: 'IC 衰减',
    monthlyReturns: '月度收益', rollingSharpe: '滚动夏普比', walkForward: 'Walk-Forward 验证',
    monteCarlo: '蒙特卡洛模拟', riskDecomposition: '风险分解', marketRegime: '市场状态',
    multiStrategy: '多策略', dataQuality: '数据质量', systemLog: '系统日志',
    noData: '暂无数据', clickRun: '点击"运行"开始回测',
    htmlReport: '生成 HTML 报告', downloadReport: '下载报告',
    exportBtn: '导出结果', demoBtn: '加载示例',
  },
  // 交易
  trading: {
    title: '实盘交易引擎', start: '启动', stop: '停止', status: '状态', account: '账户',
    positions: '持仓', orders: '订单', pnl: '盈亏', cash: '现金', marketValue: '市值',
    totalAssets: '总资产', running: '运行中', stopped: '已停止', cycle: '交易周期',
    signal: '信号', riskCheck: '风控检查', alert: '告警',
    broker: '券商', paperTrading: '模拟交易', qmt: 'QMT 实盘',
  },
  // 实时组合
  live: {
    title: '实时组合追踪', refresh: '刷新', lastUpdate: '上次更新', autoRefresh: '自动刷新',
    code: '代码', name: '名称', price: '价格', change: '涨跌幅', volume: '成交量',
    holdings: '持仓', sector: '行业',
  },
  // 订单
  oms: {
    title: '订单管理系统', orderId: '订单号', symbol: '代码', side: '方向', type: '类型',
    price: '价格', quantity: '数量', filled: '已成交', status: '状态', time: '时间',
    buy: '买入', sell: '卖出', limit: '限价', market: '市价',
    pending: '待处理', submitted: '已提交', filled: '已成', rejected: '已拒绝', cancelled: '已取消',
    tca: '交易成本分析', impact: '冲击成本', slippage: '滑点', delay: '延迟成本',
  },
  // 对比
  compare: {
    title: '策略对比', addStrategy: '添加策略', metric: '指标', value: '值',
    sharpe: '夏普比', return: '收益', risk: '风险', drawdown: '回撤',
  },
  // 扫描
  sweep: {
    title: '参数网格搜索', parameter: '参数', range: '范围', step: '步长',
    runSweep: '开始扫描', bestParams: '最优参数', progress: '进度',
  },
  // 因子
  factors: {
    title: '因子排名', rank: '排名', factor: '因子', ic: 'IC 值', icir: 'ICIR',
    stability: '稳定性', direction: '方向', positive: '正向', negative: '负向',
  },
  // 设置
  settings: {
    title: '设置', general: '通用', data: '数据', backtest: '回测', risk: '风控',
    save: '保存设置', reset: '重置', language: '语言', chinese: '中文', english: 'English',
  },
  // 监控
  monitor: {
    title: '监控大屏', riskOverview: '风控概览', tcaSummary: 'TCA 统计',
    factorStatus: '因子状态', capacityGauge: '策略容量', riskConfig: '风控配置',
    killSwitch: '紧急熔断', armed: '已就绪', triggered: '已触发',
  },
  // 面板通用
  panel: { loading: '加载中...', error: '加载失败', empty: '暂无数据', retry: '重试' },
  // 通用
  common: {
    confirm: '确认', cancel: '取消', ok: '确定', close: '关闭', yes: '是', no: '否',
    save: '保存', delete: '删除', edit: '编辑', add: '添加', search: '搜索', filter: '筛选',
    export: '导出', import: '导入', refresh: '刷新', submit: '提交', reset: '重置',
    day: '日', week: '周', month: '月', year: '年', all: '全部',
    jan: '1月', feb: '2月', mar: '3月', apr: '4月', may: '5月', jun: '6月',
    jul: '7月', aug: '8月', sep: '9月', oct: '10月', nov: '11月', dec: '12月',
  },
  // 命令面板
  command: { placeholder: '输入命令...', noResults: '无匹配结果' },
}
