export default {
  // Navigation
  nav: { terminal: 'Terminal', trading: 'Trading', live: 'Live', oms: 'OMS', compare: 'Compare', sweep: 'Sweep', factors: 'Factors', history: 'History', settings: 'Settings', monitor: 'Monitor' },
  // Status bar
  statusBar: { connected: 'CONNECTED', disconnected: 'DISCONNECTED', lastRun: 'Last run', version: 'Version', run: 'Run', save: 'Save', export: 'Export' },
  // KPI
  kpi: {
    totalReturn: 'Total Return', annualizedReturn: 'Annualized', volatility: 'Volatility', sharpe: 'Sharpe',
    maxDrawdown: 'Max Drawdown', calmar: 'Calmar', winRate: 'Win Rate', profitLossRatio: 'P/L Ratio',
    turnover: 'Turnover', alpha: 'Alpha', beta: 'Beta', informationRatio: 'Information Ratio',
    sortino: 'Sortino', var: 'VaR (95%)',
  },
  // Terminal dashboard
  terminal: {
    pipeline: 'Pipeline', runPipeline: 'Run Pipeline', loading: 'Loading...', run: 'Run', result: 'Result',
    equityCurve: 'Equity Curve', drawdown: 'Drawdown', benchmark: 'Benchmark', portfolio: 'Portfolio',
    factorIc: 'Factor IC Heatmap', riskGauges: 'Risk Gauges', portfolioExposure: 'Portfolio Exposure',
    returnDistribution: 'Return Distribution', excessReturn: 'Excess Return', topHoldings: 'Top Holdings',
    factorScatter: 'Factor Scatter', pnlAttribution: 'P&L Attribution', turnoverAnalysis: 'Turnover Analysis',
    drawdownPeriods: 'Drawdown Periods', factorCorrelation: 'Factor Correlation', icDecay: 'IC Decay',
    monthlyReturns: 'Monthly Returns', rollingSharpe: 'Rolling Sharpe', walkForward: 'Walk-Forward',
    monteCarlo: 'Monte Carlo', riskDecomposition: 'Risk Decomposition', marketRegime: 'Market Regime',
    multiStrategy: 'Multi-Strategy', dataQuality: 'Data Quality', systemLog: 'System Log',
    noData: 'No data', clickRun: 'Click "Run" to start a backtest',
    htmlReport: 'Generate HTML Report', downloadReport: 'Download Report',
    exportBtn: 'Export Results', demoBtn: 'Load Demo',
  },
  // Trading
  trading: {
    title: 'Live Trading Engine', start: 'Start', stop: 'Stop', status: 'Status', account: 'Account',
    positions: 'Positions', orders: 'Orders', pnl: 'P&L', cash: 'Cash', marketValue: 'Market Value',
    totalAssets: 'Total Assets', running: 'Running', stopped: 'Stopped', cycle: 'Cycle',
    signal: 'Signal', riskCheck: 'Risk Check', alert: 'Alert',
    broker: 'Broker', paperTrading: 'Paper Trading', qmt: 'QMT Live',
  },
  // Live portfolio
  live: {
    title: 'Live Portfolio Tracker', refresh: 'Refresh', lastUpdate: 'Last update', autoRefresh: 'Auto refresh',
    code: 'Code', name: 'Name', price: 'Price', change: 'Change', volume: 'Volume',
    holdings: 'Holdings', sector: 'Sector',
  },
  // OMS
  oms: {
    title: 'Order Management System', orderId: 'Order ID', symbol: 'Symbol', side: 'Side', type: 'Type',
    price: 'Price', quantity: 'Quantity', filled: 'Filled', status: 'Status', time: 'Time',
    buy: 'Buy', sell: 'Sell', limit: 'Limit', market: 'Market',
    pending: 'Pending', submitted: 'Submitted', filled: 'Filled', rejected: 'Rejected', cancelled: 'Cancelled',
    tca: 'Trade Cost Analysis', impact: 'Impact', slippage: 'Slippage', delay: 'Delay',
  },
  // Compare
  compare: {
    title: 'Strategy Comparison', addStrategy: 'Add Strategy', metric: 'Metric', value: 'Value',
    sharpe: 'Sharpe', return: 'Return', risk: 'Risk', drawdown: 'Drawdown',
  },
  // Sweep
  sweep: {
    title: 'Parameter Grid Search', parameter: 'Parameter', range: 'Range', step: 'Step',
    runSweep: 'Run Sweep', bestParams: 'Best Parameters', progress: 'Progress',
  },
  // Factors
  factors: {
    title: 'Factor Ranking', rank: 'Rank', factor: 'Factor', ic: 'IC', icir: 'ICIR',
    stability: 'Stability', direction: 'Direction', positive: 'Positive', negative: 'Negative',
  },
  // Settings
  settings: {
    title: 'Settings', general: 'General', data: 'Data', backtest: 'Backtest', risk: 'Risk',
    save: 'Save Settings', reset: 'Reset', language: 'Language', chinese: '中文', english: 'English',
  },
  // Monitor
  monitor: {
    title: 'Monitor', riskOverview: 'Risk Overview', tcaSummary: 'TCA Summary',
    factorStatus: 'Factor Status', capacityGauge: 'Capacity', riskConfig: 'Risk Config',
    killSwitch: 'Kill Switch', armed: 'Armed', triggered: 'Triggered',
  },
  // Panels
  panel: { loading: 'Loading...', error: 'Failed to load', empty: 'No data', retry: 'Retry', maximize: 'Maximize', restore: 'Restore' },
  // Common
  common: {
    confirm: 'Confirm', cancel: 'Cancel', ok: 'OK', close: 'Close', yes: 'Yes', no: 'No',
    save: 'Save', delete: 'Delete', edit: 'Edit', add: 'Add', search: 'Search', filter: 'Filter',
    export: 'Export', import: 'Import', refresh: 'Refresh', submit: 'Submit', reset: 'Reset',
    day: 'Day', week: 'Week', month: 'Month', year: 'Year', all: 'All',
    jan: 'Jan', feb: 'Feb', mar: 'Mar', apr: 'Apr', may: 'May', jun: 'Jun',
    jul: 'Jul', aug: 'Aug', sep: 'Sep', oct: 'Oct', nov: 'Nov', dec: 'Dec',
  },
  // Command palette
  command: { placeholder: 'Type a command or search...', noResults: 'No matching results' },
}
