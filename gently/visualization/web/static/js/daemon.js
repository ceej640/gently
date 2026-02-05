/**
 * Daemon Observability Tab
 *
 * Arousal gauge (Canvas 2D), heartbeat timeline, queue depth chart,
 * category breakdown donut, and scrolling task flow table.
 */

const DaemonTab = {
    active: false,
    charts: {},          // Chart.js instances
    gaugeAnim: null,     // requestAnimationFrame id
    arousalTarget: 0,    // smoothed target
    arousalCurrent: 0,   // animated value
    taskRows: [],        // task flow rows (max 100)

    // Ring buffers (300 points = 5 min at 1/sec)
    MAX_POINTS: 300,
    queueHistory: { cognitive: [], physical: [], interaction: [], labels: [] },
    heartbeatData: [],   // {x: timestamp, y: type_index, cat, priority}

    // Category colours
    CAT_COLORS: {
        cognitive:   '#4a9eff',
        physical:    '#51cf66',
        interaction: '#cc5de8',
    },

    // Task type → Y-axis index for heartbeat scatter
    TYPE_INDEX: {
        observe: 0, synthesize: 1, predict: 2, compare: 3,
        reflect: 4, question: 5, retrieve: 6,
        image: 7, move: 8, configure: 9, calibrate: 10,
        surface: 11, ask: 12, notify: 13,
    },

    // ================================================================
    // Lifecycle
    // ================================================================

    init() {
        // Charts are created lazily on first activate
        this._setupDarkTheme();
    },

    activate() {
        if (this.active) return;
        this.active = true;

        // Create charts on first activation
        if (!this.charts.heartbeat) this._createCharts();

        // Start gauge animation
        this._startGaugeLoop();

        // Subscribe to daemon data
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type: 'subscribe_daemon' }));
        }

        // Pulse indicator
        const pulse = document.getElementById('daemon-pulse');
        if (pulse) pulse.classList.add('active');
    },

    deactivate() {
        if (!this.active) return;
        this.active = false;

        // Stop gauge animation
        if (this.gaugeAnim) {
            cancelAnimationFrame(this.gaugeAnim);
            this.gaugeAnim = null;
        }

        // Unsubscribe
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type: 'unsubscribe_daemon' }));
        }

        const pulse = document.getElementById('daemon-pulse');
        if (pulse) pulse.classList.remove('active');
    },

    // ================================================================
    // Data Handlers (called from websocket.js)
    // ================================================================

    handleDaemonStatus(data) {
        if (!data) return;

        // Status bar
        const aliveDot = document.getElementById('daemon-alive-dot');
        const schedDot = document.getElementById('daemon-scheduler-dot');
        if (aliveDot) aliveDot.className = 'daemon-status-dot ' + (data.alive ? 'alive' : 'dead');
        if (schedDot) schedDot.className = 'daemon-status-dot ' + (data.scheduler?.alive ? 'alive' : 'dead');

        const sessionEl = document.getElementById('daemon-session-id');
        if (sessionEl) sessionEl.textContent = data.session_id ? data.session_id.slice(0, 8) : '-';

        const userEl = document.getElementById('daemon-user-present');
        if (userEl) userEl.textContent = data.user_present ? 'Present' : 'Away';

        const execEl = document.getElementById('daemon-tasks-executed');
        if (execEl) execEl.textContent = data.scheduler?.tasks_executed ?? 0;

        // Clock / arousal data
        const clock = data.clock || {};
        this.arousalTarget = clock.arousal ?? 0;

        // Gauge stats
        this._setText('gs-pace', clock.pace_seconds != null ? clock.pace_seconds + 's' : '-');
        this._setText('gs-momentum', clock.arousal_momentum != null ? clock.arousal_momentum.toFixed(3) : '-');
        this._setText('gs-avg30', clock.arousal_avg_30s != null ? clock.arousal_avg_30s.toFixed(3) : '-');
        this._setText('gs-triggers', clock.pending_triggers ?? '-');
        this._setText('gs-next', clock.time_until_next != null ? clock.time_until_next.toFixed(1) + 's' : '-');
        this._setText('gs-deep', clock.can_deep_think != null ? (clock.can_deep_think ? 'Ready' : 'Cooldown') : '-');

        // Queue depth history
        const queue = data.scheduler?.queue || {};
        const pending = queue.pending_by_category || {};
        const now = new Date();
        const label = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

        this.queueHistory.labels.push(label);
        this.queueHistory.cognitive.push(pending.cognitive || 0);
        this.queueHistory.physical.push(pending.physical || 0);
        this.queueHistory.interaction.push(pending.interaction || 0);

        // Trim ring buffer
        if (this.queueHistory.labels.length > this.MAX_POINTS) {
            this.queueHistory.labels.shift();
            this.queueHistory.cognitive.shift();
            this.queueHistory.physical.shift();
            this.queueHistory.interaction.shift();
        }

        // Update queue depth chart
        if (this.charts.queueDepth) {
            const qd = this.charts.queueDepth;
            qd.data.labels = this.queueHistory.labels;
            qd.data.datasets[0].data = this.queueHistory.cognitive;
            qd.data.datasets[1].data = this.queueHistory.physical;
            qd.data.datasets[2].data = this.queueHistory.interaction;
            qd.update('none');
        }

        // Update category donut
        if (this.charts.category) {
            const cat = this.charts.category;
            const cogCount = pending.cognitive || 0;
            const physCount = pending.physical || 0;
            const intCount = pending.interaction || 0;
            const total = cogCount + physCount + intCount;
            cat.data.datasets[0].data = [cogCount, physCount, intCount];
            // Center text plugin
            cat.options.plugins.centerText = { text: String(total) };
            cat.update('none');
        }
    },

    handleDaemonTaskUpdate(data) {
        if (!data) return;

        // Add heartbeat dot
        const typeIdx = this.TYPE_INDEX[data.type] ?? 0;
        const cat = data.category || 'cognitive';
        const now = Date.now();

        this.heartbeatData.push({
            x: now,
            y: typeIdx,
            cat: cat,
            priority: data.priority || 50,
            status: data.status,
        });

        // Trim to 5 min
        const cutoff = now - 5 * 60 * 1000;
        this.heartbeatData = this.heartbeatData.filter(d => d.x > cutoff);

        // Update heartbeat chart
        if (this.charts.heartbeat) {
            this._updateHeartbeatChart();
        }

        // Add to task flow table
        this._addTaskRow(data);
    },

    handleDaemonTasks(tasks) {
        // Bulk load of recent tasks (on subscribe)
        if (!Array.isArray(tasks)) return;
        const tbody = document.getElementById('task-flow-tbody');
        if (!tbody) return;
        tbody.innerHTML = '';
        this.taskRows = [];
        // Show most recent at top
        for (const t of tasks) {
            this._addTaskRow(t, false);
        }
    },

    // ================================================================
    // Chart Creation
    // ================================================================

    _createCharts() {
        this._setupDarkTheme();
        this._createHeartbeatChart();
        this._createQueueDepthChart();
        this._createCategoryChart();
        // Populate heartbeat with any accumulated data
        if (this.heartbeatData.length > 0) {
            this._updateHeartbeatChart();
        }
    },

    _createHeartbeatChart() {
        const ctx = document.getElementById('heartbeat-chart');
        if (!ctx) return;

        const typeLabels = Object.keys(this.TYPE_INDEX);

        this.charts.heartbeat = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    { label: 'Cognitive', data: [], backgroundColor: this.CAT_COLORS.cognitive },
                    { label: 'Physical', data: [], backgroundColor: this.CAT_COLORS.physical },
                    { label: 'Interaction', data: [], backgroundColor: this.CAT_COLORS.interaction },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: { display: true, position: 'top', labels: { boxWidth: 8, padding: 8, font: { size: 10 } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const raw = ctx.raw;
                                const name = typeLabels[raw.y] || '?';
                                return `${name} (pri: ${raw.priority || '?'})`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        display: true,
                        ticks: {
                            callback: (v) => {
                                const d = new Date(v);
                                return d.toLocaleTimeString('en-US', { hour12: false, minute: '2-digit', second: '2-digit' });
                            },
                            maxTicksLimit: 8,
                            font: { size: 10 },
                        },
                        grid: { color: 'rgba(255,255,255,0.06)' },
                    },
                    y: {
                        type: 'linear',
                        min: -0.5,
                        max: 13.5,
                        ticks: {
                            stepSize: 1,
                            callback: (v) => typeLabels[v] || '',
                            font: { size: 9 },
                        },
                        grid: { color: 'rgba(255,255,255,0.06)' },
                    }
                }
            }
        });
    },

    _updateHeartbeatChart() {
        const chart = this.charts.heartbeat;
        if (!chart) return;

        const cog = [], phys = [], inter = [];
        for (const d of this.heartbeatData) {
            const pt = { x: d.x, y: d.y, priority: d.priority };
            if (d.cat === 'cognitive') cog.push(pt);
            else if (d.cat === 'physical') phys.push(pt);
            else inter.push(pt);
        }

        chart.data.datasets[0].data = cog;
        chart.data.datasets[1].data = phys;
        chart.data.datasets[2].data = inter;

        // Update x-axis range to rolling 5min
        const now = Date.now();
        chart.options.scales.x.min = now - 5 * 60 * 1000;
        chart.options.scales.x.max = now;

        // Scale point radius by priority
        chart.data.datasets.forEach(ds => {
            ds.pointRadius = ds.data.map(d => 3 + (d.priority / 100) * 5);
        });

        chart.update('none');
    },

    _createQueueDepthChart() {
        const ctx = document.getElementById('queue-depth-chart');
        if (!ctx) return;

        this.charts.queueDepth = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Cognitive',
                        data: [],
                        borderColor: this.CAT_COLORS.cognitive,
                        backgroundColor: this.CAT_COLORS.cognitive + '20',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 1.5,
                    },
                    {
                        label: 'Physical',
                        data: [],
                        borderColor: this.CAT_COLORS.physical,
                        backgroundColor: this.CAT_COLORS.physical + '20',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 1.5,
                    },
                    {
                        label: 'Interaction',
                        data: [],
                        borderColor: this.CAT_COLORS.interaction,
                        backgroundColor: this.CAT_COLORS.interaction + '20',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 1.5,
                    },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: { display: true, position: 'top', labels: { boxWidth: 8, padding: 8, font: { size: 10 } } },
                },
                scales: {
                    x: {
                        display: true,
                        ticks: { maxTicksLimit: 6, font: { size: 10 } },
                        grid: { color: 'rgba(255,255,255,0.06)' },
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 1, font: { size: 10 } },
                        grid: { color: 'rgba(255,255,255,0.06)' },
                    }
                }
            }
        });
    },

    _createCategoryChart() {
        const ctx = document.getElementById('category-chart');
        if (!ctx) return;

        // Center text plugin (registered per-chart)
        const centerTextPlugin = {
            id: 'centerText',
            afterDraw(chart) {
                const text = chart.options.plugins.centerText?.text;
                if (text == null) return;
                const { ctx, chartArea } = chart;
                const centerX = (chartArea.left + chartArea.right) / 2;
                const centerY = (chartArea.top + chartArea.bottom) / 2;
                ctx.save();
                ctx.font = 'bold 20px sans-serif';
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text') || '#e0e0e0';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(text, centerX, centerY);
                ctx.restore();
            }
        };

        this.charts.category = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Cognitive', 'Physical', 'Interaction'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        this.CAT_COLORS.cognitive,
                        this.CAT_COLORS.physical,
                        this.CAT_COLORS.interaction,
                    ],
                    borderWidth: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                animation: false,
                plugins: {
                    legend: { display: true, position: 'bottom', labels: { boxWidth: 8, padding: 8, font: { size: 10 } } },
                    centerText: { text: '0' },
                },
            },
            plugins: [centerTextPlugin],
        });
    },

    // ================================================================
    // Arousal Gauge (Canvas 2D)
    // ================================================================

    _startGaugeLoop() {
        const draw = () => {
            if (!this.active) return;
            // Ease towards target
            this.arousalCurrent += (this.arousalTarget - this.arousalCurrent) * 0.08;
            this._drawGauge(this.arousalCurrent);
            this.gaugeAnim = requestAnimationFrame(draw);
        };
        this.gaugeAnim = requestAnimationFrame(draw);
    },

    _drawGauge(value) {
        const canvas = document.getElementById('arousal-gauge');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // HiDPI support
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            canvas.style.width = w + 'px';
            canvas.style.height = h + 'px';
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        const cx = w / 2;
        const cy = h * 0.85;
        const radius = Math.min(w / 2, h * 0.7) - 8;
        const startAngle = Math.PI;
        const endAngle = 2 * Math.PI;

        // Background arc
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, endAngle);
        ctx.lineWidth = 14;
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineCap = 'round';
        ctx.stroke();

        // Gradient arc (blue → yellow → red)
        const grad = ctx.createLinearGradient(cx - radius, cy, cx + radius, cy);
        grad.addColorStop(0, '#4a9eff');
        grad.addColorStop(0.5, '#ffd43b');
        grad.addColorStop(1, '#ff6b6b');

        const val = Math.max(0, Math.min(1, value));
        const sweepAngle = startAngle + val * Math.PI;

        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, sweepAngle);
        ctx.lineWidth = 14;
        ctx.strokeStyle = grad;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Needle
        const needleAngle = startAngle + val * Math.PI;
        const needleLen = radius - 20;
        const nx = cx + Math.cos(needleAngle) * needleLen;
        const ny = cy + Math.sin(needleAngle) * needleLen;

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(nx, ny);
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--text') || '#e0e0e0';
        ctx.lineCap = 'round';
        ctx.stroke();

        // Center dot
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text') || '#e0e0e0';
        ctx.fill();

        // Value text
        ctx.font = 'bold 16px monospace';
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text') || '#e0e0e0';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(val.toFixed(3), cx, cy + 8);

        // Min/Max labels
        ctx.font = '10px sans-serif';
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-muted') || '#888';
        ctx.textAlign = 'left';
        ctx.fillText('0', cx - radius - 4, cy + 4);
        ctx.textAlign = 'right';
        ctx.fillText('1', cx + radius + 4, cy + 4);
    },

    // ================================================================
    // Task Flow Table
    // ================================================================

    _addTaskRow(data, animate = true) {
        const tbody = document.getElementById('task-flow-tbody');
        if (!tbody) return;

        // Check for existing row with same task id — update status
        const existingRow = tbody.querySelector(`tr[data-task-id="${data.id}"]`);
        if (existingRow) {
            const statusCell = existingRow.querySelector('.task-status-badge');
            if (statusCell) {
                statusCell.className = 'task-status-badge ' + (data.status || 'pending');
                statusCell.textContent = data.status || 'pending';
            }
            return;
        }

        const tr = document.createElement('tr');
        tr.setAttribute('data-task-id', data.id);
        if (animate) tr.className = 'new-row';

        const time = data.started_at || data.created_at || new Date().toISOString();
        const timeStr = new Date(time).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

        const cat = data.category || 'cognitive';
        const priority = data.priority || 50;
        const priFrac = priority / 100;
        const priColor = priFrac > 0.7 ? '#ff6b6b' : priFrac > 0.4 ? '#ffd43b' : '#4a9eff';

        tr.innerHTML = `
            <td>${timeStr}</td>
            <td>${data.type || '-'}</td>
            <td><span class="cat-indicator ${cat}"></span>${cat.slice(0,3)}</td>
            <td title="${data.target || ''}">${data.target || '-'}</td>
            <td><span class="priority-bar"><span class="priority-bar-fill" style="width:${priFrac * 100}%;background:${priColor}"></span></span></td>
            <td><span class="task-status-badge ${data.status || 'pending'}">${data.status || 'pending'}</span></td>
            <td title="${data.reason || ''}">${data.reason || '-'}</td>
        `;

        // Insert at top
        if (tbody.firstChild) {
            tbody.insertBefore(tr, tbody.firstChild);
        } else {
            tbody.appendChild(tr);
        }

        this.taskRows.unshift(tr);

        // Max 100 visible
        while (this.taskRows.length > 100) {
            const old = this.taskRows.pop();
            old.remove();
        }
    },

    // ================================================================
    // Helpers
    // ================================================================

    _setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    },

    _setupDarkTheme() {
        if (typeof Chart === 'undefined') return;
        // Chart.js global defaults for dark theme
        Chart.defaults.color = getComputedStyle(document.documentElement).getPropertyValue('--text-muted') || '#888';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.08)';
    },
};
