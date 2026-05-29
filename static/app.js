document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Dual-Lane Metrics
    const l1DensityEl = document.getElementById('l1-density');
    const l1StuckEl = document.getElementById('l1-stuck');
    const l1CongestionBadge = document.getElementById('l1-congestion-badge');
    
    const l2DensityEl = document.getElementById('l2-density');
    const l2StuckEl = document.getElementById('l2-stuck');
    const l2CongestionBadge = document.getElementById('l2-congestion-badge');
    
    const totalLiveCountEl = document.getElementById('total-live-count');
    const totalVehiclesEl = document.getElementById('total-vehicles');
    
    // Comparison Stats
    const adaptiveDelayBar = document.getElementById('adaptive-delay-bar');
    const adaptiveDelayVal = document.getElementById('adaptive-delay-val');
    const fixedDelayBar = document.getElementById('fixed-delay-bar');
    const fixedDelayVal = document.getElementById('fixed-delay-val');
    
    const adaptiveClearedEl = document.getElementById('adaptive-cleared');
    const fixedClearedEl = document.getElementById('fixed-cleared');
    const efficiencyGainBadge = document.getElementById('efficiency-gain-badge');

    // States and Timer
    const currentStateEl = document.getElementById('current-state');
    const fixedStateEl = document.getElementById('fixed-state');
    const timeLeftEl = document.getElementById('time-left');
    
    // Traffic Light Elements
    const l1Red = document.getElementById('l1-red');
    const l1Yellow = document.getElementById('l1-yellow');
    const l1Green = document.getElementById('l1-green');
    
    const l2Red = document.getElementById('l2-red');
    const l2Yellow = document.getElementById('l2-yellow');
    const l2Green = document.getElementById('l2-green');
    
    // Swarm Decisions Log
    const swarmLogsList = document.getElementById('swarm-logs-list');

    // Helper to update traffic lights based on coordinated state
    function updateCoordinatedLights(state) {
        // Reset all lights
        l1Red.classList.remove('active');
        l1Yellow.classList.remove('active');
        l1Green.classList.remove('active');
        
        l2Red.classList.remove('active');
        l2Yellow.classList.remove('active');
        l2Green.classList.remove('active');
        
        currentStateEl.textContent = state.replace('_', ' ');
        
        if (state === 'LANE1_GREEN') {
            l1Green.classList.add('active');
            l2Red.classList.add('active');
            currentStateEl.style.color = '#06b6d4'; // Cyan
        } else if (state === 'LANE1_YELLOW') {
            l1Yellow.classList.add('active');
            l2Red.classList.add('active');
            currentStateEl.style.color = 'var(--yellow-glow)';
        } else if (state === 'LANE2_GREEN') {
            l1Red.classList.add('active');
            l2Green.classList.add('active');
            currentStateEl.style.color = '#d946ef'; // Magenta
        } else if (state === 'LANE2_YELLOW') {
            l1Red.classList.add('active');
            l2Yellow.classList.add('active');
            currentStateEl.style.color = 'var(--yellow-glow)';
        }
    }

    // Helper to update congestion badges
    function updateCongestionBadge(el, status) {
        el.textContent = status;
        el.className = 'congestion-badge ' + status.toLowerCase();
    }
    
    // Initialize Chart.js with two lines
    const ctx = document.getElementById('densityChart').getContext('2d');
    
    // Cyan gradient for Lane 1
    const l1Gradient = ctx.createLinearGradient(0, 0, 0, 180);
    l1Gradient.addColorStop(0, 'rgba(6, 182, 212, 0.4)');
    l1Gradient.addColorStop(1, 'rgba(6, 182, 212, 0)');
    
    // Magenta gradient for Lane 2
    const l2Gradient = ctx.createLinearGradient(0, 0, 0, 180);
    l2Gradient.addColorStop(0, 'rgba(217, 70, 239, 0.4)');
    l2Gradient.addColorStop(1, 'rgba(217, 70, 239, 0)');
    
    const densityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(20).fill(''),
            datasets: [
                {
                    label: 'Lane 1 Queue',
                    data: Array(20).fill(0),
                    borderColor: '#06b6d4',
                    backgroundColor: l1Gradient,
                    borderWidth: 2.5,
                    pointRadius: 1,
                    pointBackgroundColor: '#06b6d4',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Lane 2 Queue',
                    data: Array(20).fill(0),
                    borderColor: '#d946ef',
                    backgroundColor: l2Gradient,
                    borderWidth: 2.5,
                    pointRadius: 1,
                    pointBackgroundColor: '#d946ef',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Warning Limit',
                    data: Array(20).fill(6),
                    borderColor: '#ef4444',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 10,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 10 }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 10, weight: '500' },
                        boxWidth: 8,
                        usePointStyle: true
                    }
                }
            },
            interaction: { intersect: false, mode: 'index' },
            animation: { duration: 150 }
        }
    });

    let prevLogsJson = "";

    // Fetch metrics from backend periodically
    async function fetchMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            
            // 1. Update Coordinated lane densities & stuck values
            l1DensityEl.textContent = data.lane1_density;
            l1StuckEl.textContent = data.lane1_stuck;
            updateCongestionBadge(l1CongestionBadge, data.lane1_congestion);
            
            l2DensityEl.textContent = data.lane2_density;
            l2StuckEl.textContent = data.lane2_stuck;
            updateCongestionBadge(l2CongestionBadge, data.lane2_congestion);
            
            totalLiveCountEl.textContent = data.total_live_vehicles;
            totalVehiclesEl.textContent = data.total_vehicles;
            
            // 2. Update comparative performance metrics
            const adWait = data.adaptive_total_wait || 0;
            const fxWait = data.fixed_total_wait || 0;
            
            adaptiveDelayVal.textContent = adWait.toFixed(1) + 's';
            fixedDelayVal.textContent = fxWait.toFixed(1) + 's';
            
            // Set widths for delay bars
            const maxVal = Math.max(adWait, fxWait, 5);
            fixedDelayBar.style.width = ((fxWait / maxVal) * 100) + '%';
            adaptiveDelayBar.style.width = ((adWait / maxVal) * 100) + '%';
            
            adaptiveClearedEl.textContent = data.vehicles_cleared_adaptive || 0;
            fixedClearedEl.textContent = data.vehicles_cleared_fixed || 0;
            
            efficiencyGainBadge.textContent = data.efficiency_gain.toFixed(1) + '%';
            
            // 3. Update States and Timer
            updateCoordinatedLights(data.state);
            fixedStateEl.textContent = data.fixed_state.replace('_', ' ');
            timeLeftEl.textContent = data.time_left.toFixed(1) + 's';
            
            // 4. Update Vehicle Breakdown counts
            if (data.live_counts && data.total_counts) {
                document.getElementById('count-cars').innerHTML = `${data.total_counts.car} <span class="total-badge live-badge">Live: ${data.live_counts.car}</span>`;
                document.getElementById('count-bikes').innerHTML = `${data.total_counts.motorcycle} <span class="total-badge live-badge">Live: ${data.live_counts.motorcycle}</span>`;
                document.getElementById('count-buses').innerHTML = `${data.total_counts.bus} <span class="total-badge live-badge">Live: ${data.live_counts.bus}</span>`;
                document.getElementById('count-trucks').innerHTML = `${data.total_counts.truck} <span class="total-badge live-badge">Live: ${data.live_counts.truck}</span>`;
                document.getElementById('count-pedestrians').innerHTML = `${data.total_counts.person} <span class="total-badge live-badge">Live: ${data.live_counts.person}</span>`;
            }
            
            // 5. Update Swarm decisions scroll logs
            if (data.decision_logs) {
                const logsJson = JSON.stringify(data.decision_logs);
                if (logsJson !== prevLogsJson) {
                    prevLogsJson = logsJson;
                    swarmLogsList.innerHTML = "";
                    data.decision_logs.forEach(log => {
                        const li = document.createElement('li');
                        li.className = 'log-li';
                        li.textContent = log;
                        swarmLogsList.appendChild(li);
                    });
                    // Scroll to bottom
                    swarmLogsList.parentElement.scrollTop = swarmLogsList.parentElement.scrollHeight;
                }
            }
            
            // 6. Update Dual Line Chart
            const l1Data = densityChart.data.datasets[0].data;
            l1Data.shift();
            l1Data.push(data.lane1_density);
            
            const l2Data = densityChart.data.datasets[1].data;
            l2Data.shift();
            l2Data.push(data.lane2_density);
            
            densityChart.update();
            
        } catch (error) {
            console.error("Error fetching metrics:", error);
        }
    }

    // Poll server every 500ms
    setInterval(fetchMetrics, 500);
});
