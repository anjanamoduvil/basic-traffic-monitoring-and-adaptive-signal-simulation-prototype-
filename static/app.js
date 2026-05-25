document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const vehicleCountEl = document.getElementById('vehicle-count');
    const warningBanner = document.getElementById('congestion-warning');
    const currentStateEl = document.getElementById('current-state');
    const timeLeftEl = document.getElementById('time-left');
    
    // Traffic Light Elements
    const lightRed = document.getElementById('light-red');
    const lightYellow = document.getElementById('light-yellow');
    const lightGreen = document.getElementById('light-green');
    
    // Helper to update traffic light UI
    function updateTrafficLight(state) {
        // Reset all lights
        lightRed.classList.remove('active');
        lightYellow.classList.remove('active');
        lightGreen.classList.remove('active');
        
        currentStateEl.textContent = state;
        
        // Activate current light
        if (state === 'RED') {
            lightRed.classList.add('active');
            currentStateEl.style.color = 'var(--red-glow)';
        } else if (state === 'YELLOW') {
            lightYellow.classList.add('active');
            currentStateEl.style.color = 'var(--yellow-glow)';
        } else if (state === 'GREEN') {
            lightGreen.classList.add('active');
            currentStateEl.style.color = 'var(--green-glow)';
        }
    }
    
    // Initialize Chart.js
    const ctx = document.getElementById('densityChart').getContext('2d');
    const densityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(20).fill(''), // X-axis labels
            datasets: [{
                label: 'Vehicles in ROI',
                data: Array(20).fill(0),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.2)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 10,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            },
            animation: {
                duration: 0
            }
        }
    });

    // Fetch metrics from backend periodically
    async function fetchMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            
            // Update Vehicle Count
            vehicleCountEl.textContent = data.vehicles_in_roi;
            
            // Update Warning Banner
            if (data.is_congested) {
                warningBanner.classList.remove('hidden');
            } else {
                warningBanner.classList.add('hidden');
            }
            
            // Update Traffic Light
            updateTrafficLight(data.state);
            
            // Update Time Left
            timeLeftEl.textContent = data.time_left.toFixed(1) + 's';
            
            // Update Vehicle Breakdown
            if (data.live_counts && data.total_counts) {
                document.getElementById('count-cars').innerHTML = `${data.live_counts.car} <span class="total-badge">Total: ${data.total_counts.car}</span>`;
                document.getElementById('count-bikes').innerHTML = `${data.live_counts.motorcycle} <span class="total-badge">Total: ${data.total_counts.motorcycle}</span>`;
                document.getElementById('count-buses').innerHTML = `${data.live_counts.bus} <span class="total-badge">Total: ${data.total_counts.bus}</span>`;
                document.getElementById('count-trucks').innerHTML = `${data.live_counts.truck} <span class="total-badge">Total: ${data.total_counts.truck}</span>`;
            }
            
            // Update Total Tracked Vehicles Count
            if (data.total_vehicles !== undefined) {
                document.getElementById('total-vehicles').textContent = data.total_vehicles;
            }
            
            // Update Chart
            const currentData = densityChart.data.datasets[0].data;
            currentData.shift(); // Remove oldest
            currentData.push(data.vehicles_in_roi); // Add newest
            densityChart.update();
            
        } catch (error) {
            console.error("Error fetching metrics:", error);
        }
    }

    // Poll every 500ms
    setInterval(fetchMetrics, 500);
});
